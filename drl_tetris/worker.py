import tensorflow.compat.v1 as tf
import numpy as np
import struct
import time
import os
import logging

from drl_tetris.utils.timekeeper import timekeeper
from drl_tetris.utils.logging import logstamp
from drl_tetris.utils.tb_writer import tb_writer
from drl_tetris.runner import runner

import threads

logger = logging.getLogger(__name__)

class worker(runner):
    def __init__(self, settings):
        super().__init__(settings)
        self.config = tf.ConfigProto(
            log_device_placement=False,
            device_count={'GPU': 0}
        )
        # self.config.gpu_options.allow_growth = True
        self.print_freq            = 1000
        self.next_print            = 0
        self.stashed_experience    = None

    def read_settings(self):
        self.worker_agent_type     = self.settings["agent_type"]
        self.n_games               = self.settings["n_envs_per_thread"]
        self.n_players             = self.settings["n_players"]
        self.env_vector_type       = self.settings["env_vector_type"]
        self.env_type              = self.settings["env_type"]
        self.single_policy         = self.settings["single_policy"]
        self.run_standalone        = self.settings["run_standalone"]

    def set_runner_state(self, state):
        [self.env, self.worker_agent] = state

    def get_runner_state(self):
        return [self.env, self.worker_agent]

    def create_runner_state(self):
        self.env = self.env_vector_type(
            self.n_games,
            self.env_type,
            settings=self.settings,
        )
        self.worker_agent = self.worker_agent_type(
            self.n_games,
            id=self.training_state.me,
            settings=self.settings,
            sandbox=self.env_type(settings=self.settings),
            mode=threads.WORKER,
        )

    def validation_artifact(self):
        artefact = [self.env.get_state(), self.worker_agent.export_weights()]
        return artefact  # This is the state we will perform validation on

    def runner_validation(self, artefact):  # Successful recovery entails being able to reproduce the exact same nn outputs
        state, weights = artefact
        self.worker_agent.import_weights(weights)
        return self.worker_agent.get_action(
            state,
            player=[0]*len(state),
            # player=[0]*self.n_games,
            )[2]

    @logstamp(logger.info, on_exit=True)
    def graceful_exit(self):
        self.training_state.alive_flag.unset()

    def run(self):
        with tf.Session(config=self.config) as session:
            with tb_writer("worker", session) as self.tb_writer:
                ### Initialize session
                self.worker_agent.create_models(session)
                self.validate_runner()

                ### Initialize main-loop variables
                reset_list = [i for i in range(self.n_games)]
                current_player = np.random.choice(
                    [i for i in range(self.n_players if self.single_policy else 1)],
                    size=(self.n_games),
                )

                ### Run!
                try:
                    while not self.received_interrupt:
                        self.update_clock()
                        current_weights_idx = self.update_weights()

                        # Who's turn, and what state do they see?
                        current_player = 1 - current_player
                        state = self.env.get_state()

                        # What action do they want to perform?
                        action_idx, action, _ = self.worker_agent.get_action(state, player=current_player, training=True, random_action=current_weights_idx<1)

                        #Perform action!
                        reward, done = self.env.perform_action(action, player=current_player)
                        s_prime = self.env.get_state()

                        #Record what just happened!
                        experience = self.make_experience(state, action_idx, reward, s_prime, current_player, done)
                        self.worker_agent.store_experience(experience)

                        #If some game has reached termingal state, it's reset here. Agents also get a chance to update their internals...
                        reset_list = self.reset_envs(done, reward, current_player)
                        self.worker_agent.ready_for_new_round(training=True, env=reset_list)

                        # Get data back and forth to the trainer!
                        self.send_training_data()

                        #Print
                        self.print_stats()

                except Exception as e:
                    logger.error(f"worker died")
                    raise e

    @timekeeper()
    def update_clock(self):
        self.training_state.alive_flag.set(expire=10)
        self.training_state.workers_clock.tick(self.n_games)

    @timekeeper()
    @logstamp(logger.info, only_new=True)
    def update_weights(self):
        index = self.training_state.trainer_weights_index.get()
        current_index = self.training_state.weights_index.get()
        if index > current_index:
            logger.info(f"updating weights: {current_index} -> {index}")
            found, weights = self.training_state.trainer_weights.get()
            if found:
                self.worker_agent.import_weights(weights)
            self.training_state.weights_index.set(index)
        return index

    @timekeeper()
    def send_training_data(self):
        data = self.worker_agent.transfer_data()
        if len(data) == 0 or self.run_standalone:
            return
        self.training_state.data_queue.push(data)

    def print_stats(self):
        clock = self.training_state.workers_clock.get()
        self.training_state.stats.update(timekeeper.stats)
        # self.training_state.stats.update({'runner': {'current_weights_index': index}}, update_op="set")
        if clock > self.next_print:
            self.next_print = clock + self.print_freq
            logger.info(f"worker-clock: {clock}")

    def reset_envs(self, done, reward, current_player):
        #Reset the envs that reach terminal states
        reset_list = [ idx for idx,d in enumerate(done) if d]
        self.env.reset(env=reset_list)
        if self.single_policy:
            return reset_list
        #we delay the signal to the agent 1 time-step, since it's experiences are delayed as much.
        old_reset = self.old_reset_list
        self.old_reset_list = reset_list
        return old_reset

    def make_experience(self, state, action_idx, reward, s_prime, current_player, done):
        # This is done in a function just in case we are using 2 policies, because then we need to work to reconstruct the current experience using info from the previous one
        experience = [state, action_idx, reward, s_prime, current_player, done]
        if not self.single_policy:
            experience = self.merge_from_stash(experience, current_player[0])
            self.stashed_experience = experience
        return experience

    def merge_from_stash(self, new, player):
        # merge_experiences and stash is used when training 2 policies against each other.
        # It makes each agent see the other as part of the environment: (s0,s1,s2, ...) -> (s0,s2,s4,..), (s1,s3,s5,...) and similarly for s', r, done etc
        if self.stashed_experience is None:
            return [[None for _ in range(self.settings["n_envs_per_thread"])] for _ in range(6)]
        old_s, old_a, old_r, old_sp, old_p, old_d  = self.stashed_experience
        new_s,new_a,new_r,new_sp,new_p,new_d = new
        #In 2-policy-mode the current player completes the previous players experience-tuple, and reports it for them!
        experience = [
            old_s,  #s_(t-1)
            old_a,  #a_(t-1)
            [old - new for old, new in zip(old_r,new_r)],  #r_(t-1) - r_t ::: Good news for current player shows as bad news for the previous one.
            new_sp, #s_(t+1)
            old_p,  #previous player
            [x or y for x,y in zip(new_d,old_d)], #if either player's move caused the round to end, it should be seen  here!
        ]
        return experience
