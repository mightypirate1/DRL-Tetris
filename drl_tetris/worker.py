import tensorflow.compat.v1 as tf
import numpy as np
import struct
import time
import os
import logging

from drl_tetris.training_state import training_state
from drl_tetris.utils.training_utils import timekeeper

import threads
from tools.tf_hooks import quick_summary
import tools.utils as utils

logger = logging.getLogger(__name__)

class worker:
    def __init__(self, settings):
        self.settings = utils.parse_settings(settings)
        self.training_state = training_state()
        self.config = tf.ConfigProto(
            log_device_placement=False,
            device_count={'GPU': 0}
        )
        # self.config.gpu_options.allow_growth = True
        self.print_freq            = 1000
        self.next_print            = 0
        self.current_weights_index = -1
        self.stashed_experience    = None
        self.worker_agent_type     = self.settings["agent_type"]
        self.n_games               = self.settings["n_envs_per_thread"]
        self.n_players             = self.settings["n_players"]
        self.env_vector_type       = self.settings["env_vector_type"]
        self.env_type              = self.settings["env_type"]
        self.single_policy         = self.settings["single_policy"]
        self.run_standalone        = self.settings["run_standalone"]

    def run(self):
        with tf.Session(config=self.config) as session:
            self.quick_summary = quick_summary(settings=self.settings, session=session)
            #Initialize env and agents!
            self.env = self.env_vector_type(
                self.n_games,
                self.env_type,
                settings=self.settings
            )
            self.worker_agent = self.worker_agent_type(
                self.n_games,
                id=self.training_state.me,
                settings=self.settings,
                session=session,
                sandbox=self.env_type(settings=self.settings),
                mode=threads.WORKER,
            )

            ### Run!
            s_prime = self.env.get_state()
            current_player = np.random.choice([i for i in range(self.n_players)], size=(self.n_games))
            if not self.single_policy:
                current_player *= 0
            reset_list = [i for i in range(self.n_games)]
            try:
                while True:
                    self.update_weights_and_clock()

                    # Who's turn, and what state do they see?
                    current_player = 1 - current_player
                    state = self.env.get_state()

                    # What action do they want to perform?
                    action_idx, action = self.worker_agent.get_action(state, player=current_player, training=True, random_action=self.current_weights_index<1)

                    #Perform action!
                    reward, done = self.env.perform_action(action, player=current_player)
                    s_prime = self.env.get_state()

                    #Record what just happened!
                    experience = self.make_experience(state, action_idx, reward, s_prime, current_player, done)
                    self.worker_agent.store_experience(experience)

                    #Render!
                    #self.env.render() #unless turned off in settings

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
    def update_weights_and_clock(self):
        self.training_state.tick_worker_clock(self.n_games)
        found, index, weights = self.training_state.get_weights(newer_than=self.current_weights_index)
        if found:
            self.worker_agent.import_weights(weights)

    @timekeeper()
    def send_training_data(self):
        data = self.worker_agent.transfer_data()
        if len(data) == 0 or self.run_standalone:
            return
        self.training_state.push_worker_data(data)

    def print_stats(self):
        clock = self.training_state.get_worker_clock()
        self.training_state.increment_stats(timekeeper.stats)
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
