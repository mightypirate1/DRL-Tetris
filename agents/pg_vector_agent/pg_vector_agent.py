import random
import random
import numpy as np
import tensorflow as tf
from scipy.special import softmax

import threads
import aux
import aux.utils as utils
from agents.agent_utils import state_fcns
from agents.pg_vector_agent.pg_vector_agent_base import pg_vector_agent_base
from agents.pg_vector_agent.pg_vector_agent_trainer import pg_vector_agent_trainer
import agents.agent_utils as agent_utils
import agents.datatypes as dt
from agents.tetris_agent import tetris_agent
from agents.networks import pg_net
from aux.parameter import *

class pg_vector_agent(pg_vector_agent_base):
    def __init__(
                 self,
                 n_envs,                    # How many envs in the vector-env?
                 id=0,                      # What's this trainers name?
                 session=None,              # The session to operate in
                 sandbox=None,              # Sandbox to play in!
                 mode=threads.STANDALONE,   # What's our role?
                 settings=None,             # Settings-dict passed down from the ancestors
                ):

        #Some general variable initialization etc...
        pg_vector_agent_base.__init__(self, id=id, name="worker{}".format(id), session=session, sandbox=sandbox, settings=settings, mode=mode)
        #Helper variables
        self.env_idxs = [i for i in range(n_envs)]
        self.n_envs = n_envs

        #In any mode, we need a place to store transitions!
        self.current_trajectory = [dt.pg_trajectory(self.settings["n_actions"], self.dummy_state.shape) for _ in range(self.n_envs)]
        self.stored_trajectories = list()
        self.avg_trajectory_length = 9 #tau is initialized to something...

        #If we do our own training, we prepare for that
        if self.mode is threads.STANDALONE:
            #Create a trainer, and link their neural-net and experience-replay to us
            self.trainer = self.settings["trainer_type"](id="trainer_{}".format(self.id),settings=settings, session=session, sandbox=sandbox, mode=threads.PASSIVE)
            self.extrinsic_model = self.model_dict["default"] = self.trainer.extrinsic_model
            #STANDALONE agents have to keep track of their own training habits!
            self.time_to_training = self.settings['time_to_training']

        if self.mode is threads.WORKER: #If we are a WORKER, we bring our own equipment
            #Create models
            self.extrinsic_model = pg_net(
                                           self.id,
                                           "main_extrinsic",
                                           self.state_size,
                                           session,
                                           settings=self.settings,
                                          )
            self.model_dict = {
                                "extrinsic_model" : self.extrinsic_model,
                                "default"         : self.extrinsic_model,
                              }

    # # # # #
    # Agent interface fcns
    # # #

    #
    ###
    #####
    def get_action(self, state_vec, player=None, training=False, verbose=False):
        #Get hypothetical future states and turn them into vectors!
        p_list = utils.parse_arg(player, self.player_idxs)
        player_list_flat   = [None for _ in range(len(state_vec))]
        future_states_mask = [None for _ in range(len(state_vec))]
        future_states_raw  = {}
        future_states      = [np.concatenate([self.dummy_state for _ in range(self.settings["n_actions"])],axis=0) for _ in range(len(state_vec))]
        all_actions        = [None for _ in range(len(state_vec))]
        actions            = [None for _ in range(len(state_vec))]
        action_idxs        = [None for _ in range(len(state_vec))]

        #Simulate future states
        for state_idx, state in enumerate(state_vec):
            self.sandbox.set(state)
            player_action                = self.sandbox.get_actions(                    player=p_list[state_idx])
            future_states_raw[state_idx] = self.sandbox.simulate_actions(player_action, player=p_list[state_idx])
            all_actions[state_idx]       = player_action
        for state_idx in range(len(state_vec)):
            n = len(future_states_raw[state_idx])
            p = [player[state_idx] for _ in range(n)]
            future_states[state_idx][:n,:] = state_fcns.states_from_perspective(future_states_raw[state_idx], player=p)
            future_states_mask[state_idx] = self.maskmaker(n)

        #Run model!
        probs, _ = self.run_model(self.extrinsic_model, future_states, future_states_mask)
        for state_idx in range(len(state_vec)):
            p = probs[state_idx]
            if training:
                a_idx = np.random.choice(np.arange(self.settings["n_actions"]), p=p)
            else:
                a_idx = p.ravel().argmax()
            action_idxs[state_idx] = a_idx
            actions[state_idx] = all_actions[state_idx][a_idx]

        #Keep the clock going...
        if training:
            self.clock += self.n_envs
        return action_idxs, actions

    def maskmaker(self, n):
        size = self.settings["n_actions"]
        return (np.arange(size)<n).astype(np.float)
    #
    ###
    #####
    def ready_for_new_round(self, training=False, env=None):
        e_idxs, _ = utils.parse_arg(env, self.env_idxs, indices=True)
        for e in e_idxs:
            a = self.settings["tau_learning_rate"]
            if len(self.current_trajectory[env]) > 0 or training is False:
                self.avg_trajectory_length = (1-a) * self.avg_trajectory_length + a*len(self.current_trajectory[env])

        # Preprocess the trajectories specifiel to prepare them for training
        for e in e_idxs:
            if training:
                self.stored_trajectories.append(
                                                self.current_trajectory[e].process_trajectory(
                                                                                              self.run_default_model,
                                                                                              self.settings["n_actions"],
                                                                                              sandbox=self.sandbox,
                                                                                              state_fcn=state_fcns.states_from_perspective,
                                                                                              maskmaker=self.maskmaker,
                                                                                             )
                                                )
                #Increment some counters to guide what we do
                if self.mode is threads.STANDALONE:
                    self.time_to_training  -= len(self.current_trajectory[e])
            #Clear trajectory
            self.current_trajectory[e] = dt.pg_trajectory(self.settings["n_actions"], self.dummy_state.shape)

        #Standalone agents have to keep track of their training habits!
        if training and self.mode is threads.STANDALONE:
            if self.time_to_training < 1:
                self.trainer.receive_data(self.transfer_data())
                new_prio, filter = self.trainer.do_training()
                self.time_to_training = self.settings['time_to_training']

    #
    ###
    #####
    def store_experience(self, experience, env=None):
        if env is not None:
            assert type(env) is int, "This funtion only works for env=None or env=integer"
            #We assume this is only ever used for the last state.
            self.current_trajectory[env].add(experience, end_of_trajectory=True)
            return
        #Turn a list of experience ingredients into one list of experiences:
        es = utils.merge_lists(*experience)
        for i,e in enumerate(es):
            self.current_trajectory[i].add(e)
        self.log.debug("agent[{}] appends experience {} to its trajectory-buffer".format(self.id, experience))

    def transfer_data(self, keep_data=False):
        #This function gives away the data gathered
        ret = self.stored_trajectories
        if not keep_data:
            self.stored_trajectories = list()
        return ret
