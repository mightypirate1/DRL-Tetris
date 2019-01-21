import random
import random
import numpy as np
import tensorflow as tf
from scipy.special import softmax

import threads
import aux
import aux.utils as utils
from agents.vector_agent.vector_agent_base import vector_agent_base
from agents.vector_agent.vector_agent_trainer import vector_agent_trainer
import agents.agent_utils as agent_utils
import agents.datatypes as dt
from agents.tetris_agent import tetris_agent
from agents.networks import prio_vnet
from aux.parameter import *

class vector_agent(vector_agent_base):
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
        vector_agent_base.__init__(self, id=id, name="worker{}".format(id), session=session, sandbox=sandbox, settings=settings, mode=mode)
        #Helper variables
        self.env_idxs = [i for i in range(n_envs)]
        self.n_envs = n_envs

        #In any mode, we need a place to store transitions!
        self.current_trajectory = [dt.trajectory() for _ in range(self.n_envs)]
        self.stored_trajectories = list()
        self.avg_trajectory_length = 9 #tau is initialized to something...

        #If we do our own training, we prepare for that
        if self.mode is threads.STANDALONE:
            #Create a trainer, and link their neural-net and experience-replay to us
            self.trainer = self.settings["trainer_type"](id="trainer_{}".format(self.id),settings=settings, session=session, sandbox=sandbox, mode=threads.PASSIVE)
            self.extrinsic_model = self.model_dict["default"] = self.trainer.extrinsic_model
            self.experience_replay                            = self.trainer.experience_replay
            #STANDALONE agents have to keep track of their own training habits!
            self.time_to_training = self.settings['time_to_training']

        if self.mode is threads.WORKER: #If we are a WORKER, we bring our own equipment
            #Create models
            self.extrinsic_model = prio_vnet(
                                             self.id,
                                             "main_extrinsic",
                                             self.state_size,
                                             session,
                                             settings=self.settings,
                                             on_cpu=self.settings["worker_net_on_cpu"]
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
        player_list_flat, future_states, future_states_flat = [], {}, []
        all_actions = [None for _ in range(len(state_vec))]

        #Simulate future states
        for state_idx, state in enumerate(state_vec):
            self.sandbox.set(state)
            player_action            = self.sandbox.get_actions(                    player=p_list[state_idx])
            future_states[state_idx] = self.sandbox.simulate_actions(player_action, player=p_list[state_idx])
            all_actions[state_idx]   = player_action
            player_list_flat        += [ 1-p_list[state_idx] for _ in range(len(player_action)) ] #View the state from the perspective of the enemy!

        #Flatten (kinda like ravel for np.arrays)
        unflatten_dict, temp = {}, 0
        for state_idx in range(len(state_vec)):
            future_states_flat += future_states[state_idx]
            unflatten_dict[state_idx] = slice(temp,temp+len(future_states[state_idx]),1)

        #Run model!
        extrinsic_values_flat = self.run_model(self.extrinsic_model, future_states_flat, player=player_list_flat)
        values_flat = -extrinsic_values_flat #We flip the sign, since this is the evaluation from the perspective of our opponent!

        #Undo flatten
        values = [values_flat[unflatten_dict[state_idx]] for state_idx in range(len(state_vec))]
        argmax = [ np.argmax( values_flat[unflatten_dict[state_idx]]) for state_idx in range(len(state_vec))]

        #Epsilon rolls
        action_idxs = argmax
        if training:
            for state_idx in range(len(state_vec)):
                if "distribution" in self.settings["dithering_scheme"]:
                    if "boltzman" in self.settings["dithering_scheme"]:
                        theta = self.settings["action_temperature"].get_value(self.clock)
                        p = softmax(theta*values[state_idx]).ravel()
                    elif "pareto" in self.settings["dithering_scheme"]:
                        theta = self.avg_trajectory_length * 4 / self.settings["game_area"]
                        p = utils.pareto(values[state_idx], temperature=theta)
                    a_idx = np.random.choice(np.arange(values[state_idx].size), p=p)
                if self.settings["dithering_scheme"] == "adaptive_epsilon":
                    dice = random.random()
                    #if random, change the action
                    if dice < self.settings["epsilon"].get_value(self.clock) * self.avg_trajectory_length**(-1):
                        a_idx = np.random.choice(np.arange(values[state_idx].size))
                        action_idxs[state_idx] = a_idx
        actions = [all_actions[state_idx][action_idxs[state_idx]] for state_idx in range(len(state_vec)) ]

        #Keep the clock going...
        if training:
            self.clock += self.n_envs
        return action_idxs, actions

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
                self.stored_trajectories.append(self.current_trajectory[e])
                #Increment some counters to guide what we do
                if self.mode is threads.STANDALONE:
                    self.time_to_training  -= len(self.current_trajectory[e])
            #Clear trajectory
            self.current_trajectory[e] = dt.trajectory()

        #Standalone agents have to keep track of their training habits!
        if training and self.mode is threads.STANDALONE:
            if self.time_to_training < 1:
                self.trainer.receive_data(self.stored_trajectories)
                new_prio, filter = self.trainer.do_training()
                self.stored_trajectories = list()
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
