import logging
import random
import pickle
import random
import numpy as np
import tensorflow as tf
import collections

import threads
import aux
import aux.utils as utils
import agents.vector_agent as va
import agents.agent_utils as agent_utils
from agents.tetris_agent import tetris_agent
from agents.networks import value_net
from aux.parameter import *


class vector_agent(va.vector_agent_base):
    def __init__(self, n_envs, id=0, session=None, sandbox=None, trajectory_queue=None, settings=None, mode=threads.STANDALONE):
        #Some general variable initialization etc...
        va.vector_agent_base.__init__(self, n_envs, id=id, session=session, sandbox=sandbox, trajectory_queue=trajectory_queue, settings=settings, mode=mode)
        #Logger
        self.log = logging.getLogger("vector_agent")
        self.log.debug("vector_agent created! mode={}".format(self.mode))

        #If we do our own training, we prepare for that
        if self.mode is threads.STANDALONE:
            self.trainer = vector_agent.trainer(settings=settings, sandbox=sandbox, mode=threads.PASSIVE)
            self.experience_replay = trainer.experience_replay
            #STANDALONE agents have to keep track of their own training habits!
            self.time_to_training = self.settings['time_to_training']

        if self.mode is threads.WORKER:
            self.experience_replay = agent_utils.experience_replay(self.settings["experience_replay_size"], prioritized=self.settings["prioritized_experience_replay"])
            self.time_to_send_data = self.settings["worker_data_send_fequency"]

        if self.mode in [threads.WORKER, threads.STANDALONE]: #This means everyone :)
            self.current_trajectory = [agent_utils.trajectory(self.state_size) for _ in range(self.n_envs)]
            self.avg_trajectory_length = 5 #tau is initialized to something...
            #Create models
            self.extrinsic_model =           value_net(self.id, "main_extrinsic",      self.state_size, session, settings=self.settings, on_cpu=self.settings["worker_net_on_cpu"])
            self.model_dict = {"extrinsic_model" : self.extrinsic_model}


    # # # # #
    # Agent interface fcns
    # # #

    #
    ###
    #####
    def get_action(self, state_vec, player=None, training=False, verbose=False):
        #Get hypothetical future states and turn them into vectors!
        p_list = utils.parse_arg(player, self.player_idxs)
        player_pairs_flat, future_states, future_states_flat = [], {}, []
        all_actions = [None for _ in range(len(state_vec))]

        #Simulate future states
        for state_idx, state in enumerate(state_vec):
            self.sandbox.set(state)
            player_action            = self.sandbox.get_actions(                    player=p_list[state_idx])
            future_states[state_idx] = self.sandbox.simulate_actions(player_action, player=p_list[state_idx])
            all_actions[state_idx] = player_action
            player_pairs_flat += [ (p_list[state_idx],1-p_list[state_idx]) for _ in range(len(player_action)) ]

        #Flatten (kinda like ravel for np.arrays)
        unflatten_dict, temp = {}, 0
        for state_idx in range(len(state_vec)):
            future_states_flat += future_states[state_idx]
            unflatten_dict[state_idx] = slice(temp,temp+len(future_states[state_idx]),1)

        #Run model!
        extrinsic_values_flat, _ = self.run_model(self.extrinsic_model, future_states_flat, player_lists=player_pairs_flat)
        values_flat = -extrinsic_values_flat

        #Undo flatten
        values = [values_flat[unflatten_dict[state_idx]] for state_idx in range(len(state_vec))]
        argmax = [ np.argmax( values_flat[unflatten_dict[state_idx]]) for state_idx in range(len(state_vec))]

        #Epsilon rolls
        action_idxs = argmax
        if training:
            for state_idx in range(len(state_vec)):
                if "boltzmann" in self.settings["dithering_scheme"]:
                    assert False, "use adaptive_epsilon as your dithering scheme with this agent!"
                if self.settings["dithering_scheme"] == "adaptive_epsilon":
                    dice = random.random()
                    #if random, change the action
                    if dice < self.settings["epsilon"].get_value(self.n_train_steps) * self.avg_trajectory_length**(-1):
                        a_idx = np.random.choice(np.arange(values[state_idx].size))
                        action_idxs[state_idx] = a_idx
        actions = [all_actions[state_idx][action_idxs[state_idx]] for state_idx in range(len(state_vec)) ]

        #Keep the clock going...
        if training:
            self.n_train_steps += 1
        return action_idxs, actions

    #
    ###
    #####
    def ready_for_new_round(self,training=False, env=None):
        e_idxs, _ = utils.parse_arg(env, self.env_idxs, indices=True)
        for e in e_idxs:
            a = self.settings["tau_learning_rate"]
            if len(self.current_trajectory[env]) > 0 and training:
                self.avg_trajectory_length = (1-a) * self.avg_trajectory_length + a*len(self.current_trajectory[env])
        if not training: #If we for some reason call this method even while not training.
            return

        # Preprocess the trajectories specifiel to prepare them for training
        for e in e_idxs:
            #Process trajectory to get some advantages etc...
            prio, data = self.current_trajectory[e].process_trajectory(self.extrinsic_model)
            # data = utils.merge_lists(*samples)

            #Add data to experience replay sorted by surprise factor
            self.experience_replay.add_samples(data, prio)

            #Increment some counters to guide what we do
            if self.mode is threads.STANDALONE:
                self.time_to_training -= len(self.current_trajectory[e])
            if self.mode is threads.WORKER:
                self.time_to_send_data -= len(self.current_trajectory[e])
            #Clear trajectory
            self.current_trajectory[e] = agent_utils.trajectory(self.state_size)

        #Standalone agents have to keep track of their training habits!
        if self.mode is threads.STANDALONE:
            if self.time_to_training < 1:
                self.trainer.do_training()
                self.time_to_training = self.settings['time_to_training']

    #
    ###
    #####
    def run_model(self, net, states, player_lists=None):
        if isinstance(states, np.ndarray):
            if player_list is not None: self.log.warning("run_model was called with an np.array as an argument, and non-None player list. THIS IS NOT MENT TO BE, AND IF YOU DONT KNOW WHAT YOU ARE DOING, EXPECT INCORRECT RESULTS!")
            states_vector = states
        else:
            states_vector = self.states_to_vectors(states, player_lists=player_lists)
        return net.evaluate(states_vector)

    def store_experience(self, experience):
        #Turn a list of experience ingredients into one list of experiences:
        es = utils.merge_lists(*experience)
        for i,e in enumerate(es):
            self.current_trajectory[i].add(e)
        self.log.debug("agent[{}] appends experience {} to its trajectory-buffer".format(self.id, experience))
