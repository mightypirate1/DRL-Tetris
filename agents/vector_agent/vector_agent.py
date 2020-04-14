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
                 n_workers=1,               # How many workers run in parallel? If you don't know, guess it's just 1
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
        self.n_workers = n_workers

        self.send_count = 0

        #In any mode, we need a place to store transitions!
        self.trajectory_type = dt.trajectory if self.settings["single_policy"] else dt.trajectory_dualpolicy
        self.current_trajectory = [self.trajectory_type() for _ in range(self.n_envs if self.settings["single_policy"] else 2*self.n_envs)]
        self.stored_trajectories = list()
        self.avg_trajectory_length = 12 #tau is initialized to something...
        self.action_entropy = 0
        self.theta = 0

        #If we do our own training, we prepare for that
        if self.mode is threads.STANDALONE:
            #Create a trainer, and link their neural-net and experience-replay to us
            self.trainer = self.settings["trainer_type"](id="trainer_{}".format(self.id),settings=settings, session=session, sandbox=sandbox, mode=threads.PASSIVE)
            self.model_dict = self.trainer.model_dict
            #STANDALONE agents have to keep track of their own training habits!
            self.time_to_training = self.settings['time_to_training']

        if self.mode is threads.WORKER: #If we are a WORKER, we bring our own equipment
            #Create models
            self.model_dict = {}
            models = ["value_net"] if self.settings["single_policy"] else ["policy_0", "policy_1"]
            for model in models:
                m = prio_vnet(
                              self.id,
                              model,
                              self.state_size,
                              session,
                              settings=self.settings,
                              on_cpu=self.settings["worker_net_on_cpu"]
                             )
                self.model_dict[model] = m

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

        #Set up some stuff that depends on what type of training we do...
        if self.settings["single_policy"]:
            model = self.model_dict["value_net"]
            perspective = lambda x:1-x
            k = -1
        else:
            assert p_list[0] == p_list[-1], "{} ::: In dual-policy mode we require queries to be for one policy at a time... (for speed)".format(p_list)
            model = self.model_dict["policy_{}".format(p_list[0])]
            perspective = lambda x:x
            k = 1

        #Simulate future states
        for state_idx, state in enumerate(state_vec):
            self.sandbox.set(state)
            player_action            = self.sandbox.get_actions(                    player=p_list[state_idx])
            future_states[state_idx] = self.sandbox.simulate_actions(player_action, player=p_list[state_idx])
            all_actions[state_idx]   = player_action
            player_list_flat        += [ perspective(p_list[state_idx]) for _ in range(len(player_action)) ]

        #Flatten (kinda like ravel for np.arrays)
        unflatten_dict, temp = {}, 0
        for state_idx in range(len(state_vec)):
            future_states_flat += future_states[state_idx]
            unflatten_dict[state_idx] = slice(temp,temp+len(future_states[state_idx]),1)

        #Run model!
        extrinsic_values_flat = self.run_model(model, future_states_flat, player=player_list_flat)
        values_flat = k * extrinsic_values_flat

        #Undo flatten
        values = [values_flat[unflatten_dict[state_idx]] for state_idx in range(len(state_vec))]
        action_idxs = [ np.argmax( values_flat[unflatten_dict[state_idx]]) for state_idx in range(len(state_vec))]

        #Choose an action . . .
        if training:
            for state_idx in range(len(state_vec)):
                a_idx = action_idxs[state_idx]
                if "distribution" in self.settings["dithering_scheme"]:
                    theta = self.theta = self.settings["action_temperature"](self.clock)
                    if "boltzman" in self.settings["dithering_scheme"]:
                        p = softmax(theta*values[state_idx]).ravel()
                    elif "pareto" in self.settings["dithering_scheme"]:
                        p = utils.pareto(values[state_idx], temperature=theta)
                    self.action_entropy = utils.entropy(p)
                    a_idx = np.random.choice(np.arange(values[state_idx].size), p=p)
                if "epsilon" in self.settings["dithering_scheme"]:
                    epsilon = self.settings["epsilon"](self.clock)
                    if "adaptive" in self.settings["dithering_scheme"]:
                        epsilon *= self.avg_trajectory_length**(-1)
                    dice = random.random()
                    if dice < epsilon:
                        a_idx = np.random.choice(np.arange(values[state_idx].size))
                action_idxs[state_idx] = a_idx
        actions = [all_actions[state_idx][action_idxs[state_idx]] for state_idx in range(len(state_vec)) ]

        #Keep the clock going...
        if training:
            self.clock += self.n_envs * self.n_workers
        return action_idxs, actions

    #
    ###
    #####
    def ready_for_new_round(self, training=False, env=None):
        e_idxs, _ = utils.parse_arg(env, self.env_idxs, indices=True)
        if not self.settings["single_policy"]:
            e_idxs += [e_idx + self.n_envs for e_idx in e_idxs]
        for e in e_idxs:
            a = self.settings["tau_learning_rate"]
            if len(self.current_trajectory[e]) > 0 or training is False:
                self.avg_trajectory_length = (1-a) * self.avg_trajectory_length + a*len(self.current_trajectory[e])

        # Preprocess the trajectories specifiel to prepare them for training
        for e in e_idxs:
            if training and len(self.current_trajectory[e]) > 0:
                t = self.current_trajectory[e]
                if self.settings["workers_do_processing"]:
                    model = self.model_dict["value_net"] if self.settings["single_policy"] else self.model_dict["policy_{}".format(int(e>=self.n_envs))]
                    data = t.process_trajectory(
                                                self.model_runner(model),
                                                self.unpack,
                                                reward_shaper=self.settings["reward_shaper"](self.settings["reward_shaper_param"](self.clock), single_policy=self.settings["single_policy"]),
                                                gamma_discount=self.settings["gamma_extrinsic"]
                                                )
                else:
                    data = t
                metadata = {
                            "policy"    : int(e>=self.n_envs),
                            "winner"    : t.get_winner(),
                            "length"    : len(t),
                            "worker"    : self.id,
                            "packet_id" : self.send_count,
                            }
                self.stored_trajectories.append((metadata,data))
                #Increment some counters to guide what we do
                self.send_count += 1
                if self.mode is threads.STANDALONE:
                    self.time_to_training  -= len(self.current_trajectory[e])
            #Clear trajectory
            self.current_trajectory[e] = self.trajectory_type()

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
        if env is None: endflag = False
        else:           endflag = True
        env_list = utils.parse_arg(env, self.env_idxs)
        #Turn a list of experience ingredients into one list of experiences:
        es = utils.merge_lists(*experience)
        assert len(env_list) == len(es), "WTF!!!! {} != {}".format(len(env_list), len(es))
        for i,e in zip(env_list, es):
            if self.settings["single_policy"]:
                self.current_trajectory[i].add(e, end_of_trajectory=endflag)
            if not self.settings["single_policy"]:
                self.current_trajectory[i + self.n_envs*e[4]].add(e, end_of_trajectory=endflag)

        self.log.debug("agent[{}] appends experience {} to its trajectory-buffer".format(self.id, experience))

    def transfer_data(self, keep_data=False):
        #This function gives away the data gathered
        ret = self.stored_trajectories
        if not keep_data:
            self.stored_trajectories = list()
        return ret
