import random
import random
import numpy as np
import tensorflow as tf
from scipy.special import softmax

import threads
import aux
import aux.utils as utils
from agents.vector_q_agent.vector_q_agent_base import vector_q_agent_base
from agents.vector_q_agent.vector_q_agent_trainer import vector_q_agent_trainer
import agents.agent_utils as agent_utils
import agents.datatypes as dt
from agents.agent_utils import q_helper_fcns as q
from agents.networks import prio_qnet
from aux.parameter import *

class vector_q_agent(vector_q_agent_base):
    def __init__(
                 self,
                 n_envs,                    # How many envs in the vector-env?
                 n_workers=1,               # How many workers run in parallel? If you don't know, guess it's just 1
                 id=0,                      # What's this workers name?
                 session=None,              # The session to operate in
                 sandbox=None,              # Sandbox to play in!
                 mode=threads.STANDALONE,   # What's our role?
                 settings=None,             # Settings-dict passed down from the ancestors
                 init_weights=None,
                 init_clock=0,
                ):

        #Some general variable initialization etc...
        vector_q_agent_base.__init__(self, id=id, name="worker{}".format(id), session=session, sandbox=sandbox, settings=settings, mode=mode)
        #Helper variables
        self.env_idxs = [i for i in range(n_envs)]
        self.n_envs = n_envs
        self.n_workers = n_workers
        self.n_experiences,self.send_count, self.send_length = 0, 0, 0

        #In any mode, we need a place to store transitions!
        self.trajectory_type = dt.q_trajectory
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
            self.time_to_training = max(self.settings['time_to_training'],self.settings['n_samples_each_update'])

        if self.mode is threads.WORKER: #If we are a WORKER, we bring our own equipment
            #Create models
            self.model_dict = {}
            models = ["q_net"] if self.settings["single_policy"] else ["policy_0", "policy_1"]
            for model in models:
                m = prio_qnet(
                              self.id,
                              model,
                              self.state_size,
                              [self.n_rotations, self.n_translations, self.n_pieces], #Output_shape
                              session,
                              worker_only=True,
                              settings=self.settings,
                             )
                self.model_dict[model] = m

        if init_weights is not None:
            print("Agent{} initialized from weights: {} and clock: {}".format(self.id, init_weights, init_clock))
            self.update_clock(init_clock)
            self.load_weights(init_weights,init_weights)

    # # # # #
    # Agent interface fcns
    # # #

    #
    ###
    #####
    def get_action(self, state_vec, player=None, random_action=False, training=False, verbose=False):
        #Get hypothetical future states and turn them into vectors!
        p_list = utils.parse_arg(player, self.player_idxs)

        #Set up some stuff that depends on what type of training we do...
        if self.settings["single_policy"]:
            model = self.model_dict["q_net"]
        else:
            assert False, "not tested yet. comment out this line if you brave"
            assert p_list[0] == p_list[-1], "{} ::: In dual-policy mode we require queries to be for one policy at a time... (for speed)".format(p_list)
            model = self.model_dict["policy_{}".format(p_list[0])]

        #Run model!
        Q, _, pieces = self.run_model(model, state_vec, player=p_list)

        #Choose an action . . .
        distribution = self.settings["eval_distribution"] if not training else self.settings["dithering_scheme"]
        action_idxs = [None for _ in state_vec]
        for i, (state, _piece, player) in enumerate(zip(state_vec,pieces,p_list)):
            piece, _ = _piece
            if distribution == "argmax":
                (r, t), entropy = q.action_argmax(Q[i,:,:,piece])
            elif distribution == "pareto_distribution":
                theta = self.theta = self.settings["action_temperature"](self.clock)
                (r, t), entropy = q.action_pareto(Q[i,:,:,piece], theta)
            elif distribution == "boltzman_distribution":
                theta = self.theta = self.settings["action_temperature"](self.clock)
                (r, t), entropy = q.action_boltzman(Q[i,:,:,piece], theta)
            elif distribution == "adaptive_epsilon":
                epsilon = self.settings["epsilon"](self.clock) * self.avg_trajectory_length**(-1)
                (r, t), entropy = q.action_epsilongreedy(Q[i,:,:,piece], epsilon)
            else:
                raise Exception("specify a supported distribution for selecting actions please! see code right above this line to see what the options are :)")
            self.action_entropy = entropy
            action_idxs[i] = (r,t,piece)

        #Nearly done! Just need to create the actions...
        actions = [q.make_q_action(r,t) for r,t,_ in action_idxs]

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
                    model = self.model_dict["q_net"] if self.settings["single_policy"] else self.model_dict["policy_{}".format(int(e>=self.n_envs))]
                    data = t.process_trajectory(
                                                self.model_runner(model),
                                                self.unpack,
                                                reward_shaper=self.settings["reward_shaper"](self.settings["reward_shaper_param"](self.clock), single_policy=self.settings["single_policy"]),
                                                gamma_discount=self.gamma,
                                               )
                else:
                    data = t
                metadata = {
                            "policy"    : int(e>=self.n_envs),
                            "winner"    : t.winner,
                            "length"    : len(t),
                            "worker"    : self.id,
                            "packet_id" : self.send_count,
                            }
                self.stored_trajectories.append((metadata,data))
                #Increment some counters to guide what we do
                self.send_count += 1
                self.send_length += len(t)
                if self.mode is threads.STANDALONE:
                    self.time_to_training  -= len(self.current_trajectory[e])
            #Clear trajectory
            self.current_trajectory[e] = self.trajectory_type()

        #Standalone agents have to keep track of their training habits!
        if training and self.mode is threads.STANDALONE:
            if self.time_to_training < 1:
                self.trainer.receive_data(self.transfer_data())
                self.trainer.do_training()
                self.time_to_training = self.settings['time_to_training']

    #
    ###
    #####
    def store_experience(self, experience, env=None):
        env_list = utils.parse_arg(env, self.env_idxs)
        #Turn a list of experience ingredients into one list of experiences:
        es = utils.merge_lists(*experience)
        assert len(env_list) == len(es), "WTF!!!! {} != {}".format(len(env_list), len(es))
        for i,e in zip(env_list, es):
            if e[0] is None:
                continue
            if self.settings["single_policy"]:
                self.current_trajectory[i].add(e)
            if not self.settings["single_policy"]:
                #Player1's trajectories strored first (n_envs many) and then player2's:
                self.current_trajectory[i + e[4]*self.n_envs].add(e)
            self.n_experiences += 1
        self.log.debug("agent[{}] appends experience {} to its trajectory-buffer".format(self.id, experience))

    def transfer_data(self, keep_data=False):
        #This function gives away the data gathered
        ret = self.stored_trajectories
        if not keep_data:
            self.stored_trajectories = list()
        return ret
