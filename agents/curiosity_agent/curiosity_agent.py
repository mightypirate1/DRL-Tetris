import logging
import random
import pickle
import random
import numpy as np
import tensorflow as tf
import os

import aux
from agents.tetris_agent import tetris_agent
from agents.networks.value_net import value_net
from agents.networks.curiosity_network import curiosity_network
from agents.agent_utils.experience_replay import experience_replay
from aux.parameter import *


class curiosity_agent(tetris_agent):
    class experience(tuple): #This is to be able to do quick surprise_factor comparison on experiences
        def __lt__(self,x):
            if isinstance(x, my_agent.experience):
                return self[-1] < x[-1] #Last element in an experience is the surprise_factor
            return self[-1] < x
        def __le__(self,x):
            if isinstance(x, my_agent.experience):
                return self[-1] <= x[-1] #Last element in an experience is the surprise_factor
            return self[-1] <= x
        def __gt__(self,x):
            if isinstance(x, my_agent.experience):
                return self[-1] > x[-1] #Last element in an experience is the surprise_factor
            return self[-1] > x
        def __ge__(self,x):
            if isinstance(x, my_agent.experience):
                return self[-1] >= x[-1] #Last element in an experience is the surprise_factor
            return self[-1] >= x

    def __init__(self, id=0, session=None, sandbox=None, settings=None, reuse_nets=False):
        self.log = logging.getLogger("agent")
        self.log.debug("Test agent created!")
        self.id = id
        self.n_train_steps = 0
        self.sandbox = sandbox
        self.load_settings(settings)
        settings_ok = self.process_settings() #Checks so that the settings are not conflicting
        assert settings_ok, "Settings are not ok! See previous error messages..."
        #Initialize training variables
        self.time_to_training = self.settings['time_to_training']
        self.time_to_reference_update = 0#self.settings['time_to_reference_update']

        self.state_size = self.state_to_vector(self.sandbox.get_state()).shape[1:]
        self.experience_replay = experience_replay(self.settings["experience_replay_size"], prioritized=self.settings["prioritized_experience_replay"])
        self.current_trajectory = []

        self.extrinsic_model =           value_net(self.id, "main_extrinsic",      self.state_size, session, settings=self.settings, reuse_nets=reuse_nets)
        self.reference_extrinsic_model = value_net(self.id, "reference_extrinsic", self.state_size, session, settings=self.settings, reuse_nets=reuse_nets)
        if self.settings["use_curiosity"]:
            self.curiosity_network =         curiosity_network(self.id, self.state_size, session, reuse_nets=reuse_nets)
            self.intrinsic_model =           value_net(self.id, "main_intrinsic",      self.state_size, session, settings=self.settings, output_activation="elu_plus1", reuse_nets=reuse_nets)
            self.reference_intrinsic_model = value_net(self.id, "reference_intrinsic", self.state_size, session, settings=self.settings, output_activation="elu_plus1", reuse_nets=reuse_nets)

        self.round_stats = {"r":[], "value":[],"prob":[], "tau" : 0}
        self.avg_trajectory_length = 5 #tau is initialized to something...
    # # # # #
    # Agent interface fcns
    # # #
    def get_action(self, state, training=False, verbose=False):
        #Get hypothetical future states and turn them into vectors!
        self.sandbox.set(state)
        actions = self.sandbox.get_actions(player=self.id)
        future_states = self.sandbox.simulate_actions(actions, self.id)

        #Run model!
        extrinsic_values, _ = self.run_model(self.extrinsic_model, future_states)
        if (training or self.settings["curiosity_during_testing"]) and self.settings["use_curiosity"]:
            intrinsic_values, _ = self.run_model(self.intrinsic_model, future_states)
            values = extrinsic_values + intrinsic_values * self.settings["curiosity_amount"].get_value(self.n_train_steps)
        else:
            values = extrinsic_values
        #Choose action!
        if training:
            if self.settings["dithering_scheme"] == "boltzmann":
                exp_v = np.exp( self.settings["action_temperature"].get_value(self.n_train_steps) * (values.reshape(-1) - np.max(values)) )
                p = exp_v / exp_v.sum()
                # print( self.settings["action_temperature"].get_value(self.n_train_steps) * (values.reshape(-1)) )
                a_idx = np.random.choice(np.arange(values.size), p=p)
            if self.settings["dithering_scheme"] == "adaptive_epsilon":
                dice = random.random()
                if dice < self.settings["epsilon"].get_value(self.n_train_steps) * self.avg_trajectory_length**(-1):
                    a_idx = np.random.choice(np.arange(values.size))
                else:
                    a_idx = np.argmax(values.reshape((-1)))
        else:
            a_idx = np.argmax(values.reshape((-1)))
        action = actions[a_idx]
        #Store some stats...
        self.round_stats["value"].append(values[a_idx,0])
        self.round_stats["tau"] += 1
        #Print some stuff...
        str = "agent{} chose action{}:{}\nvals:{}\n---".format(self.id,a_idx,action,np.around(values[:,0],decimals=2))
        if verbose:
            print(str)
        self.log.debug(str)
        if training:
            self.n_train_steps += 1
        return a_idx, action

    def run_model(self, net, states):
        if isinstance(states, np.ndarray):
            states_vector = states
        else:
            states_vector = self.states_to_vectors(states)
        return net.evaluate(states_vector)

    def store_experience(self, experience):
        #unpack the experience, and get the reward that is hidden in s'
        s,a,s_prime, done, info = experience
        r = s_prime[self.id]["reward"]
        self.current_trajectory.append( (s,a,r,s_prime,[done]) )
        self.round_stats["r"].append(r[0])
        self.log.debug("agent[{}] appends experience {} to its trajectory-buffer".format(self.id, info))

    def ready_for_new_round(self,training=False):
        if len(self.current_trajectory) == 0:
            return
        if training:
            #Update tau!
            a = self.settings["tau_learning_rate"]
            self.avg_trajectory_length = (1-a) * self.avg_trajectory_length + a*len(self.current_trajectory)
            #Process the trajectory to calculate the TD-error of the value-predictions of each transition
            states_visited, extrinsic_rewards = [], []
            for s,a,r,s_prime,done in self.current_trajectory:
                states_visited.append(s)
                extrinsic_rewards.append(r)
            states_visited.append(self.current_trajectory[-1][3]) #add the last s' too!
            #states_visited should now be all states we visited
            extrinsic_values, _ = self.run_model(self.extrinsic_model, states_visited)
            extrinsic_td_errors = np.array(extrinsic_rewards) + self.settings["gamma_extrinsic"] * extrinsic_values[1:] - extrinsic_values[:-1] #for each i, td[i]=r[i]+gamma_extrinsic*V(s'[i])-V(s[i])
            surprise_factors = np.abs(extrinsic_td_errors)
            if self.settings["use_curiosity"]:
                intrinsic_rewards =  self.settings["curiosity_reward_multiplier"] * (1 - self.settings["gamma_intrinsic"]) * self.run_model(self.curiosity_network, states_visited[:-1])
                intrinsic_values, _ = self.run_model(self.intrinsic_model, states_visited)
                intrinsic_td_errors = intrinsic_rewards + self.settings["gamma_intrinsic"] * intrinsic_values[1:] - intrinsic_values[:-1] #for each i, td[i]=r[i]+gamma_extrinsic*V(s'[i])-V(s[i])
                surprise_factors += self.settings["curiosity_amount"].get_value(self.n_train_steps) * np.abs(intrinsic_td_errors)
                self.curiosity_network.train(self.states_to_vectors(states_visited[:-1]), lr=self.settings["curiosity_lr"].get_value(self.n_train_steps))
                self.log.debug(
                                "round_expected_curiosity[avg,min,max]:{},{},{}".format(
                                                                                np.mean(self.settings["curiosity_amount"].get_value(self.n_train_steps)*intrinsic_values[:]),
                                                                                np.min(self.settings["curiosity_amount"].get_value(self.n_train_steps)*intrinsic_values[:]),
                                                                                np.max(self.settings["curiosity_amount"].get_value(self.n_train_steps)*intrinsic_values[:])
                                                                                ) +
                                "round_recieved_curiosity[avg,min,max]:{},{},{}".format(
                                                                                np.mean(intrinsic_rewards[:]),
                                                                                np.min(intrinsic_rewards[:]),
                                                                                np.max(intrinsic_rewards[:])
                                                                                )
                                )

            self.experience_replay.add_samples(self.current_trajectory, surprise_factors)
            self.time_to_training -= len(self.current_trajectory)
            self.current_trajectory.clear()
        #Empty stats
        self.round_stats = {"r":[], "value":[],"prob":[], "tau" : 0}

    def is_ready_for_training(self):
        return (self.time_to_training <= 0) and (len(self.experience_replay) > self.settings["n_samples_each_update"])

    def do_training(self):
        self.log.debug("agent[{}] doing training".format(self.id))
        print("agent{} training...".format(self.id))
        ''' random sample from experience_replay '''
        samples, _, time_stamps, is_weights, return_idxs =\
                self.experience_replay.get_random_sample(
                                                         self.settings["n_samples_each_update"],
                                                         alpha=self.settings["prioritized_replay_alpha"].get_value(self.n_train_steps),
                                                         beta=self.settings["prioritized_replay_beta"].get_value(self.n_train_steps),
                                                         )
        ''' calculate target values'''
        #Unpack data (This feels inefficient... think thins through some day)
        states, actions, rewards, next_states, dones = [[None]*self.settings["n_samples_each_update"] for x in range(5)]
        for i,s in enumerate(samples):
            states[i], actions[i], rewards[i], next_states[i], dones[i] = s
        #Get data into numpy-arrays...
        state_vec = self.states_to_vectors(states)
        done_vec = np.concatenate(dones)[:,np.newaxis]
        extrinsic_reward_vec = np.concatenate(rewards)[:,np.newaxis]
        next_extrinsic_value_vec_ref, _ = self.run_model(self.reference_extrinsic_model, next_states)
        target_extrinsic_value_vec = extrinsic_reward_vec + (1-done_vec.astype(np.int)) * self.settings["gamma_extrinsic"] * next_extrinsic_value_vec_ref #If surprises are negative, that cancels out here!
        if self.settings["use_curiosity"]:
            intrinsic_reward_vec = self.settings["curiosity_reward_multiplier"] * self.run_model(self.curiosity_network, state_vec)
            next_intrinsic_value_vec_ref, _ = self.run_model(self.reference_intrinsic_model, next_states)
            target_intrinsic_value_vec = intrinsic_reward_vec + (1-done_vec.astype(np.int)) * self.settings["gamma_intrinsic"] * next_intrinsic_value_vec_ref #If surprises are negative, that cancels out here!
        is_weights = np.array(is_weights)
        minibatch_size = self.settings["minibatch_size"]
        #TRAIN!
        for t in range(self.settings["n_train_epochs_per_update"]):
            perm = np.random.permutation(self.settings["n_train_epochs_per_update"])
            for i in range(0,self.settings["n_samples_each_update"],minibatch_size):
                self.extrinsic_model.train(state_vec[perm[i:i+minibatch_size]],target_extrinsic_value_vec[perm[i:i+minibatch_size]], weights=is_weights[perm[i:i+minibatch_size]], lr=self.settings["value_lr"].get_value(self.n_train_steps))
                if self.settings["use_curiosity"]:
                    self.intrinsic_model.train(state_vec[perm[i:i+minibatch_size]],target_intrinsic_value_vec[perm[i:i+minibatch_size]], weights=is_weights[perm[i:i+minibatch_size]], lr=self.settings["value_lr"].get_value(self.n_train_steps))
        ''' check if / update reference net'''
        if self.time_to_reference_update == 0:
            print("Updating agent{} reference model!".format(self.id))
            if self.settings["alternating_models"]:
                #alternate extrinsic
                tmp = self.reference_extrinsic_model
                self.reference_extrinsic_model = self.extrinsic_model
                self.extrinsic_model = tmp
                if self.settings["use_curiosity"]:
                    #alternate intrinsic
                    tmp = self.reference_intrinsic_model
                    self.reference_intrinsic_model = self.intrinsic_model
                    self.intrinsic_model = tmp
            else:
                weights = self.extrinsic_model.get_weights(self.extrinsic_model.trainable_vars)
                self.reference_extrinsic_model.set_weights(self.reference_extrinsic_model.trainable_vars,weights)
                if self.settings["use_curiosity"]:
                    weights = self.intrinsic_model.get_weights(self.intrinsic_model.trainable_vars)
                    self.reference_intrinsic_model.set_weights(self.reference_intrinsic_model.trainable_vars,weights)
            self.time_to_reference_update = self.settings["time_to_reference_update"]
        else:
            self.time_to_reference_update -= 1
        #Get new surprise values for the samples, then store them back into experience replay!
        extrinsic_value_vec, _ = self.run_model(self.extrinsic_model, state_vec)
        extrinsic_td_errors = (extrinsic_reward_vec + self.settings["gamma_extrinsic"] * next_extrinsic_value_vec_ref - extrinsic_value_vec).reshape(-1).tolist()
        surprise_factors = np.abs(extrinsic_td_errors)
        if self.settings["use_curiosity"]:
            intrinsic_value_vec, _ = self.run_model(self.intrinsic_model, state_vec)
            intrinsic_td_errors = (intrinsic_reward_vec + self.settings["gamma_intrinsic"] * next_intrinsic_value_vec_ref - intrinsic_value_vec).reshape(-1).tolist()
            surprise_factors += self.settings["curiosity_amount"].get_value(self.n_train_steps) *np.abs(intrinsic_td_errors)
        self.experience_replay.add_samples( samples, surprise_factors, filter=return_idxs, time_stamps=time_stamps)
        '''---'''

        print("---")
        print("TAU:{}".format(self.avg_trajectory_length))
        print("EPSILON:{}".format(self.settings["epsilon"].get_value(self.n_train_steps)/self.avg_trajectory_length))
        print("curiosity_amount:{}".format(self.settings["curiosity_amount"].get_value(self.n_train_steps)))
        print("value_lr:{}".format(self.settings["value_lr"].get_value(self.n_train_steps)))
        print("---")

        #Reset delay
        self.time_to_training = self.settings["time_to_training"]

    # # # # #
    # Training fcns
    # # #
    def init_training(self):
        self.current_trajectory = []

    def get_next_states(self, state):
        self.sandbox.set(state)
        return self.sandbox.simulate_all_actions(self.id)

    # # # # #
    # Memory management fcns
    # # #
    def save_settings(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.settings, f, protocol=pickle.HIGHEST_PROTOCOL)
    def load_settings_from_file(self,file):
        # assert os.path.exists(file), "Not valid path:{}".format(file)
        if os.path.exists(file):
            with open(file, "rb") as f:
                _settings = pickle.load(f)
            self.load_settings(_settings)
        else:
            print("No settings-file found!")
    def load_settings(self, s):
        self.settings = aux.settings.default_settings.copy()
        if s is not None:
            for x in s:
                self.settings[x] = s[x]

    def save_weights(self, file):
        #Fix this proper one day!!
        self.save_settings(file.replace("weights","settings"))

        output = {}
        extrinsic_model_weight_dict, extrinsic_model_name = self.extrinsic_model.get_weights(self.extrinsic_model.all_vars)
        reference_extrinsic_model_weight_dict, reference_extrinsic_model_name = self.reference_extrinsic_model.get_weights(self.reference_extrinsic_model.all_vars)
        output["extrinsic_model"]           = (extrinsic_model_weight_dict           , extrinsic_model_name          ),
        output["reference_extrinsic_model"] = (reference_extrinsic_model_weight_dict , reference_extrinsic_model_name),
        if self.settings["use_curiosity"]:
            intrinsic_model_weight_dict, intrinsic_model_name = self.intrinsic_model.get_weights(self.intrinsic_model.all_vars)
            reference_intrinsic_model_weight_dict, reference_intrinsic_model_name = self.reference_intrinsic_model.get_weights(self.reference_intrinsic_model.all_vars)
            output["intrinsic_model"]           = (intrinsic_model_weight_dict           , intrinsic_model_name          ),
            output["reference_intrinsic_model"] = (reference_intrinsic_model_weight_dict , reference_intrinsic_model_name),
            ''' Implement this if you really want it... '''
            # curiosity_network_weight_dict, curiosity_network_name = self.curiosity_network.get_weights(self.curiosity_network.all_vars)
            # output["curiosity_network"]         = ((curiosity_network_weight_dict)         , curiosity_network_name        ),
        with open(file, 'wb') as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_weights(self, file):
        with open(file, 'rb') as f:
            weight_dict = pickle.load(f)
        if "extrinsic_model" not in weight_dict:
            weight_dict["extrinsic_model"] = weight_dict["model"]
            weight_dict["reference_extrinsic_model"] = weight_dict["reference_model"]
        for x in weight_dict: #One version had an error in the save-code, so this is for model backward compatibility
            if len(weight_dict[x]) == 1:
                weight_dict[x] = weight_dict[x][0]
        self.extrinsic_model.set_weights(self.extrinsic_model.all_vars,weight_dict["extrinsic_model"])
        self.reference_extrinsic_model.set_weights(self.reference_extrinsic_model.all_vars,weight_dict["reference_extrinsic_model"])
        if self.settings["use_curiosity"]:
            if "intrinsic_model" in weight_dict:
                self.intrinsic_model.set_weights(self.intrinsic_model.all_vars,weight_dict["intrinsic_model"])
                self.reference_intrinsic_model.set_weights(self.reference_intrinsic_model.all_vars,weight_dict["reference_intrinsic_model"])
                self.settings["use_curiosity"] = False
            elif "curiosity_network" in weight_dict:
                self.curiosity_network.set_weights(self.curiosity_network.all_vars,weight_dict["curiosity_network"])
            else:
                self.settings["use_curiosity"] = False

    def save_memory(self, file):
        tmp = deque(maxlen=self.settings["experience_replay_size"])
        if len(self.experience_replay) > 0:
            #Turns out PythonHandle-objects (corner-stone of the tetris-env is not picklable)...
            #Because of that we convert all states to np-arrays and store those instead. (presumed to be less compact)
            for e in self.experience_replay:
                s, a, r, s_prime, done, surprise = e
                tmp.append( (self.state_to_vector(s), a, r, self.state_to_vector(s_prime), done, surprise) )
        with open(file, 'wb') as f:
            pickle.dump(tmp, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_memory(self, file):
        self.log.warning("You are loading the memory of an agent. This is not compatible with all agents and settings etc. If your agent's state_to_vector function takes no parameters, or is always called wit hthe same parameters, you should be fine, but be adviced that this feature is onlöy partially implemented!")
        with open(file, 'rb') as f:
            self.experience_replay = pickle.load(f)

    def save(self,path, option="all"):
        if option in ["all", "weights"]:
            self.save_weights(path+"/weights")
        if option in ["all", "mem"]:
            self.save_memory(path+"/mem")

    def load(self,path, option="all"):
        if option in ["all", "weights"]:
            self.load_weights(path+"/weights")
            print("agent{} loaded weights from {}".format(self.id, path))
        if option in ["all", "mem"]:
            self.load_memory(path+"/mem")
            print("agent{} loaded memory from {}".format(self.id, path))
        self.load_weights(path+"/weights")

    # # # # #
    # Helper fcns
    # # #
    def state_to_vector(self, state):
        if isinstance(state,np.ndarray):
            return state
        def collect_data(state_dict):
            tmp = []
            for x in state_dict:
                if not x in ["reward", "dead"]:
                    tmp.append(state_dict[x].reshape((1,-1)))
            return np.concatenate(tmp, axis=1)
        data = [collect_data(state[self.id])]
        data += [collect_data(state[i]) for i in range(self.settings['n_players']) if not i==self.id]
        data = np.concatenate(data, axis=1)
        return data

    def states_to_vectors(self, states):
        if isinstance(states,list) or isinstance(states,deque):
            #Assume it is a list of states if it is a list
            ret = [self.state_to_vector(s) for s in states]
        else:
            #Assume it is a single state if it is not a list
            ret = [self.state_to_vector(states)]
        return np.concatenate(ret, axis=0)

    def process_settings(self):
        print("process_settings not implemented yet!")
        return True
