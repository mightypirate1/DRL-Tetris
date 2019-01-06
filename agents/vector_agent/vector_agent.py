import logging
import random
import pickle
import random
import numpy as np
import tensorflow as tf
import collections

import aux
from agents.tetris_agent import tetris_agent
from agents.networks.value_net import value_net
from agents.networks.curiosity_network import curiosity_network
from agents.agent_utils.experience_replay import experience_replay
from aux.parameter import *


class vector_agent(tetris_agent):
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

    def __init__(self, n_envs, id=0, session=None, sandbox=None, settings=None):
        self.log = logging.getLogger("agent")
        self.log.debug("Test agent created!")
        self.n_envs = n_envs
        self.id = id
        self.n_train_steps = 0
        self.sandbox = sandbox
        self.settings = aux.settings.default_settings.copy()
        if settings is not None:
            for x in settings:
                self.settings[x] = settings[x]
        assert self.settings["n_players"] == 2, "2-player mode only as of yet..."
        settings_ok = self.process_settings() #Checks so that the settings are not conflicting
        assert settings_ok, "Settings are not ok! See previous error messages..."
        #Initialize training variables
        self.time_to_training = self.settings['time_to_training']
        self.time_to_reference_update = 0#self.settings['time_to_reference_update']

        self.state_size = self.state_to_vector(self.sandbox.get_state(), player_list=[0,0]).shape[1:]
        self.experience_replay = experience_replay(self.settings["experience_replay_size"], prioritized=self.settings["prioritized_experience_replay"])
        self.current_trajectory = []

        self.extrinsic_model =           value_net(self.id, "main_extrinsic",      self.state_size, session, settings=self.settings)
        self.reference_extrinsic_model = value_net(self.id, "reference_extrinsic", self.state_size, session, settings=self.settings)
        if self.settings["use_curiosity"]:
            self.curiosity_network =         curiosity_network(self.id, self.state_size, session)
            self.intrinsic_model =           value_net(self.id, "main_intrinsic",      self.state_size, session, settings=self.settings, output_activation="elu_plus1")
            self.reference_intrinsic_model = value_net(self.id, "reference_intrinsic", self.state_size, session, settings=self.settings, output_activation="elu_plus1")

        self.nemesis = [(i +1)%self.settings["n_players"] for i in range(self.n_envs)]
        self.avg_trajectory_length = 5 #tau is initialized to something...
    # # # # #
    # Agent interface fcns
    # # #
    def get_action(self, state_vec, player=None, training=False, verbose=False):
        #Get hypothetical future states and turn them into vectors!
        if player is None: p_list = [i for i in range(self.n_envs)]
        else : p_list             = player if type(player) is list else [player]
        feed_vector = []
        change_nemesis_flag = [False] * self.n_envs
        idx = 0
        idx_dict = dict( zip([p for p in p_list], [{} for _ in state_vec ]  ) )

        all_actions = [dict(zip(p_list, [ None for i,_ in enumerate(state_vec)])) for _ in range(self.n_envs)]

        for i,s in enumerate(state_vec):
            for p in range(self.settings["n_players"]):
                if p in p_list:
                    self.sandbox.set(s)
                    a = self.sandbox.get_actions(player=p)
                    future_states = self.sandbox.simulate_actions(a, p)
                    feed_vector += future_states
                    all_actions[i][p] = a
                    idx_dict[p][i] = slice(idx,idx+len(future_states),1)
                    idx += len(future_states)

        #Run model!
        player_pairs = [(1-player, player) for _ in range(len(feed_vector))]
        extrinsic_values, _ = self.run_model(self.extrinsic_model, feed_vector, player_lists=player_pairs)
        values = -extrinsic_values

        actions     = [ [ [0]  for _ in range(self.settings["n_players"])  ]  for _ in range(self.n_envs)]
        action_idxs = [ [  0   for _ in range(self.settings["n_players"])  ]  for _ in range(self.n_envs)]

        for e in range(self.n_envs):
            for p in range( self.settings["n_players"] ):
                if p not in p_list: #We choose a dummy-action for the players who dont get to act!
                    actions[e][p] = [0]
                    action_idxs[e][p] = 0
                else:
                    #Choose action!
                    my_values = values[idx_dict[p][e]]
                    if training:
                        if "boltzmann" in self.settings["dithering_scheme"]:
                            assert False, "use adaptive_epsilon as your dithering scheme with this agent!"
                        if self.settings["dithering_scheme"] == "adaptive_epsilon":
                            dice = random.random()
                            if dice < self.settings["epsilon"].get_value(self.n_train_steps) * self.avg_trajectory_length**(-1):
                                a_idx = np.random.choice(np.arange(my_values.size))
                            else:
                                a_idx = np.argmax(my_values.reshape((-1)))
                    else:
                        a_idx = np.argmax(my_values.reshape((-1)))
                    actions[e][p] = all_actions[e][p][a_idx]
                    action_idxs[e][p] = a_idx

        #Print some stuff...
        str = "agents{} chose actions{}:{}\nvals:{}\n---".format(p_list,a_idx,action_idxs,np.around(values[:,0],decimals=2))
        if verbose:
            print(str)
        self.log.debug(str)
        if training:
            self.n_train_steps += 1
        # print(player,actions);exit()
        return action_idxs, actions

    def run_model(self, net, states, player_lists=None):
        if isinstance(states, np.ndarray):
            if player_list is not None: self.log.warning("run_model was called with an np.array as an argument, and non-None player list. THIS IS NOT MENT TO BE, AND IF YOU DONT KNOW WHAT YOU ARE DOING, EXPECT INCORRECT RESULTS!")
            states_vector = states
        else:
            states_vector = self.states_to_vectors(states, player_lists=player_lists)
        return net.evaluate(states_vector)

    def store_experience(self, experience):
        #unpack the experience, and get the reward that is hidden in s'
        s,a,s_prime, players, done, info = experience
        r = s_prime[self.id]["reward"]
        self.current_trajectory.append( [s,a,r,s_prime, players,[done]] )
        self.log.debug("agent[{}] appends experience {} to its trajectory-buffer".format(self.id, info))

    def ready_for_new_round(self,training=False, env=None):
        if e_list is None: e_list = [i for i in range(self.n_envs)]
        for env in e_list:
            a = self.settings["tau_learning_rate"]
            if len(self.current_trajectory[env]) > 0 and training:
                self.avg_trajectory_length = (1-a) * self.avg_trajectory_length + a*len(self.current_trajectory[env])

            if training:
                #Process the trajectory to calculate the TD-error of the value-predictions of each transition
                states_visited, extrinsic_rewards, p_lists, s_primes = [], [], [], []
                for e in self.current_trajectory[env]:
                    s,a,r,s_prime, players,done = e
                    states_visited.append(e[0]) #s
                    extrinsic_rewards.append(e[2]) #r
                    self.sandbox.set(e[0]) #s
                    actions = self.sandbox.get_actions(player=self.id)
                    s_prime = self.sandbox.simulate_actions([actions[e[1]]], self.id)[0] #a
                    s_primes.append(s_prime)
                    p_lists.append(e[4])

                #states_visited should now be all states we visited
                extrinsic_value_s, _      = self.run_model(self.extrinsic_model, states_visited, player_lists=p_lists)
                extrinsic_value_sprime, _ = self.run_model(self.extrinsic_model, s_primes, player_list=p_lists)
                extrinsic_td_errors = np.array(extrinsic_rewards) - self.settings["gamma_extrinsic"] * extrinsic_value_sprime - extrinsic_value_s
                surprise_factors = np.abs(extrinsic_td_errors)
                self.experience_replay.add_samples([(*e, self.nemesis) for e in self.current_trajectory], surprise_factors)
                self.time_to_training -= len(self.current_trajectory)
                self.current_trajectory.clear()

    def is_ready_for_training(self):
        return False
        # return (self.time_to_training <= 0) and (len(self.experience_replay) > self.settings["n_samples_each_update"])

    def do_training(self):
        assert False
        self.log.debug("agent[{}] doing training".format(self.id))
        print("agent{} training...".format(self.id))
        samples, _, time_stamps, is_weights, return_idxs =\
                self.experience_replay.get_random_sample(
                                                         self.settings["n_samples_each_update"],
                                                         alpha=self.settings["prioritized_replay_alpha"].get_value(self.n_train_steps),
                                                         beta=self.settings["prioritized_replay_beta"].get_value(self.n_train_steps),
                                                         )
        #Unpack data (This feels inefficient... think thins through some day)
        states, actions, rewards, next_states, dones, nemeses = [[None]*self.settings["n_samples_each_update"] for x in range(6)]
        for i,s in enumerate(samples):
            states[i], actions[i], rewards[i], next_states[i], dones[i], nemeses[i] = s
        player_list = list(zip(nemeses, [self.id for x in range(len(samples))]))
        #Get data into numpy-arrays...
        state_vec = self.states_to_vectors(states)
        done_vec = np.concatenate(dones)[:,np.newaxis]
        extrinsic_reward_vec = np.concatenate(rewards)[:,np.newaxis]
        next_extrinsic_value_vec_ref, _ = self.run_model(self.reference_extrinsic_model, next_states, player_list=player_list)
        target_extrinsic_value_vec = extrinsic_reward_vec + (1-done_vec.astype(np.int)) * -self.settings["gamma_extrinsic"] * next_extrinsic_value_vec_ref
        if self.settings["use_curiosity"]:
            intrinsic_reward_vec = self.settings["curiosity_reward_multiplier"] * self.run_model(self.curiosity_network, state_vec, player_list=[self.id, self.nemesis])
            next_intrinsic_value_vec_ref, _ = self.run_model(self.reference_intrinsic_model, next_states, player_list=players)
            target_intrinsic_value_vec = intrinsic_reward_vec + (1-done_vec.astype(np.int)) * self.settings["gamma_intrinsic"] * next_intrinsic_value_vec_ref
        is_weights = np.array(is_weights)
        minibatch_size = self.settings["minibatch_size"]
        #TRAIN!
        for t in range(self.settings["n_train_epochs_per_update"]):
            perm = np.random.permutation(self.settings["n_train_epochs_per_update"])
            for i in range(0,self.settings["n_samples_each_update"],minibatch_size):
                self.extrinsic_model.train(state_vec[perm[i:i+minibatch_size]],target_extrinsic_value_vec[perm[i:i+minibatch_size]], weights=is_weights[perm[i:i+minibatch_size]], lr=self.settings["value_lr"].get_value(self.n_train_steps))
                if self.settings["use_curiosity"]:
                    self.intrinsic_model.train(state_vec[perm[i:i+minibatch_size]],target_intrinsic_value_vec[perm[i:i+minibatch_size]], weights=is_weights[perm[i:i+minibatch_size]], lr=self.settings["value_lr"].get_value(self.n_train_steps))
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

        # Get new surprise values for the samples, then store them back into experience replay!
        extrinsic_value_vec, _ = self.run_model(self.extrinsic_model, state_vec, player_list=[self.id, self.nemesis])
        extrinsic_td_errors = (extrinsic_reward_vec - self.settings["gamma_extrinsic"] * next_extrinsic_value_vec_ref - extrinsic_value_vec).reshape(-1).tolist()
        surprise_factors = np.abs(extrinsic_td_errors)
        if self.settings["use_curiosity"]:
            intrinsic_value_vec, _ = self.run_model(self.intrinsic_model, state_vec, player_list=[self.id, self.nemesis])
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

    def save_weights(self, file):
        #Fix this proper one day!!
        self.save_settings(file.replace("weights","settings"))

        output = {}
        extrinsic_model_weight_dict, extrinsic_model_name = self.extrinsic_model.get_weights(self.extrinsic_model.all_vars)
        reference_extrinsic_model_weight_dict, reference_extrinsic_model_name = self.reference_extrinsic_model.get_weights(self.reference_extrinsic_model.all_vars)
        output["extrinsic_model"]           = ((extrinsic_model_weight_dict)           , extrinsic_model_name          ),
        output["reference_extrinsic_model"] = ((reference_extrinsic_model_weight_dict) , reference_extrinsic_model_name),
        if self.settings["use_curiosity"]:
            intrinsic_model_weight_dict, intrinsic_model_name = self.intrinsic_model.get_weights(self.intrinsic_model.all_vars)
            reference_intrinsic_model_weight_dict, reference_intrinsic_model_name = self.reference_intrinsic_model.get_weights(self.reference_intrinsic_model.all_vars)
            output["intrinsic_model"]           = ((intrinsic_model_weight_dict)           , intrinsic_model_name          ),
            output["reference_intrinsic_model"] = ((reference_intrinsic_model_weight_dict) , reference_intrinsic_model_name),
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
            weight_dict["extrinsic_reference_model"] = weight_dict["reference_model"]
        for x in weight_dict: #One version had an error in the save-code, so this is for model backward compatibility
            if len(weight_dict[x]) == 1:
                print("You were saved by a compatibility hack")
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

    def states_from_perspective(self, states, player):
        assert self.settings["n_players"] == 2, "onöy 2player mode as of yet..."
        return self.states_to_vectors(states, player_list=[player, 1-player])

    def state_to_vector(self, state, player_list=None):
        if isinstance(state,np.ndarray):
            return state
        def collect_data(state_dict):
            tmp = []
            for x in state_dict:
                if not x in ["reward", "dead"]:
                    tmp.append(state_dict[x].reshape((1,-1)))
            return np.concatenate(tmp, axis=1)
        if player_list is None: #If None is specified, we assume we want the state of all players!
            player_list = [self.id] + [i for i in range(self.settings['n_players']) if not i==self.id]
        data = [collect_data(state[player]) for player in player_list]
        ret = np.concatenate(data, axis=1)
        return ret

    def states_to_vectors(self, states, player_lists=None):
        assert player_lists is not None
        if type(states) is list:
            ret = [self.state_to_vector(s, player_list=player_lists[i]) for i,s in enumerate(states)]
        else:
            #Assume it is a single state if it is not a list
            ret = [self.state_to_vector(states, player_list=player_lists)]
        return np.concatenate(ret, axis=0)

    def process_settings(self):
        print("process_settings not implemented yet!")
        return True
