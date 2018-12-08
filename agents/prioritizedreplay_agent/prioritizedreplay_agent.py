import logging
import random
import pickle
import random
import numpy as np
import bisect
from agents.networks.value_net import value_net
from agents.tetris_agent import tetris_agent
from agents.prioritizedreplay_agent.prioritized_experience_replay import prioritized_experience_replay

default_settings = {
                    "option" : "entry",
                    "minibatch_size" : 32,
                    "time_to_training" : 10**3,
                    "time_to_reference_update" : 5,
                    "experience_replay_size" : 10**5,
                    "n_samples_each_update" : 10**2,
                    "n_train_epochs_per_update" : 5,
                    "initial_action_prob_temp" : 1.0,
                    "tau_learning_rate" : 0.01,
                    "tau_tolerance" : 0.4,
                    "temp_increase_factor" : 1.02,
                    "temp_decrease_factor" : 0.998,
                    "dithering_scheme" : "adaptive_epsilon",
                    "adaptive_epsilon" : 1.0,
                    "gamma" : 1.0,
                    "ALTERNATE_MODELS_INSTEAD_OF_MOVING_REFERENCEMODEL" : True,
                    "prioritized_replay_alpha_init" : 0.7,
                    "prioritized_replay_beta_init" : 0.5, #0.5, used in paper, then linear increase to 1...
                    }

class prioritizedreplay_agent(tetris_agent):
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

    def __init__(self, id=0, session=None, sandbox=None, settings=None):
        self.log = logging.getLogger("agent")
        self.log.debug("Test agent created!")
        self.id = id
        self.sandbox = sandbox
        self.settings = default_settings.copy()
        if settings is not None:
            for x in settings:
                self.settings[x] = settings[x]
        settings_ok = self.process_settings() #Checks so that the settings are not conflicting
        assert settings_ok, "Settings are not ok! See previous error messages..."
        #Initialize training variables
        self.time_to_training = self.settings['time_to_training']
        self.time_to_reference_update = 0#self.settings['time_to_reference_update']

        self.state_size = self.state_to_vector(self.sandbox.get_state()).shape[1:]
        self.experience_replay = prioritized_experience_replay(self.settings["experience_replay_size"])
        self.current_trajectory = []

        self.model = value_net(self.id, "main", self.state_size, session, settings=self.settings)
        self.reference_model = value_net(self.id, "reference", self.state_size, session, settings=self.settings)
        self.round_stats = {"r":[], "value":[],"prob":[], "tau" : 0}

        self.avg_trajectory_length = 5 #tau
        self.action_prob_temperature = self.settings["initial_action_prob_temp"]
        self.prioritized_replay_alpha = self.settings["prioritized_replay_alpha_init"]
        self.prioritized_replay_beta = self.settings["prioritized_replay_beta_init"]
    # # # # #
    # Agent interface fcns
    # # #
    def get_action(self, state, training=False, verbose=False):
        #Get hypothetical future states and turn them into vectors!
        self.sandbox.set(state)
        actions = self.sandbox.get_actions(player=self.id)
        future_states = self.sandbox.simulate_actions(actions, self.id)
        #Get temperature, just in case that is our dithering scheme...
        if self.settings["dithering_scheme"] == "boltzmann_tau":
            time = 1+len(self.current_trajectory)
            temperature = self.action_prob_temperature * time
        else:
            temperature = self.action_prob_temperature
        #Run model!
        values, probabilities = self.run_model(self.model, future_states, temperature=temperature)
        #Choose action!
        p=probabilities.reshape((-1))/probabilities.sum() #Sometimes rounding breaks down...
        if training:
            if "boltzmann" in self.settings["dithering_scheme"]:
                a_idx = np.random.choice(np.arange(p.size), p=p)
            if self.settings["dithering_scheme"] == "adaptive_epsilon":
                dice = random.random()
                if dice < self.settings["adaptive_epsilon"]*self.avg_trajectory_length**(-1):
                    a_idx = np.random.choice(np.arange(p.size))
                else:
                    a_idx = np.argmax(values.reshape((-1)))
        else:
            a_idx = np.argmax(values.reshape((-1)))
        action = actions[a_idx]
        #Store some stats...
        self.round_stats["value"].append(values[a_idx,0])
        self.round_stats["prob"].append(probabilities[a_idx,0])
        self.round_stats["tau"] += 1
        #Print some stuff...
        str = "agent{} chose action{}:{}\nprobs:{}\nvals:{}\n---".format(self.id,a_idx,action,np.around(p,decimals=2),np.around(values[:,0],decimals=2))
        if verbose:
            print(str)
        self.log.debug(str)
        return a_idx, action

    def run_model(self, net, states, temperature=1.0):
        if isinstance(states, np.ndarray):
            states_vector = states
        else:
            states_vector = self.states_to_vectors(states)
        return net.evaluate(states_vector, temperature)

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
            states_visited, rs = [], []
            for s,a,r,s_prime,done in self.current_trajectory:
                states_visited.append(s)
                rs.append(r)
            states_visited.append(self.current_trajectory[-1][3]) #add the last s' too!
            #states_visited should now be all states we visited
            values, _ = self.run_model(self.model, states_visited)
            td_errors = np.array(rs) + self.settings["gamma"] * values[1:] - values[:-1] #for each i, td[i]=r[i]+gamma*V(s'[i])-V(s[i])
            surprise_factors = np.abs(td_errors)
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
                                                         alpha=self.prioritized_replay_alpha,
                                                         beta=self.prioritized_replay_beta,
                                                         )
        ''' calculate target values'''
        #Unpack data (This feels inefficient... think thins through some day)
        states, actions, rewards, next_states, dones = [[None]*self.settings["n_samples_each_update"] for x in range(5)]
        for i,s in enumerate(samples):
            states[i], actions[i], rewards[i], next_states[i], dones[i] = s
        #Get data into numpy-arrays...
        state_vec = self.states_to_vectors(states)
        done_vec = np.concatenate(dones)[:,np.newaxis]
        reward_vec = np.concatenate(rewards)[:,np.newaxis]
        next_value_vec_ref, _ = self.run_model(self.reference_model, next_states)
        target_value_vec = reward_vec + (1-done_vec.astype(np.int)) * self.settings["gamma"] * next_value_vec_ref #If surprises are negative, that cancels out here!
        is_weights = np.array(is_weights)
        minibatch_size = self.settings["minibatch_size"]
        #TRAIN!
        for t in range(self.settings["n_train_epochs_per_update"]):
            perm = np.random.permutation(self.settings["n_train_epochs_per_update"])
            for i in range(0,self.settings["n_samples_each_update"],minibatch_size):
                self.model.train(state_vec[perm[i:i+minibatch_size]],target_value_vec[perm[i:i+minibatch_size]], weights=is_weights[perm[i:i+minibatch_size]])
        ''' check if / update reference net'''
        if self.time_to_reference_update == 0:
            print("Updating agent{} reference model!".format(self.id))
            if self.settings["ALTERNATE_MODELS_INSTEAD_OF_MOVING_REFERENCEMODEL"]:
                tmp = self.reference_model
                self.reference_model = self.model
                self.model = tmp
            else:
                weights = self.model.get_weights(self.model.trainable_vars)
                self.reference_model.set_weights(self.reference_model.trainable_vars,weights)
            self.time_to_reference_update = self.settings["time_to_reference_update"]
        else:
            self.time_to_reference_update -= 1
        #Get new surprise values for the samples, then store them back into experience replay!
        value_vec, _ = self.run_model(self.model, state_vec)
        # next_value_vec, _ = self.run_model(self.model, next_states)
        td_errors = (reward_vec + self.settings["gamma"] * next_value_vec_ref - value_vec).reshape(-1).tolist()
        self.experience_replay.add_samples( samples, td_errors, filter=return_idxs, time_stamps=time_stamps)
        '''---'''

        print("---")
        print("TAU:{}".format(self.avg_trajectory_length))
        print("EPSILON:{}".format(self.settings["adaptive_epsilon"]/self.avg_trajectory_length))
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
    def save_weights(self, file):
        model_weight_dict, model_name = self.model.get_weights(self.model.all_vars)
        reference_weight_dict, reference_name = self.model.get_weights(self.model.all_vars)
        output = {
                    "model" : (model_weight_dict, model_name),
                    "reference_model" : (reference_weight_dict, reference_name)
                  }
        with open(file, 'wb') as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_weights(self, file):
        with open(file, 'rb') as f:
            weight_dict = pickle.load(f)
        self.model.set_weights(self.model.all_vars, weight_dict["model"])
        self.reference_model.set_weights(self.reference_model.all_vars, weight_dict["reference_model"])

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
