import logging
import random
import pickle
import numpy as np
from collections import deque
from agents.networks.value_net import value_net
from agents.tetris_agent import tetris_agent

default_settings = {
                    "option" : "entry",
                    "time_to_training" : 100,
                    "time_to_reference_update" : 10,
                    "experience_replay_size" : 10000,
                    "n_samples_each_update" : 1000,
                    "n_train_epochs_per_update" : 5,
                    "initial_action_prob_temp" : 1.0,
                    }

class my_agent(tetris_agent):
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

        self.action_prob_temperature = self.settings["initial_action_prob_temp"]
        self.state_size = self.state_to_vector(self.sandbox.get_state()).shape[1:]
        self.experience_replay = deque(maxlen=self.settings["experience_replay_size"])
        self.model = value_net(self.id, "main", self.state_size, session, settings=self.settings)
        self.reference_model = value_net(self.id, "reference", self.state_size, session, settings=self.settings)
        self.round_stats = {"r":[], "value":[],"prob":[]}
    # # # # #
    # Agent interface fcns
    # # #
    def get_action(self, state, training=False):
        #Get hypothetical future states and turn them into vectors
        self.sandbox.set(state)
        actions = self.sandbox.get_actions(player=self.id)
        future_states = self.sandbox.simulate_actions(actions, self.id)
        values, probabilities = self.run_model(self.model, future_states, self.action_prob_temperature)
        #Choose action
        if training:
            p=probabilities.reshape((-1))/probabilities.sum() #Sometimes rounding breaks down...
            a_idx = np.random.choice(np.arange(p.size), p=p)
        else:
            a_idx = np.argmax(probabilities.reshape((-1)))
        #store some stats
        self.round_stats["value"].append(values[a_idx,0])
        self.round_stats["prob"].append(probabilities[a_idx,0])
        return a_idx, actions[a_idx]

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
        self.experience_replay.append( (s,a,r,s_prime,[done]) )
        self.time_to_training -= 1
        self.round_stats["r"].append(r[0])
        self.log.debug("agent[{}] stores experience {}".format(self.id, info))

    def ready_for_new_round(self):
        print("Agent{} round-stats!".format(self.id))
        for x in self.round_stats:
            print(x, ":" ,np.around(self.round_stats[x], decimals=2))
        self.round_stats = {"r":[], "value":[],"prob":[]}

    def is_ready_for_training(self):
        return (self.time_to_training <= 0)

    def do_training(self):
        self.log.debug("agent[{}] doing training".format(self.id))
        print("agent{} training...".format(self.id))
        ''' random sample from experience_replay '''
        n = len(self.experience_replay)
        all_indices = np.arange(n)
        p = all_indices/all_indices.sum()
        indices = np.random.choice(all_indices, size=self.settings["n_samples_each_update"], p=p).tolist()
        ''' calculate target values'''
        states = [None] * self.settings["n_samples_each_update"]
        rewards = [None] * self.settings["n_samples_each_update"]
        dones = [None] * self.settings["n_samples_each_update"]
        next_states = [None] * self.settings["n_samples_each_update"]
        for i,j in enumerate(indices):
            states[i], _, rewards[i], next_states[i], dones[i] = self.experience_replay[j]
        state_vec = self.states_to_vectors(states)
        done_vec = np.concatenate(dones)[:,np.newaxis]
        reward_vec = np.concatenate(rewards)[:,np.newaxis]
        next_value_vec, _ = self.run_model(self.reference_model, next_states)
        target_value_vec = reward_vec + (1-done_vec.astype(np.int)) * next_value_vec
        minibatch_size = 32
        for t in range(self.settings["n_train_epochs_per_update"]):
            for i in range(0,self.settings["n_samples_each_update"],minibatch_size):
                self.model.train(state_vec[i:i+minibatch_size],target_value_vec[i:i+minibatch_size])
        ''' check if / update reference net'''
        if self.time_to_reference_update == 0:
            print("Updating agent{} reference model!".format(self.id))
            weights = self.model.get_weights(self.model.trainable_vars)
            self.reference_model.set_weights(self.reference_model.trainable_vars, weights)
            self.time_to_reference_update = self.settings["time_to_reference_update"]
        else:
            self.time_to_reference_update -= 1
        self.time_to_training = self.settings["time_to_training"]

    # # # # #
    # Training fcns
    # # #
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
                s, a, r, s_prime, done = e
                tmp.append( (self.state_to_vector(s), a, r, self.state_to_vector(s_prime), done) )
        with open(file, 'wb') as f:
            pickle.dump(tmp, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_memory(self, file):
        with open(file, 'rb') as f:
            self.experience_replay = pickle.load(f)

    def save_all(self,path):
        self.save_weights(path+"/weights")
        self.save_memory(path+"/mem")

    def load_all(self,path):
        self.load_weights(path+"/weights")
        print("agent{} loaded weights from {}".format(self.id, path))
        self.load_memory(path+"/mem")

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
