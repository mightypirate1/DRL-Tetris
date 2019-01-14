import threads
import aux.utils as utils
import numpy as np
import logging
import pickle

class vector_agent_base:
    def __init__(self, id=0, name="base_type!", session=None, sandbox=None, shared_vars=None, settings=None, mode=threads.STANDALONE):
        #Parse settings
        self.settings = utils.parse_settings(settings)
        settings_ok = self.process_settings() #Checks so that the settings are not conflicting
        assert self.settings["n_players"] == 2, "2-player mode only as of yet..."
        assert settings_ok, "Settings are not ok! See previous error messages..."

        #Set up some helper variables
        self.player_idxs = [p for p in range(self.settings["n_players"])]
        self.id = id
        self.name = name
        self.mode = mode
        self.clock = 0

        #Logger
        self.log = logging.getLogger(self.name)
        self.log.debug("name created! type={} mode={}".format(self.name,self.mode))

        #Some basic core functionality
        self.shared_vars = shared_vars
        self.sandbox = sandbox.copy()
        self.state_size = self.state_to_vector(self.sandbox.get_state(), player_list=[0,0]).shape[1:]
        self.model_dict = {}

    def update_clock(self, clock):
        old_clock = self.experience_replay.time
        self.experience_replay.time = self.clock = clock
        print("{} UPDATED CLOCK {} -> {}".format(self.id,old_clock,clock))

    def run_model(self, net, states, player=None):
        assert player is not None, "Specify a player to run the model for!"
        if isinstance(states, np.ndarray):
            assert False, "This should not ever happen"
            if player_list is not None: self.log.warning("run_model was called with an np.array as an argument, and non-None player list. THIS IS NOT MENT TO BE, AND IF YOU DONT KNOW WHAT YOU ARE DOING, EXPECT INCORRECT RESULTS!")
            states_vector = states
        else:
            states_vector = self.states_from_perspective(states, player)
        return net.evaluate(states_vector)

    def run_default_model(self, states, player=None):
        return self.run_model(self.model_dict["default"], states, player=player)

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
        assert self.settings["n_players"] == 2, "only 2player mode as of yet..."
        p_list = utils.parse_arg(player, self.player_idxs)
        return self.states_to_vectors(states, player_lists=[[p, 1-p] for p in p_list])

    def state_from_perspective(self, state, player):
        assert self.settings["n_players"] == 2, "only 2player mode as of yet..."
        return self.state_to_vector(state, [player, 1-player])

    def state_to_vector(self, state, player_list=None):
        if isinstance(state,np.ndarray):
            assert False, "This should not happen"
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
        assert len(player_lists) == len(states), "player_list:{}, states:{}".format(player_lists, states)
        if type(states) is list:
            ret = [self.state_to_vector(s, player_list=player_lists[i]) for i,s in enumerate(states)]
        else:
            #Assume it is a single state if it is not a list
            ret = [self.state_to_vector(states, player_list=player_lists)]
        return np.concatenate(ret, axis=0)

    def update_weights(self, w): #As passed by the trainer's export_weights-fcn..
        self.model_dict["default"].set_weights(self.model_dict["default"].all_vars,w)

    def process_settings(self):
        print("process_settings not implemented yet!")
        return True

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'log' in d:
            d['log'] = d['log'].name
        return d

    def __setstate__(self, d):
        if 'log' in d:
            d['log'] = logging.getLogger(d['log'])
        self.__dict__.update(d)
