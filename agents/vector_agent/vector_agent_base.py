import threads
import aux.utils as utils
import numpy as np

class vector_agent_base:
    def __init__(self, n_envs, id=0, session=None, sandbox=None, trajectory_queue=None, settings=None, mode=threads.STANDALONE):
        self.n_envs = n_envs
        self.env_idxs = [i for i in range(n_envs)]
        self.id = id
        self.mode = mode
        self.n_train_steps = 0
        self.sandbox = sandbox.copy()
        self.settings = utils.parse_settings(settings)
        settings_ok = self.process_settings() #Checks so that the settings are not conflicting
        assert settings_ok and self.settings["n_players"] == 2, "2-player mode only as of yet..."
        self.player_idxs = [p for p in range(self.settings["n_players"])]
        self.trajectory_queue = trajectory_queue
        assert settings_ok, "Settings are not ok! See previous error messages..."
        self.state_size = self.state_to_vector(self.sandbox.get_state(), player_list=[0,0]).shape[1:]

    # # # # #
    # Memory management fcns
    # # #
    def save_settings(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.settings, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_weights(self, file):
        #Fix this proper one day!!
        self.save_settings(file.replace("weights","settings"))

        # output = {}
        # extrinsic_model_weight_dict, extrinsic_model_name = self.extrinsic_model.get_weights(self.extrinsic_model.all_vars)
        # reference_extrinsic_model_weight_dict, reference_extrinsic_model_name = self.reference_extrinsic_model.get_weights(self.reference_extrinsic_model.all_vars)
        # output["extrinsic_model"]           = ((extrinsic_model_weight_dict)           , extrinsic_model_name          ),
        # output["reference_extrinsic_model"] = ((reference_extrinsic_model_weight_dict) , reference_extrinsic_model_name),
        # if self.settings["use_curiosity"]:
        #     intrinsic_model_weight_dict, intrinsic_model_name = self.intrinsic_model.get_weights(self.intrinsic_model.all_vars)
        #     reference_intrinsic_model_weight_dict, reference_intrinsic_model_name = self.reference_intrinsic_model.get_weights(self.reference_intrinsic_model.all_vars)
        #     output["intrinsic_model"]           = ((intrinsic_model_weight_dict)           , intrinsic_model_name          ),
        #     output["reference_intrinsic_model"] = ((reference_intrinsic_model_weight_dict) , reference_intrinsic_model_name),
        #     ''' Implement this if you really want it... '''
        #     # curiosity_network_weight_dict, curiosity_network_name = self.curiosity_network.get_weights(self.curiosity_network.all_vars)
        #     # output["curiosity_network"]         = ((curiosity_network_weight_dict)         , curiosity_network_name        ),

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
        assert self.settings["n_players"] == 2, "only 2player mode as of yet..."
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

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'log' in d:
            d['log'] = d['log'].name
        return d

    def __setstate__(self, d):
        if 'log' in d:
            d['log'] = logging.getLogger(d['log'])
        self.__dict__.update(d)
