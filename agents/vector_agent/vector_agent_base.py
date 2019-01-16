import threads
import aux.utils as utils
import numpy as np
import logging
import pickle
import os

class vector_agent_base:
    def __init__(
                  self,
                  id=0,
                  name="base_type!",
                  session=None,
                  sandbox=None,
                  settings=None,
                  mode=threads.STANDALONE
                 ):

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
    def save_weights(self, folder, file): #folder is a sub-string of file!  e.g. folder="path/to/folder", file="path/to/folder/file"
        #recommended use for standardized naming is .save_weights(*aux.utils.weight_location(...)) and similarily for the load_weights fcn
        output = {}
        for net in self.model_dict:
            if net is "default": continue
            weight_dict, model_name = self.model_dict[net].get_weights(self.model_dict[net].all_vars)
            output[net] = weight_dict, model_name
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(file, 'wb') as f:
            print("SAVED WEIGHTS TO ",file)
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_weights(self, folder, file):  #folder is a sub-string of file!  e.g. folder="path/to/folder", file="path/to/folder/file"
        with open(file, 'rb') as f:
            input_models = pickle.load(f)
        for net in self.model_dict:
            if net is "default": continue
            weight_dict, model_name = input_models[net]
            self.model_dict[net].set_weights(
                                             self.model_dict[net].all_vars,
                                             input_models[net]
                                            )

    # # # # #
    # Helper fcns                        TODO: MOVE THES TO UTILS!!! <-------
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

    def update_weights(self, w, model=None): #As passed by the trainer's export_weights-fcn..
        if model is None: model = self.model_dict["default"]
        model.set_weights(model.all_vars,w)

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
