import numpy as np
import logging
import pickle
import os

import threads
import aux.utils as utils
from agents.agent_utils import state_unpack

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
        self.stats = {}

        #Some basic core functionality
        self.sandbox = sandbox.copy()
        self.unpack = state_unpack.unpacker(
                                            self.sandbox.get_state(),
                                            observation_mode='separate' if self.settings["field_as_image"] else 'vector',
                                            player_mode='separate' if self.settings["players_separate"] else 'vector',
                                            state_from_perspective=self.settings["relative_state"],
                                            )
        self.state_size = self.unpack.get_shapes()
        self.n_vec, self.n_vis = len(self.state_size[0]), len(self.state_size[1])
        self.model_dict = {}

    def update_clock(self, clock):
        old_clock = self.clock
        self.clock = clock
        print("{} UPDATED CLOCK {} -> {}".format(self.id,old_clock,clock))

    def run_model(self, net, states, player=None):
        assert player is not None, "Specify a player to run the model for!"
        states_np = self.unpack(states, player)
        return net.evaluate(states_np)

    def model_runner(self, net):
        if type(net) is str:
            _net = self.model_dict["net"]
        else:
            _net = net
        def runner(data, player=None):
            return self.run_model(_net, data, player=player)
        return runner

    # # # # #
    # Memory management fcns
    # # #
    def save_weights(self, folder, file): #folder is a sub-string of file!  e.g. folder="path/to/folder", file="path/to/folder/file"
        #recommended use for standardized naming is .save_weights(*aux.utils.weight_location(...)) and similarily for the load_weights fcn
        output = {}
        for net in self.model_dict:
            if net is "default": continue
            weights = self.model_dict[net].get_weights(self.model_dict[net].main_net_vars), self.model_dict[net].get_weights(self.model_dict[net].reference_net_vars)
            output[net] = weights
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(file, 'wb') as f:
            print("SAVED WEIGHTS TO ",file)
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(folder+"/settings", 'wb') as f:
            pickle.dump(self.settings, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_weights(self, folder, file):  #folder is a sub-string of file!  e.g. folder="path/to/folder", file="path/to/folder/file"
        assert folder in file, "folder is supposed to be a substring of file"
        with open(file, 'rb') as f:
            input_models = pickle.load(f)
        for net in self.model_dict:
            main_weights, ref_weights = input_models[net]
            self.model_dict[net].set_weights(
                                             self.model_dict[net].main_net_assign_list,
                                             main_weights
                                            )
            if not self.model_dict[net].worker_only:
                self.model_dict[net].set_weights(
                                                 self.model_dict[net].reference_net_assign_list,
                                                 ref_weights
                                                )

    def update_weights(self, weight_list): #As passed by the trainer's export_weights-fcn..
        models = sorted([x for x in self.model_dict])
        for m,w in zip(models, weight_list):
            model = self.model_dict[m]
            model.set_weights(model.main_net_assign_list,w)

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
