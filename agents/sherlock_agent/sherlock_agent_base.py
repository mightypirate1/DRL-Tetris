import numpy as np
import logging
import pickle
import os

#Core internals
import threads
import tools.utils as utils
from agents.agent_utils import state_unpack
#Datatypes for flavours
from agents.networks import delta_ppo_nets
import agents.datatypes as dt

class sherlock_agent_base:
    def __init__(
                  self,
                  id=0,
                  name="base_type!",
                  session=None,
                  sandbox=None,
                  settings=None,
                  mode=threads.STANDALONE
                 ):
        #init!
        self.id = id
        self.name = name
        self.mode = mode
        #Logger
        self.logger = logging.getLogger(self.name)
        self.logger.debug("name created! type={} mode={}".format(self.name,self.mode))
        #Parse settings
        self.settings = utils.parse_settings(settings)
        settings_ok, error = self.process_settings() #Checks so that the settings are not conflicting
        if not settings_ok:
            raise Exception("settings not ok: " + error+"={}".format(self.settings[error]) + " is not not ok")

        #Provide some data-types !
        self.trajectory_type = dt.sherlock_trajectory
        self.trainer_type = self.settings["trainer_type"]
        self.network_type = delta_ppo_nets
        self.exp_rep_sample = "empty"

        #Some basic core functionality
        self.sandbox = sandbox.copy()
        self.unpack = state_unpack.unpacker(
                                            self.sandbox.get_state(),
                                            observation_mode='separate',
                                            player_mode='separate',
                                            state_from_perspective=True,
                                            separate_piece=self.settings["state_processor_separate_piece"],
                                            piece_in_statevec=self.settings["state_processor_piece_in_statevec"],
                                            )
        #nn-shapes etc
        self.state_size = self.unpack.get_shapes()
        self.n_vec, self.n_vis = len(self.state_size[0]), len(self.state_size[1])
        self.piece_in_statevec = self.settings["state_processor_piece_in_statevec"]
        self.n_pieces = 7 if self.settings["separate_piece_values"] else 1
        self.game_size = self.settings["game_size"]
        self.model_output_shape = [*self.game_size, self.n_pieces]
        #distrubutions
        self.eval_dist = self.settings["eval_distribution"]
        self.train_dist = self.settings["train_distribution"]
        #some helper vars
        self.player_idxs = [p for p in range(self.settings["n_players"])]
        self.workers_do_processing = self.settings["workers_do_processing"]
        #variables
        self.gamma = self.settings["gamma"] if not self.settings["single_policy"] else -self.settings["gamma"]
        #we gunna need some models
        self.model_dict = {}
        #stats etc
        self.clock = 0
        self.stats = {}

    def update_clock(self, clock):
        old_clock = self.clock
        self.clock = clock

    def run_model(self, net, states, **kwargs):
        player = kwargs.pop('player')
        assert player is not None, "Specify a player to run the model for!"
        vec, vis, piece = self.unpack(states, player)
        return (*net.evaluate((vec, vis), **kwargs), piece[player[0]])

    def model_runner(self, net):
        if type(net) is str:
            net = self.model_dict[net]
        def runner(data, player=None):
            return self.run_model(net, data, player=player)
        return runner

    # # # # #
    # Memory management fcns
    # # #
    def save_weights(self, folder, file, verbose=False): #folder is a sub-string of file!  e.g. folder="path/to/folder", file="path/to/folder/file"
        #recommended use for standardized naming is .save_weights(*tools.utils.weight_location(...)) and similarily for the load_weights fcn
        output = {}
        for net in self.model_dict:
            if net is "default": continue
            weights = self.model_dict[net].get_weights(self.model_dict[net].main_net.variables), self.model_dict[net].get_weights(self.model_dict[net].ref_net.variables)
            output[net] = weights
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(file, 'wb') as f:
            if verbose: print("SAVED WEIGHTS TO ",file)
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

    def import_weights(self, weight_list): #As passed by the trainer's export_weights-fcn..
        models = sorted([x for x in self.model_dict])
        for m,w in zip(models, weight_list):
            model = self.model_dict[m]
            model.set_weights(model.main_net_assign_list,w)

    def export_weights(self):
        models = sorted([x for x in self.model_dict])
        weights = [self.model_dict[x].get_weights(self.model_dict[x].variables) for x in models]
        return weights

    def process_settings(self):
        #General requirements:
        if self.settings["n_players"] != 2:
            return False, "n_players"
        allowed_dists = ["argmax", "pi"]
        forced_settings = {"experience_replay_sample_mode" : "empty"}
        for key in forced_settings.keys():
            if key in self.settings:
                self.logger.warning("Overriding setting: " + key + " : {}->{}".format(self.settings[key],forced_settings[key]))
        self.settings.update(forced_settings)
        if self.settings["eval_distribution"] not in allowed_dists:
            return False, "eval_distribution"
        if self.settings["train_distribution"] not in allowed_dists:
            return False, "train_distribution"
        return True, None
