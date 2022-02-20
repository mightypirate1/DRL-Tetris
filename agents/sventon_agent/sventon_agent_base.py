import numpy as np
import logging
import pickle
import os

#Core internals
import threads
import tools.utils as utils
from agents.agent_utils import state_unpack
#Datatypes for flavours
from agents.networks import ppo_nets, prio_qnet
import agents.datatypes as dt

logger = logging.getLogger(__name__)

class sventon_agent_base:
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
        logger.debug("name created! type={} mode={}".format(self.name,self.mode))
        #Parse settings
        self.settings = utils.parse_settings(settings)
        settings_ok, error = self.process_settings() #Checks so that the settings are not conflicting
        if not settings_ok:
            raise Exception("settings not ok: " + error+"={}".format(self.settings[error]) + " is not not ok")

        #Provide some data-types for flavours!
        flavour = self.flavour = self.settings["sventon_flavour"]
        self.trajectory_type = dt.sventon_trajectory
        if flavour == "ppo":
            self.trainer_type = self.settings["trainer_type"]
            self.network_type = ppo_nets
            self.exp_rep_sample = "empty"
        elif flavour == "dqn":
            self.trainer_type = self.settings["trainer_type"]
            self.network_type = prio_qnet
            self.exp_rep_sample = self.settings["experience_replay_sample_mode"]

        #Some basic core functionality
        self.sandbox = sandbox.copy()
        self.unpack = state_unpack.unpacker(
                                            self.sandbox.get_state(),
                                            observation_mode='separate',
                                            player_mode='separate',
                                            state_from_perspective=True,
                                            separate_piece=True,
                                            piece_in_statevec=self.settings["state_processor_piece_in_statevec"],
                                            )
        #nn-shapes etc
        self.state_size = self.unpack.get_shapes()
        self.n_vec, self.n_vis = len(self.state_size[0]), len(self.state_size[1])
        self.n_rotations = 4
        self.n_translations = self.settings["game_size"][1]
        self.piece_in_statevec = self.settings["state_processor_piece_in_statevec"]
        self.n_pieces = 7 if not self.piece_in_statevec else 1
        self.model_output_shape = [self.n_rotations, self.n_translations, self.n_pieces]
        #
        self.n_envs = 1
        self.n_workers = 1
        #distrubutions
        self.eval_dist = self.settings["eval_distribution"]
        self.train_dist = self.settings["train_distribution"]
        #some helper vars
        self.player_idxs = [p for p in range(self.settings["n_players"])]
        self.workers_do_processing = self.settings["workers_do_processing"]
        #variables
        self.gamma = self.settings["gamma"] if not self.settings["single_policy"] else -self.settings["gamma"]

        if session is not None:
            self.create_models(session)
        else:
            logger.warning(f"tf.Session NOT provided. You will have to create agent-models manually wigh create_model_dict when you have one")

    def run_model(self, net, states, **kwargs):
        player = kwargs.pop('player')
        assert player is not None, "Specify a player to run the model for!"
        vec, vis, piece = self.unpack(states, player)
        return (*net.evaluate((vec, vis), **kwargs), piece[player[0]])

    def model_runner(self, net):
        if type(net) is str:
            net = self.model_dict[net]
        def runner(*args, **kwargs):
            return self.run_model(net, *args, **kwargs)
        return runner

    def create_models(self,session):
        models = ["main_net"] if self.settings["single_policy"] else ["policy_0", "policy_1"]
        model_dict = {}
        worker_only = self.mode == threads.WORKER
        for model in models:
            m = self.network_type(
                self.id,
                model,
                self.state_size,
                self.model_output_shape,
                session,
                worker_only=worker_only,
                settings=self.settings,
            )
            model_dict[model] = m
        self.model_dict = model_dict
        return model_dict

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
                                             main_weights,
                                            )
            if not self.model_dict[net].worker_only:
                self.model_dict[net].set_weights(
                                                 self.model_dict[net].reference_net_assign_list,
                                                 ref_weights
                                                )

    def import_weights(self, weight_list, seed=None): #As passed by the trainer's export_weights-fcn..
        apply_noise = "parameter_noise" in self.settings
        models = sorted([x for x in self.model_dict])
        for m,w in zip(models, weight_list):
            model = self.model_dict[m]
            model.set_weights(model.main_net_assign_list,w, seed=seed, apply_noise=apply_noise)

    def export_weights(self):
        models = sorted([x for x in self.model_dict])
        weights = [self.model_dict[x].get_weights(self.model_dict[x].variables) for x in models]
        return weights

    def process_settings(self):
        #General requirements:
        if self.settings["sventon_flavour"] not in ["dqn", "ppo"]:
            return False, "sventon_flavour"
        if self.settings["n_players"] != 2:
            return False, "n_players"

        #flavour-specific requirements
        if self.settings["sventon_flavour"] == "ppo":
            allowed_dists = ["argmax", "pi"]
            forced_settings = {"experience_replay_sample_mode" : "empty"}
        if self.settings["sventon_flavour"] == "dqn":
            allowed_dists = ["argmax", "pi", "pareto_distribution", "boltzman_distribution", "adaptive_epsilon", "epsilon"]
            forced_settings = {}
        for key in forced_settings.keys():
            if key in self.settings:
                logger.warning("Overriding setting: " + key + " : {}->{}".format(self.settings[key],forced_settings[key]))
        self.settings.update(forced_settings)
        if self.settings["eval_distribution"] not in allowed_dists:
            return False, "eval_distribution"
        if self.settings["train_distribution"] not in allowed_dists:
            return False, "train_distribution"
        return True, None

    #Pickling needed for multiprocessing...
    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("model_dict", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
