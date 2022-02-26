import time
import numpy as np
import logging
import threads
import tools
import tools.utils as utils
import agents.agent_utils as agent_utils
from agents.agent_utils.experience_replay import experience_replay
from tools.parameter import *
from agents.sventon_agent.sventon_agent_base import sventon_agent_base as sventon_agent_base

logger = logging.getLogger(__name__)

class sventon_agent_trainer_base(sventon_agent_base):
    def __init__(
                 self,
                 id=0,                      # What's this trainers name?
                 session=None,              # The session to operate in
                 sandbox=None,              # Sandbox to play in!
                 mode=threads.ACTIVE,       # What's our role?
                 settings=None,             # Settings-dict passed down from the ancestors
                ):

        #Some general variable initialization etc...
        sventon_agent_base.__init__(self, id=id, name="trainer{}".format(id), session=session, sandbox=sandbox, settings=settings, mode=mode)
        self.verbose_training = self.settings["run_standalone"]

        #Create models
        self.experience_replay_dict = {}
        self.n_train_steps = {}
        self.time_to_reference_update = {}
        models = ["main_net"] if self.settings["single_policy"] else ["policy_0", "policy_1"]

        for model in models:
            self.experience_replay_dict[model] = experience_replay(
                state_size=self.state_size,
                action_size=[[3,],[3,]], #(rotation, translation, piece) x 2 players
                max_size=int(self.settings["experience_replay_size"]/len(models)),
                k_step=self.settings["n_step_value_estimates"],
                sample_mode=self.settings["experience_replay_sample_mode"],
            )
            self.time_to_reference_update[model] = 0

        self.n_train_epochs = self.settings["n_train_epochs_per_update"]
        self.n_samples_to_start_training = max(self.settings["n_samples_each_update"], self.settings["n_samples_to_start_training"])

    #What if someone just sends us some experiences?! :D
    def receive_data(self, data_list, time=0):
        if len(data_list) == 0:
            return 0, 0
        if type(data_list[0]) is list:
            input_data = list()
            for d in data_list:
                input_data += d
        else: input_data = data_list
        n_trajectories, tot = 0, 0
        for metadata,data in input_data:
                n_trajectories += 1
                if not self.settings["workers_do_processing"]:
                    assert False, "this line is not tested"
                    d, p = data.process_trajectory(
                        self.model_runner(metadata["policy"]),
                        self.unpack,
                        gamma_discount=self.gamma,
                        compute_advantages=False,
                        gae_lambda=tools.parameter.param_eval(self.settings["gae_lambda"], time),
                        augment=self.settings["augment_data"],
                    )

                else:
                    d, p = data
                if self.settings["single_policy"]:
                    exp_rep = self.experience_replay_dict["main_net"]
                else:
                    exp_rep = self.experience_replay_dict["policy_{}".format(metadata["policy"])]
                exp_rep.add_samples(d,p)
                tot += metadata["length"]
        avg = tot/n_trajectories if n_trajectories>0 else 0
        return tot, avg

    def do_training(self, sample=None, policy=None):
        raise Exception("sventon_trainer_base is abstract!")

    #Moves the reference model to be equal to the model, or changes their role (depending on setting)
    def reference_update(self, net=None):
        if net is None:
            nets = [x for x in self.model_dict]
        else:
            nets = [net]
        print("Updating agent{} reference model!".format(self.id))
        for x in nets:
            self.model_dict[x].reference_update()
