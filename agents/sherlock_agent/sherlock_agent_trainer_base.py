import time
import numpy as np
import threads
import tools
import tools.utils as utils
import agents.agent_utils as agent_utils
from agents.agent_utils.experience_replay import experience_replay
from tools.parameter import *
from agents.sherlock_agent.sherlock_agent_base import sherlock_agent_base as sherlock_agent_base

import agents
class sherlock_agent_trainer_base(agents.sherlock_agent.sherlock_agent_base.sherlock_agent_base):
    def __init__(
                 self,
                 id=0,                      # What's this trainers name?
                 session=None,              # The session to operate in
                 sandbox=None,              # Sandbox to play in!
                 mode=threads.ACTIVE,       # What's our role?
                 settings=None,             # Settings-dict passed down from the ancestors
                 init_weights=None,
                 init_clock=0,
                 summarizer=None,
                ):

        #Some general variable initialization etc...
        sherlock_agent_base.__init__(self, id=id, name="trainer{}".format(id), session=session, sandbox=sandbox, settings=settings, mode=mode)
        self.verbose_training = self.settings["run_standalone"]
        self.train_stats_raw = list()

        #Create models
        self.model_dict = {}
        self.experience_replay_dict = {}
        self.n_train_steps = {}
        self.time_to_reference_update = {}
        self.scoreboard = {}
        models = ["main_net"] if self.settings["single_policy"] else ["policy_0", "policy_1"]
        for model in models:
            m = self.network_type(
                                    "TRAINER",
                                    model,
                                    self.state_size, #Input-shape
                                    self.model_output_shape, #Output_shape
                                    session,
                                    worker_only=False,
                                    k_step=self.settings["n_step_value_estimates"],
                                    settings=self.settings,
                                )
            self.model_dict[model] = m
            self.experience_replay_dict[model] = experience_replay(
                                                                    state_size=self.state_size,
                                                                    action_size=[[1,], [1,], [1,], [1,], [1,], [*self.game_size,1], [*self.game_size,1]],
                                                                    max_size=int(self.settings["experience_replay_size"]/len(models)),
                                                                    k_step=self.settings["n_step_value_estimates"],
                                                                    sample_mode=self.settings["experience_replay_sample_mode"],
                                                                   )
            self.scoreboard[model] = 0.5
            self.n_train_steps[model] = 0
            self.time_to_reference_update[model] = 0
        self.n_train_epochs = self.settings["n_train_epochs_per_update"] if not self.settings["dynamic_n_epochs"] else 1
        self.n_train_epochs_lock = False
        self.n_samples_to_start_training = max(self.settings["n_samples_each_update"], self.settings["n_samples_to_start_training"])
        if init_weights is not None:
            print("Trainer{} initialized from weights: {} and clock: {}".format(self.id, init_weights, init_clock))
            self.update_clock(init_clock)
            self.load_weights(init_weights,init_weights)
            self.n_samples_to_start_training = self.settings["restart_training_delay"]
            self.load_weights(init_weights,init_weights)

        self.summarizer = summarizer
        self.conv_visualizations = {}
        self.n_train_steps["total"] = 0
        self.n_samples_from = [0 for _ in range(self.settings["n_workers"])]
        self.train_time_log = {
                                "total"         : 0.0,
                                "sample"        : 0.0,
                                "unpack"        : 0.0,
                                "train"         : 0.0,
                                "ref_update"    : 0.0,
                                "update_sample" : 0.0,
                              }

    #What if someone just sends us some experiences?! :D
    def receive_data(self, data_list):
        if len(data_list) == 0: return
        if type(data_list[0]) is list:
            input_data = list()
            for d in data_list:
                input_data += d
        else: input_data = data_list
        n_trajectories, tot = 0, 0
        for metadata,data in input_data:
                n_trajectories += 1
                # print("trainer recieved:", metadata["worker"], metadata["packet_id"], "len", metadata["length"])
                if not self.settings["workers_do_processing"]:
                    assert False, "this line is not tested"
                    d, p = data.process_trajectory(
                                                    self.model_runner(metadata["policy"]),
                                                    self.unpack,
                                                    gamma_discount=self.gamma,
                                                    compute_advantages=False,
                                                    gae_lambda=self.settings["gae_lambda"],
                                                    augment=self.settings["augment_data"],
                                                    )

                else:
                    d, p = data
                if self.settings["single_policy"]:
                    exp_rep = self.experience_replay_dict["main_net"]
                else:
                    exp_rep = self.experience_replay_dict["policy_{}".format(metadata["policy"])]
                exp_rep.add_samples(d,p)
                self.n_samples_from[metadata["worker"]] += metadata["length"]
                self.update_scoreboard(metadata["winner"])
                tot += metadata["length"]
        self.clock += tot
        avg = tot/n_trajectories if n_trajectories>0 else 0
        return tot, avg

    def do_training(self, sample=None, policy=None):
        raise Exception("sherlock_trainer_base is abstract!")

    def adjust_n_epochs(self, sample_size):
        if not self.settings["dynamic_n_epochs"]:
            return
        frac_samples = sample_size / self.settings["n_samples_each_update"]
        frac_epochs = self.n_train_epochs / max(self.n_train_epochs - 1,1)
        if frac_samples / frac_epochs > 1:
            self.n_train_epochs = max(1, self.n_train_epochs - 1)
        self.n_train_epochs_lock = False
        print("DBG: current n_epochs:", self.n_train_epochs)
    def adjust_epochs_up(self):
        if not self.settings["dynamic_n_epochs"]:
            return
        if self.clock > self.settings["n_samples_each_update"] and not self.n_train_epochs_lock:
            self.n_train_epochs = min(self.settings["n_train_epochs_per_update"], self.n_train_epochs + 1 )
            self.n_train_epochs_lock = True
            print("DBG: bump up!")

    def generate_training_stats(self):
        stats, counts = {}, {}
        for train_stat_batch in self.train_stats_raw:
            for tensor, data in train_stat_batch:
                key = tensor if type(tensor) is str else tensor.name #sometimes I just want to wedge in a number :)
                if key not in stats:
                    stats[key] = data
                    counts[key] = 1
                else:
                    if 'max' in key:
                        stats[key] = np.maximum(stats[key], data)
                    elif 'min' in key:
                        stats[key] = np.minimum(stats[key], data)
                    else:
                        stats[key] += data
                        counts[key] += 1
        for tensor_name in stats:
            stats[tensor_name] = stats[tensor_name] / counts[tensor_name]
        self.train_stats_raw.clear()
        self.stats.update(stats)
        return self.stats

    def update_stats(self, stats, scope=None):
        if scope is None:
            scope = ""
        else:
            scope += "/"
        for key in stats:
            self.stats[scope+key] = stats[key]

    def report_stats(self):
        self.summarizer.update(self.stats, time=self.clock)
        self.summarizer.image(self.conv_visualizations, time=self.clock)

    def update_scoreboard(self, winner):
        if type(winner) is not str:
            winner = "policy_{}".format(winner)
        for p in self.scoreboard:
            if p == winner:
                score = 1
            else:
                score = 0
            self.scoreboard[p] = (1-self.settings["winrate_learningrate"]) * self.scoreboard[p] + self.settings["winrate_learningrate"] * score

    #Moves the reference model to be equal to the model, or changes their role (depending on setting)
    def reference_update(self, net=None):
        if net is None:
            nets = [x for x in self.model_dict]
        else:
            nets = [net]
        print("Updating agent{} reference model!".format(self.id))
        for x in nets:
            self.model_dict[x].reference_update()
