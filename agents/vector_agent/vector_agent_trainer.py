import time
import numpy as np
import threads
import tools
import tools.utils as utils
from agents.vector_agent.vector_agent_base import vector_agent_base
import agents.agent_utils as agent_utils
from agents.networks import prio_vnet
from agents.agent_utils.experience_replay import experience_replay
from tools.parameter import *

class vector_agent_trainer(vector_agent_base):
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
        vector_agent_base.__init__(self, id=id, name="trainer{}".format(id), session=session, sandbox=sandbox, settings=settings, mode=mode)
        self.verbose_training = self.settings["run_standalone"]
        self.train_stats_raw = list()
        self.summarizer = summarizer

        #Create models
        self.model_dict = {}
        self.experience_replay_dict = {}
        self.n_train_steps = {}
        self.time_to_reference_update = {}
        self.scoreboard = {}
        models = ["value_net"] if self.settings["single_policy"] else ["policy_0", "policy_1"]
        for model in models:
            m = prio_vnet(
                          "TRAINER",
                          model,
                          self.state_size,
                          session,
                          worker_only=False,
                          k_step=self.settings["n_step_value_estimates"],
                          settings=self.settings,
                         )
            self.model_dict[model] = m
            self.experience_replay_dict[model] = experience_replay(
                                                                    state_size=self.state_size,
                                                                    max_size=int(self.settings["experience_replay_size"]/len(models)),
                                                                    k_step=self.settings["n_step_value_estimates"],
                                                                    sample_mode=self.settings["experience_replay_sample_mode"],
                                                                   )
            self.scoreboard[model] = 0.5
            self.n_train_steps[model] = 0
            self.time_to_reference_update[model] = 0

        self.n_samples_to_start_training = max(self.settings["n_samples_each_update"], self.settings["n_samples_to_start_training"])
        if init_weights is not None:
            print("Trainer{} initialized from weights: {} and clock: {}".format(self.id, init_weights, init_clock))
            self.update_clock(init_clock)
            self.load_weights(init_weights,init_weights)
            self.n_samples_to_start_training = self.settings["restart_training_delay"]
            self.load_weights(init_weights,init_weights)

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
                    d, p = data.process_trajectory(
                                                    self.model_runner(metadata["policy"]),
                                                    self.unpack,
                                                    reward_shaper=self.settings["reward_shaper"](self.settings["reward_shaper_param"](self.clock), single_policy=self.settings["single_policy"]),
                                                    gamma_discount=self.settings["gamma"]
                                                    ) #This is a (s,None,r,s',d) tuple where each entry is a np-array with shape (n,k) where n is the lentgh of the trajectory, and k is the size of that attribute

                else:
                    d, p = data
                if self.settings["single_policy"]:
                    exp_rep = self.experience_replay_dict["value_net"]
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
        #Figure out what policy, model, and experience replay to use...
        if self.settings["single_policy"]:
            policy = "value_net"
        else:
            assert policy is not None, "In multi-policy mode you must specify which model to train!"
            policy = "policy_{}".format(policy)
            if self.scoreboard[policy] > 0.5 + self.settings["winrate_tolerance"]:
                print("Policy \"{}\" did NOT TRAIN, due to too much winning! ({})".format(policy, self.scoreboard[policy]))
                return False
        exp_rep = self.experience_replay_dict[policy]
        model = self.model_dict[policy]

        #If we dont have enough samples yet we quit early...
        if sample is None and len(exp_rep) < self.n_samples_to_start_training:
            if not self.settings["run_standalone"]: time.sleep(1) #If we are a separate thread, we can be a little patient here
            return False

        #Start!
        self.train_lossstats_raw = list()
        self.log.debug("trainer[{}] doing training".format(self.id))

        #Some values:
        minibatch_size = self.settings["minibatch_size"]
        n_epochs       = self.settings["n_train_epochs_per_update"]
        n = self.settings["n_samples_each_update"]

        #Get a sample!
        update_prio_flag = False
        if sample is None: #If no one gave us one, we get one ourselves!
            update_prio_flag = True
            sample, is_weights, filter = \
                            exp_rep.get_random_sample(
                                                        self.settings["n_samples_each_update"],
                                                        alpha=self.settings["prioritized_replay_alpha"].get_value(self.clock),
                                                        beta=self.settings["prioritized_replay_beta"].get_value(self.clock),
                                                        compute_stats=True,
                                                      )
        #Unpack a little...
        states, _, rewards, dones = sample
        vector_states, visual_states = states
        new_prio = np.empty((n,1))

        #TRAIN!
        for t in range(n_epochs):
            if self.verbose_training: print("[",end='',flush=False); last_print = 0
            last_epoch = t+1 == n_epochs
            perm = np.random.permutation(n) if not last_epoch else np.arange(n)
            for i in range(0,n,minibatch_size):
                _new_prio, stats = model.train(
                                               [vec_s[perm[i:i+minibatch_size]] for vec_s in vector_states],
                                               [vis_s[perm[i:i+minibatch_size]] for vis_s in visual_states],
                                               rewards[perm[i:i+minibatch_size]],
                                               dones[perm[i:i+minibatch_size]],
                                               weights=is_weights[perm[i:i+minibatch_size]],
                                               lr=self.settings["value_lr"].get_value(self.clock)
                                              )
                self.train_stats_raw.append(stats)
                if last_epoch: new_prio[i:i+minibatch_size] = _new_prio
                if self.verbose_training and (i-last_print)/n > 0.02: print("-",end='',flush=False); last_print = i
            if self.verbose_training: print("]",flush=False)
        #Sometimes we do a reference update
        if self.time_to_reference_update[policy] == 0:
            self.reference_update(net=policy)
            self.time_to_reference_update[policy] = self.settings["time_to_reference_update"]
        else:
            self.time_to_reference_update[policy] -= 1

        #Count training
        self.n_train_steps['total'] += 1
        self.n_train_steps[policy]  += 1
        #Some stats:
        self.generate_training_stats()
        self.update_stats(exp_rep.stats, scope="ExpRep_"+policy)

        #Update prios if we sampled the sample ourselves...
        if update_prio_flag:
            exp_rep.update_prios(new_prio, filter)

        return True

    def generate_training_stats(self):
        stats, counts = {}, {}
        for train_stat_batch in self.train_stats_raw:
            for tensor, data in train_stat_batch:
                if tensor.name not in stats:
                    stats[tensor.name] = data
                    counts[tensor.name] = 1
                else:
                    if 'max' in tensor.name:
                        stats[tensor.name] = np.maximum(stats[tensor.name], data)
                    elif 'min' in tensor.name:
                        stats[tensor.name] = np.minimum(stats[tensor.name], data)
                    else:
                        stats[tensor.name] += data
                        counts[tensor.name] += 1
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
        pass

    def update_scoreboard(self, winner):
        if type(winner) is not str:
            winner = "policy_{}".format(winner)
        for p in self.scoreboard:
            if p == winner:
                score = 1
            else:
                score = 0
            self.scoreboard[p] = (1-self.settings["winrate_learningrate"]) * self.scoreboard[p] + self.settings["winrate_learningrate"] * score

    def export_weights(self):
        models = sorted([x for x in self.model_dict])
        weights = [self.model_dict[x].get_weights(self.model_dict[x].main_net_vars) for x in models]
        return self.n_train_steps["total"], weights

    #Moves the reference model to be equal to the model, or changes their role (depending on setting)
    def reference_update(self, net=None):
        if net is None:
            nets = [x for x in self.model_dict]
        else:
            nets = [net]
        print("Updating agent{} reference model!".format(self.id))
        for x in nets:
            if self.settings["alternating_models"]:
                self.model_dict[x].swap_networks()
            else:
                self.model_dict[x].reference_update()
