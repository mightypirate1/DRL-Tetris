import time
import numpy as np
import threads
import aux
import aux.utils as utils
from agents.vector_agent.vector_agent_base import vector_agent_base
import agents.agent_utils as agent_utils
from agents.networks import prio_vnet
from agents.agent_utils.experience_replay import experience_replay
from aux.parameter import *
from agents.agent_utils import state_fcns


class vector_agent_trainer(vector_agent_base):
    def __init__(
                 self,
                 id=0,                      # What's this trainers name?
                 session=None,              # The session to operate in
                 sandbox=None,              # Sandbox to play in!
                 mode=threads.ACTIVE,       # What's our role?
                 settings=None,             # Settings-dict passed down from the ancestors
                ):

        #Some general variable initialization etc...
        vector_agent_base.__init__(self, id=id, name="trainer{}".format(id), session=session, sandbox=sandbox, settings=settings, mode=mode)
        self.verbose_training = self.settings["run_standalone"]
        self.n_train_steps = 0

        #Bucket of data
        self.experience_replay = experience_replay(
                                                    max_size=self.settings["experience_replay_size"],
                                                    state_size=self.state_size,
                                                   )

        #Models
        self.extrinsic_model = prio_vnet(
                                         self.id,
                                         "prio_vnet",
                                         self.state_size,
                                         session,
                                         settings=self.settings,
                                         on_cpu=self.settings["trainer_net_on_cpu"]
                                        )
        self.model_dict = {
                            "extrinsic_model"           : self.extrinsic_model,
                            "default"                   : self.extrinsic_model,
                          }

        self.time_to_reference_update = 0#self.settings['time_to_reference_update']
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
            data = list()
            for d in data_list:
                data += d
        else: data = data_list

        tot = 0
        n = 0
        for trajectory in data:
                n += 1
                data, prio = trajectory.process_trajectory(self.run_default_model, state_fcns.states_from_perspective) #This is a (s,None,r,s',d) tuple where each entry is a np-array with shape (n,k) where n is the lentgh of the trajectory, and k is the size of that attribute
                self.experience_replay.add_samples(data,prio)
                tot += len(trajectory)
        self.clock += tot
        avg = tot/n if n>0 else 0
        return tot, avg

    def unpack_sample(self, sample):
        #Put it in arrays
        n = self.settings["n_samples_each_update"]
        states        = np.zeros((n,*self.state_size ))
        target_values = np.zeros((n,1 ))
        is_weights    = np.zeros((n,1 ))
        for i,s in enumerate(sample):
            s,p,tv, isw        = s.get_data()
            states[i,:]        = self.state_from_perspective(s, p)
            target_values[i,:] = tv
            is_weights[i,:]    = isw
        return states, target_values, is_weights

    def do_training(self, sample=None):
        if sample is None and len(self.experience_replay) < self.settings["n_samples_each_update"]:
            if not self.settings["run_standalone"]: time.sleep(1) #If we are a separate thread, we can be a little patient here
            return None, None

        #Start!
        self.log.debug("trainer[{}] doing training".format(self.id))

        #Some values:
        minibatch_size = self.settings["minibatch_size"]
        n_epochs       = self.settings["n_train_epochs_per_update"]
        n = self.settings["n_samples_each_update"]

        #Get a sample!
        if sample is None: #If no one gave us one, we get one ourselves!
            sample, is_weights, filter = self.experience_replay.get_random_sample(
                                                                                  self.settings["n_samples_each_update"],
                                                                                  alpha=self.settings["prioritized_replay_alpha"].get_value(self.clock),
                                                                                  beta=self.settings["prioritized_replay_beta"].get_value(self.clock),
                                                                                 )

        states, _, rewards, s_primes, dones = sample
        new_prio = np.empty((n,1))
        #TRAIN!
        for t in range(n_epochs):
            last_epoch = t+1 == n_epochs
            if self.verbose_training: print("[",end='',flush=False); last_print = 0
            perm = np.random.permutation(n) if not last_epoch else np.arange(n)
            for i in range(0,n,minibatch_size):
                _new_prio = self.extrinsic_model.train(
                                                      states[perm[i:i+minibatch_size]],
                                                      rewards[perm[i:i+minibatch_size]],
                                                      s_primes[perm[i:i+minibatch_size]],
                                                      dones[perm[i:i+minibatch_size]],
                                                      weights=is_weights[perm[i:i+minibatch_size]],
                                                      lr=self.settings["value_lr"].get_value(self.clock)
                                                     )
                if last_epoch: new_prio[i:i+minibatch_size] = _new_prio
                if self.verbose_training and (i-last_print)/n > 0.02: print("-",end='',flush=False); last_print = i
            if self.verbose_training: print("]",flush=False)
        #Sometimes we do a reference update
        if self.time_to_reference_update == 0:
            self.reference_update()
            self.time_to_reference_update = self.settings["time_to_reference_update"]
        else:
            self.time_to_reference_update -= 1

        #Count training
        self.n_train_steps += 1
        return new_prio, filter

    def output_stats(self):
        print("----Trainer stats (step {}, clock {})".format(self.n_train_steps, self.clock))
        print("train time log:")
        for x in self.train_time_log:
            print("\t{} : {}".format(x.rjust(15), self.train_time_log[x]))
        # print("---")
        # print("TAU:{}".format(self.avg_trajectory_length))
        # print("EPSILON:{}".format(self.settings["epsilon"].get_value(self.clock)/self.avg_trajectory_length))
        # print("curiosity_amount:{}".format(self.settings["curiosity_amount"].get_value(self.clock)))
        # print("value_lr:{}".format(self.settings["value_lr"].get_value(self.clock)))
        # print("---")

    #Moves the reference model to be equal to the model, or changes their role (depending on setting)
    def reference_update(self):
        print("Updating agent{} reference model!".format(self.id))
        if self.settings["alternating_models"]:
            self.extrinsic_model.swap_networks()
        else:
            weights = self.extrinsic_model.get_weights(self.extrinsic_model.main_net_vars)
            self.extrinsic_model.set_weights(self.extrinsic_model.main_net_assign_list,weights)

    def export_weights(self):
        return self.n_train_steps, self.model_dict["default"].get_weights(self.model_dict["default"].main_net_vars)
        #return n_steps, (m_weight_dict, m_name)
