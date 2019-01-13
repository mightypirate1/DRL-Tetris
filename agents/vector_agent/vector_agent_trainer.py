import time
import numpy as np
import threads
import aux
import aux.utils as utils
from agents.vector_agent.vector_agent_base import vector_agent_base
import agents.agent_utils as agent_utils
from agents.networks import value_net
from agents.agent_utils.experience_replay import experience_replay
from aux.parameter import *


class vector_agent_trainer(vector_agent_base):
    def __init__(
                 self,
                 id=0,                      # What's this trainers name?
                 session=None,              # The session to operate in
                 sandbox=None,              # Sandbox to play in!
                 shared_vars=None,          # This is to send data between nodes
                 mode=threads.ACTIVE,       # What's our role?
                 settings=None,             # Settings-dict passed down from the ancestors
                ):

        #Some general variable initialization etc...
        vector_agent_base.__init__(self, id=id, name="trainer{}".format(id), session=session, sandbox=sandbox, shared_vars=shared_vars, settings=settings, mode=mode)
        self.n_train_steps = 0
        self.global_clock = 0
        #Bucket of data
        self.experience_replay = agent_utils.experience_replay(self.settings["experience_replay_size"], prioritized=self.settings["prioritized_experience_replay"])

        #Models
        self.reference_extrinsic_model = value_net(self.id, "reference_extrinsic", self.state_size, session, settings=self.settings, on_cpu=self.settings["worker_net_on_cpu"])
        self.extrinsic_model = value_net(
                                         self.id,
                                         "main_extrinsic",
                                         self.state_size,
                                         session,
                                         settings=self.settings,
                                         on_cpu=self.settings["trainer_net_on_cpu"]
                                        )
        self.model_dict = {
                            "reference_extrinsic_model" : self.reference_extrinsic_model,
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
        for data in data_list:
            self.experience_replay.merge_in(data)
            self.global_clock += len(data)
    def export_weights(self):
        return self.n_train_steps, self.model_dict["default"].get_weights(self.model_dict["default"].all_vars)
        #return n_steps, (m_weight_dict, m_name)

    def do_training(self):
        if len(self.experience_replay) < self.settings["n_samples_each_update"]:
            time.sleep(5)
            return

        #Start!
        t_sample = time.time()
        self.log.debug("trainer[{}] doing training".format(self.id))

        #Some values:
        minibatch_size = self.settings["minibatch_size"]
        n_epochs       = self.settings["n_train_epochs_per_update"]
        n              = self.settings["n_samples_each_update"]

        #Get a sample!
        sample, filter = self.experience_replay.get_random_sample(
                                                                  n,
                                                                  alpha=self.settings["prioritized_replay_alpha"].get_value(self.global_clock),
                                                                  beta=self.settings["prioritized_replay_beta"].get_value(self.global_clock),
                                                                 )
        #Put it in arrays
        t_unpack = time.time()
        states        = np.zeros((n,*self.state_size ))
        target_values = np.zeros((n,1 ))
        is_weights    = np.zeros((n,1 ))
        for i,s in enumerate(sample):
            s,p,tv, isw        = s.get_data()
            states[i,:]        = self.state_from_perspective(s, p)
            target_values[i,:] = tv
            is_weights[i,:]    = isw
        #TRAIN!
        t_train = time.time()
        for t in range(n_epochs):
            perm = np.random.permutation(n)
            for i in range(0,n,minibatch_size):
                self.extrinsic_model.train(
                                            states[perm[i:i+minibatch_size]],
                                            target_values[perm[i:i+minibatch_size]],
                                            weights=is_weights[perm[i:i+minibatch_size]],
                                            lr=self.settings["value_lr"].get_value(self.global_clock)
                                           )
        #Sometimes we do a reference update
        t_ref_update = time.time()
        if self.time_to_reference_update == 0:
            self.reference_update()
            self.time_to_reference_update = self.settings["time_to_reference_update"]
        else:
            self.time_to_reference_update -= 1

        #Update all samples! (priorities and target_value are re-computed using the default model)
        t_update_sample = time.time()
        for i in filter:
            sample[i].update_value(self.run_default_model)
        t_done = time.time()

        #Keep stats and tell the world
        self.train_time_log["sample"]         +=  t_unpack        -  t_sample
        self.train_time_log["unpack"]         +=  t_train         -  t_unpack
        self.train_time_log["train"]          +=  t_ref_update    -  t_train
        self.train_time_log["ref_update"]     +=  t_update_sample -  t_ref_update
        self.train_time_log["update_sample"]  +=  t_done          -  t_update_sample
        self.train_time_log["total"]          +=  t_done          -  t_sample
        self.output_stats()

        #Count training
        self.n_train_steps += 1

    def output_stats(self):
        print("----Trainer stats (step {})".format(self.n_train_steps))
        print("train time log:")
        for x in self.train_time_log:
            print("\t{} : {}".format(x.rjust(15), self.train_time_log[x]))
        # print("---")
        # print("TAU:{}".format(self.avg_trajectory_length))
        # print("EPSILON:{}".format(self.settings["epsilon"].get_value(self.global_clock)/self.avg_trajectory_length))
        # print("curiosity_amount:{}".format(self.settings["curiosity_amount"].get_value(self.global_clock)))
        # print("value_lr:{}".format(self.settings["value_lr"].get_value(self.global_clock)))
        # print("---")

    #Moves the reference model to be equal to the model, or changes their role (depending on setting)
    def reference_update(self):
        print("Updating agent{} reference model!".format(self.id))
        if self.settings["alternating_models"]:
            #alternate extrinsic
            tmp = self.reference_extrinsic_model
            self.reference_extrinsic_model = self.extrinsic_model
            self.extrinsic_model = tmp
            self.model_dict["default"] = self.reference_extrinsic_model
        else:
            weights = self.extrinsic_model.get_weights(self.extrinsic_model.trainable_vars)
            self.reference_extrinsic_model.set_weights(self.reference_extrinsic_model.trainable_vars,weights)
