import multiprocessing as mp
import tensorflow as tf
import numpy as np
import struct
import time
import os

from aux.settings import default_settings
import aux.utils as utils
import threads

class trainer_thread(mp.Process):
    def __init__(self, id=id, settings=None, shared_vars=None, patience=0.1):
        mp.Process.__init__(self, target=self)
        settings["render"] = False #Trainers dont render.
        self.settings = utils.parse_settings(settings)
        self.id = id
        self.patience = patience
        self.gpu_count = 0 if self.settings["trainer_net_on_cpu"] else 1
        self.last_global_clock = 0
        self.running = False
        self.shared_vars = shared_vars
        self.print_frequency = 10
        self.last_print_out = 0
        self.stats = {
                        "t_start"          : None,
                        "t_stop"           : None,
                        "t_total"          : None,
                        "t_training"       : None,
                        "t_loading"        : None,
                        "t_training_total" : 0,
                        "t_loading_total"  : 0,
                       }
    def __call__(self, *args):
        self.running = True
        self.run(*args)
        self.running = False

    def run(self, *args):
        # self.var_test()
        '''
        Main code [TRAINER]:
        '''
        myid=mp.current_process()._identity[0]
        np.random.seed(myid^struct.unpack("<L",os.urandom(4))[0])
        # Be Nice
        niceness=os.nice(0)
        # os.nice(niceness-3) #Not allowed it seems :)
        with tf.Session(config=tf.ConfigProto(log_device_placement=False,device_count={'GPU': self.gpu_count})) as session:
            #Initialize!
            self.trainer = self.settings["trainer_type"](
                                                         id=self.id,
                                                         mode=threads.ACTIVE,
                                                         sandbox=self.settings["env_type"](settings=self.settings),
                                                         session=session,
                                                         settings=self.settings,
                                                        )

            #
            ##Run!
            #####
            #Thread-logics
            self.running = True
            #Init run
            self.stats["t_start"] = time.time()

            #This is ALL we do. All day long.
            while self.workers_running():
                sample = self.get_sample()
                self.do_training(sample)
                self.return_sample()
                self.print_stats()
                self.update_clock()
                if self.trainer.n_train_steps % self.settings["weight_transfer_frequency"] == 0:
                    self.transfer_weights()
                if self.trainer.n_train_steps % 1000 == 0:
                    self.trainer.save_weights(*utils.weight_location(self.settings,idx=self.trainer.n_train_steps))

            #Report when done!
            print("trainer done")
            self.stats["t_stop"] = time.time()
            self.stats["t_total"] = self.stats["t_stop"] - self.stats["t_start"]
            runtime = self.stats["t_total"]

    def do_training(self, sample):
        t = time.time()
        self.trainer.do_training(sample=sample)
        t = time.time() - t
        self.stats["t_training"] = t
        self.stats["t_training_total"] += t

    def print_stats(self):
        if time.time() < self.last_print_out + self.print_frequency:
            return
        self.last_print_out = time.time()
        frac_load  = self.stats["t_loading_total"] / (self.stats["t_loading_total"] + self.stats["t_training_total"])
        frac_train = self.stats["t_training_total"] / (self.stats["t_loading_total"] + self.stats["t_training_total"])
        print("-------trainer info-------")
        print("trained for {}s".format(self.stats["t_training"]))
        print("loaded  for {}s".format(self.stats["t_loading"]))
        print("fraction in training/loading: {} / {}".format(frac_train,frac_load))
        print("time to reference update / save: {}".format(self.trainer.time_to_reference_update))
        # self.trainer.output_stats()
        self.shared_vars["trainer_stats"]["Time loading"] = frac_load

    def get_sample(self):
        t = time.time()
        # print("requesting SAMPLE...")
        sample = self.shared_vars["trainer_feed"].get()
        # print("found SAMPLE! :D")
        t = time.time() - t
        self.stats["t_loading"] = t
        self.stats["t_loading_total"] += t
        return sample

    def return_sample(self):
        self.shared_vars["trainer_return"].put(1) #Thread language for "I ate a sample you gave me!"

    def transfer_weights(self):
        #Get some data
        n, w = self.trainer.export_weights()
        #Make it the worlds
        self.shared_vars["update_weights_lock"].acquire()
        self.shared_vars["update_weights"]["idx"] = n
        self.shared_vars["update_weights"]["weights"] = w
        self.shared_vars["update_weights_lock"].release()

    def update_clock(self):
        if self.shared_vars["global_clock"].value > self.last_global_clock:
            self.last_global_clock = self.shared_vars["global_clock"].value
            self.trainer.update_clock(self.last_global_clock)

    def workers_running(self):
        for i in self.shared_vars["run_flag"]:
            if i == 1: return True
        return False

    def __str__(self):
        return "thread( type={}, ID={})".format(type(self), self.id)
