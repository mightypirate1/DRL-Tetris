import multiprocessing as mp
import tensorflow as tf
import numpy as np
import struct
import time
import os

from aux.tf_hooks import quick_summary
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
        self.last_saved_weights = 0
        self.stats = {
                        "t_start"          : None,
                        "t_stop"           : None,
                        "t_total"          : None,
                        "t_training"       : None,
                        "t_loading"        : None,
                        "t_training_total" : 0,
                        "t_loading_total"  : 0,

                        "n_samples_total"  : 0,
                       }
    def __call__(self, *args):
        self.running = True
        self.run(*args)
        self.running = False

    def join(self):
        while self.running:
            time.sleep(10)

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
        self.shared_vars["run_flag"][-1] = 1
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.settings["trainer_gpu_fraction"])

        config = tf.ConfigProto(log_device_placement=False,device_count={'GPU': self.gpu_count})
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
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
            #Init run
            self.stats["t_start"] = time.time()
            self.quick_summary = quick_summary(settings=self.settings, session=session)
            #This is ALL we do. All day long.
            while self.workers_running():
                self.load_worker_data()
                self.do_training()
                self.print_stats()
                self.update_global_clock()
                if self.trainer.n_train_steps["total"] % self.settings["weight_transfer_frequency"] == 0:
                    self.transfer_weights()
                if self.trainer.n_train_steps["total"] % self.settings["trainer_thread_save_freq"] == 0 and self.trainer.n_train_steps["total"] > self.last_saved_weights:
                    self.trainer.save_weights(*utils.weight_location(self.settings,idx=self.trainer.n_train_steps["total"]))
            #Report when done!
            self.load_worker_data()
            print("trainer done. {} data processed.".format(self.trainer.clock))
            self.stats["t_stop"] = time.time()
            self.stats["t_total"] = self.stats["t_stop"] - self.stats["t_start"]
            runtime = self.stats["t_total"]
            self.shared_vars["run_flag"][-1] = 0

    def do_training(self):
        t = time.time()
        if self.settings["single_policy"]:
            trained = self.trainer.do_training()
        else:
            trained0 = self.trainer.do_training(policy=0)
            trained1 = self.trainer.do_training(policy=1)
            trained = trained0 or trained1
        t = time.time() - t
        if trained:
            self.quick_summary.update(self.trainer.output_stats(), time=self.current_step())
        self.stats["t_training"] = t
        self.stats["t_training_total"] += t

    def load_worker_data(self):
        t = time.time() #Tick
        data_from_workers = list()
        while not self.shared_vars["data_queue"].empty():
            try:
                d = self.shared_vars["data_queue"].get()
                data_from_workers.append(d)
            except ValueError:
                while True:
                    print("really really bad")
        if len(data_from_workers) == 0:
            return
        n_samples, avg_length = self.trainer.receive_data(data_from_workers)
        t = time.time() - t #Tock
        self.stats["n_samples_total"] += n_samples
        self.stats["t_loading"] = t
        self.stats["t_loading_total"] += t
        if avg_length > 0:
            frac_train = self.stats["t_training_total"] / (self.stats["t_loading_total"] + self.stats["t_training_total"] + 0.0000001)
            current_speed = self.stats["n_samples_total"] / self.walltime()
            s = {
                 "Average trajectory length" : avg_length,
                 "Current speed"             : current_speed,
                 "Time spent training"       : frac_train,
                }
            self.quick_summary.update(s, time=self.current_step())

    def print_stats(self):
        if time.time() < self.last_print_out + self.print_frequency:
            return
        self.last_print_out = time.time()
        frac_load  = self.stats["t_loading_total"] / (self.stats["t_loading_total"] + self.stats["t_training_total"] + 0.0000001)
        frac_train = self.stats["t_training_total"] / (self.stats["t_loading_total"] + self.stats["t_training_total"] + 0.0000001)
        print("-------trainer info-------")
        print("clock: {}".format(self.current_step()))
        print("trained for {}s".format(self.stats["t_training"]))
        print("loaded  for {}s".format(self.stats["t_loading"]))
        print("fraction in training/loading: {} / {}".format(frac_train,frac_load))
        # print("time to reference update / save: {}".format(self.trainer.time_to_reference_update))

    def transfer_weights(self):
        #Get some data
        n, w = self.trainer.export_weights()
        #Make it the worlds
        self.shared_vars["update_weights_lock"].acquire()
        self.shared_vars["update_weights"]["idx"] = n
        self.shared_vars["update_weights"]["weights"] = w
        self.shared_vars["update_weights_lock"].release()

    def update_global_clock(self):
        self.shared_vars["global_clock"].value = self.trainer.clock

    def workers_running(self):
        for i in self.shared_vars["run_flag"][:-1]: #Don't check the last one.. that is the trainer-thread!
            if i == 1: return True
        return False

    def current_step(self):
        return self.trainer.clock

    def walltime(self):
        return time.time() - self.stats["t_start"]

    def __str__(self):
        return "thread( type={}, ID={})".format(type(self), self.id)
