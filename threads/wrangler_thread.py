from collections import deque
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import struct
import time
import os

from aux.settings import default_settings
import aux.utils as utils
import threads

WRANGLER_DEBUG = True

class wrangler_thread(mp.Process):
    def __init__(self, id=id, settings=None, shared_vars=None, patience=0.1):
        mp.Process.__init__(self, target=self)
        settings["render"] = False #Trainers dont render.
        self.settings = utils.parse_settings(settings)
        self.id = id
        self.patience = patience
        self.gpu_count = 0 if self.settings["worker_net_on_cpu"] else 1
        self.running = False
        self.shared_vars = shared_vars
        self.target_trainer_queue_length = 20
        self.current_weights = 0 #I have really old weights
        self.sample_buffer = deque(maxlen=5)
        self.stats = {
                        "t_start"          : None,
                        "t_stop"           : None,
                        "t_total"          : None,
                        "t_updating"       : None,
                        "t_loading"        : None,
                        "t_updating_total" : 0,
                        "t_loading_total"  : 0,

                        "n_samples_total"  : 0,
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
        # os.nice(niceness-5) #Not allowed it seems :)
        with tf.Session(config=tf.ConfigProto(log_device_placement=False,device_count={'GPU': self.gpu_count})) as session:
            #Initialize!
            self.wrangler = self.settings["trainer_type"](
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
            clock = -1
            while self.workers_running():
                clock += 1
                self.load_worker_data()
                self.feed_trainer()
                self.replace_trainer_samples()
                self.print_stats()
                if clock % 10 == 0:
                    self.set_global_clock()
                    self.update_weights()

            #Report when done!
            self.stats["t_stop"] = time.time()
            self.stats["t_total"] = self.stats["t_stop"] - self.stats["t_start"]
            runtime = self.stats["t_total"]

    def load_worker_data(self):
        if WRANGLER_DEBUG: print("WRANGLER entering: load_worker_data")
        t = time.time()
        data = list()
        while not self.shared_vars["data_queue"].empty():
            try:
                d = self.shared_vars["data_queue"].get()
                self.stats["n_samples_total"] += len(d)
                data.append(d)
            except ValueError:
                while True:
                    print("really really bad")
        self.wrangler.receive_data(data)
        t = time.time() - t
        self.stats["t_loading"] = t
        self.stats["t_loading_total"] += t
        if WRANGLER_DEBUG: print("WRANGLER exiting: load_worker_data")

    def set_global_clock(self):
        if WRANGLER_DEBUG: print("WRANGLER entering: set_global_clock")
        with self.shared_vars["global_clock"].get_lock():
            self.shared_vars["global_clock"].value = self.wrangler.clock
        if WRANGLER_DEBUG: print("WRANGLER exiting: set_global_clock")

    def update_weights(self):
        if WRANGLER_DEBUG: print("WRANGLER entering: update_weights")
        if self.shared_vars["update_weights"]["idx"] > self.current_weights:
            self.shared_vars["update_weights_lock"].acquire()
            w = self.shared_vars["update_weights"]["weights"]
            self.current_weights = self.shared_vars["update_weights"]["idx"]
            self.shared_vars["update_weights_lock"].release()
            print("AGENT{} updates to weight_{}!!!".format(self.id, self.current_weights))
            self.wrangler.update_weights(w)
        if WRANGLER_DEBUG: print("WRANGLER exiting: update_weights")

    def feed_trainer(self):
        if WRANGLER_DEBUG: print("WRANGLER entering: feed_trainer")
        if len(self.wrangler.experience_replay) < self.settings["n_samples_each_update"]:
            return
        already_queing = self.shared_vars["trainer_feed"].qsize()
        print("Wrangler: already_queing: {}".format(already_queing))
        for _ in range(already_queing, self.target_trainer_queue_length):
            print("Wrangler ADDS ONE SAMPLE to bus!")
            new_sample, new_filter = self.wrangler.experience_replay.get_random_sample(
                                                                                       self.settings["n_samples_each_update"],
                                                                                       alpha=self.settings["prioritized_replay_alpha"].get_value(self.wrangler.global_clock),
                                                                                       beta=self.settings["prioritized_replay_beta"].get_value(self.wrangler.global_clock),
                                                                                      )
            #HERE IS GOOD PLACE TO PUT CODE THAT PREPARES DATA FOR TRAINER...
            self.shared_vars["trainer_feed"].put(new_sample)
            if WRANGLER_DEBUG: print("WRANGLER PASSED SAMPLE TO DATA")
            self.sample_buffer.append((new_sample,new_filter)) #To the rights side
        if WRANGLER_DEBUG: print("WRANGLER exiting: feed_trainer")

    def replace_trainer_samples(self):
        if WRANGLER_DEBUG: print("WRANGLER entering: replace_trainer_samples")
        t = time.time()
        if not self.shared_vars["trainer_return"].empty():
            try:
                self.shared_vars["trainer_return"].get(timeout=0.1)
                if len(self.sample_buffer) > 0:
                    ret_sample, filter = self.sample_buffer.popleft() #From the left side
                    for i in filter:
                        ret_sample[i].update_value(self.wrangler.run_default_model)
            except queue.Empty:
                pass
        t = time.time() - t
        self.stats["t_updating_total"] += t
        self.stats["t_updating"] = t
        if WRANGLER_DEBUG: print("WRANGLER exiting: replace_trainer_samples")

    def print_stats(self):
        frac_load     = self.stats["t_loading_total" ] / (self.stats["t_loading_total"] + self.stats["t_updating_total"])
        frac_update   = self.stats["t_updating_total"] / (self.stats["t_loading_total"] + self.stats["t_updating_total"])
        current_speed = self.stats["n_samples_total" ] / (time.time() - self.stats["t_start"]                             )
        print("-------WRANGLER info-------")
        print("updated for {}s".format(self.stats["t_updating"]))
        print("loaded  for {}s".format(self.stats["t_loading" ]))
        print("fraction in training/loading: {} / {}".format(frac_update,frac_load))
        print("current speed: {} samples/sec".format(current_speed))
        print("experience_replay_size: {}".format(len(self.wrangler.experience_replay)))

    def workers_running(self):
        for i in self.shared_vars["run_flag"]:
            if i == 1: return True
        return False

    def __str__(self):
        return "thread( type={}, ID={})".format(type(self), self.id)
