from collections import deque
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

WRANGLER_DEBUG = False

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
        self.current_weights = 0 #I have really old weights
        self.sample_buffer = deque(maxlen=5)
        self.time_budget = 3 if self.settings["wrangler_update_mode"] == "budget" else None
        self.print_frequency = 10
        self.last_print_out = 0
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
        os.nice(1-niceness)
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
            self.quick_summary = quick_summary(settings=self.settings,session=session, init_time=time.time())

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
                data.append(d)
            except ValueError:
                while True:
                    print("really really bad")
        n_samples, avg_length = self.wrangler.receive_data(data)
        self.stats["n_samples_total"] += n_samples
        if avg_length > 0:
            self.quick_summary.update({"Average trajectory length":avg_length}, time=time.time())
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

        if self.settings["wrangler_update_mode"] == "budget":
            if already_queing > 5:
                self.time_budget *= 1.01
            if already_queing < 2:
                self.time_budget *= 0.99

        if WRANGLER_DEBUG: print("Wrangler: already_queing: {} [time_budget: {}]".format(already_queing,self.time_budget))
        for _ in range(already_queing, self.settings["wrangler_trainerfeed_target_length"]):
            if WRANGLER_DEBUG: print("Wrangler ADDS ONE SAMPLE to bus!")
            _sample, _filter = self.wrangler.experience_replay.get_random_sample(
                                                                                       self.settings["n_samples_each_update"],
                                                                                       alpha=self.settings["prioritized_replay_alpha"].get_value(self.wrangler.clock),
                                                                                       beta=self.settings["prioritized_replay_beta"].get_value(self.wrangler.clock),
                                                                                )
            #Either we prepare, or we just send...
            if self.settings["wrangler_unpacks"]:
                sample = self.wrangler.unpack_sample(_sample)
            else:
                sample = _sample
            self.shared_vars["trainer_feed"].put(sample)
            if WRANGLER_DEBUG: print("WRANGLER PASSED SAMPLE TO DATA")
            self.sample_buffer.append((sample,_filter)) #To the rights side
        if WRANGLER_DEBUG: print("WRANGLER exiting: feed_trainer")

    def replace_trainer_samples(self):
        if WRANGLER_DEBUG: print("WRANGLER entering: replace_trainer_samples")
        t = time.time()
        while not self.shared_vars["trainer_return"].empty():
            try:
                self.shared_vars["trainer_return"].get(timeout=0.1)
                if len(self.sample_buffer) > 0:
                    ret_sample, filter = self.sample_buffer.popleft() #From the left side
                    #If the update mode is set in some appropriate way...
                    if not (self.settings["wrangler_update_mode"] == "none" or (self.settings["wrangler_update_mode"] == "budget" and time.time() - t < self.time_budget)):
                        for i in filter:
                            ret_sample[i].update_value(self.wrangler.run_default_model)
            except:
                print("EXCEPTION!")
                pass
        t = time.time() - t
        self.stats["t_updating_total"] += t
        self.stats["t_updating"] = t
        if WRANGLER_DEBUG: print("WRANGLER exiting: replace_trainer_samples")

    def print_stats(self):
        if time.time() < self.last_print_out + self.print_frequency:
            return
        self.last_print_out = time.time()
        frac_load     = self.stats["t_loading_total" ] / (self.stats["t_loading_total"] + self.stats["t_updating_total"])
        frac_update   = self.stats["t_updating_total"] / (self.stats["t_loading_total"] + self.stats["t_updating_total"])
        current_speed = self.stats["n_samples_total" ] / (time.time() - self.stats["t_start"]                             )
        print("-------WRANGLER info-------")
        print("updated for {}s".format(self.stats["t_updating"]))
        print("loaded  for {}s".format(self.stats["t_loading" ]))
        print("fraction in updating/loading: {} / {}".format(frac_update,frac_load))
        print("current speed: {} samples/sec".format(current_speed))
        print("experience_replay_size: {}".format(len(self.wrangler.experience_replay)))
        print("time budget for updates: {}".format(self.time_budget))
        tf_stats = {
                    "Samples per second" : current_speed,
                    "Time spent updating" : frac_update,
                    "Experience replay size" : len(self.wrangler.experience_replay),
                    }
        self.quick_summary.update(tf_stats, time=time.time())
        self.quick_summary.update(dict(self.shared_vars["trainer_stats"]),time=time.time())


    def workers_running(self):
        for i in self.shared_vars["run_flag"]:
            if i == 1: return True
        return False

    def __str__(self):
        return "thread( type={}, ID={})".format(type(self), self.id)
