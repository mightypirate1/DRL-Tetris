import tensorflow as tf
import multiprocessing as mp
import time
import pickle
import numpy as np
import aux.utils as utils
import threads
from threads.worker_thread import worker_thread
from threads.trainer_thread import trainer_thread

class threaded_runner:
    def __init__(self, settings=None, restart=(None, 0)):
        #Parse settings
        self.settings = utils.parse_settings(settings)

        #Set up some shared variables to use for inter-thread communications (data transfer etc)
        manager = mp.Manager()
        init_f, init_c = restart #Initializes clock and index to 0,0 or the value specified by the restart-tuple
        n_threads = self.settings["n_workers"] + int(not self.settings["run_standalone"])
        self.shared_vars = {
                            #run_flag is up when a worker is running. run_time is the exectution-time of a worker.
                             "run_flag"            : mp.Array("i", [  0 for _ in range(n_threads)] ),
                             "run_time"            : mp.Array("d", [0.0 for _ in range(settings["n_workers"])] ),
                            #Time
                             "global_clock"        : mp.Value("i", init_c),
                            #Weights
                             "update_weights"      : manager.dict(zip(["idx", "weights", "timestamp"], [0,None, None] ) ), #This means that the last issued weights is "None" with batch_no "0"
                             "update_weights_lock" : mp.Lock(),
                            #data_flag signals that a worker put something on it's data_bus
                             # "data_queue"          : mp.Queue(),
                             "data_queue"          : manager.Queue(),
                           }

        #Init all threads!
        if self.settings["run_standalone"]:
            assert self.settings["n_workers"] == 1, "If you run standalone, just do one worker, please!"
        self.threads = {"workers" : list(), "trainer" : None}
        self.all_threads = list()
        #Add N workers
        for i in range(self.settings["n_workers"]):
            thread = worker_thread(
                                   id=i,
                                   settings=settings,
                                   shared_vars=self.shared_vars,
                                   init_weights=init_f,
                                   init_clock=init_c,
                                  )
            thread.deamon = True
            self.threads["workers"].append(thread)
            self.all_threads.append(thread)

        if not self.settings["run_standalone"]:
            #Add 1 trainer
            trainer = trainer_thread(
                                     id=threads.TRAINER_ID,
                                     settings=settings,
                                     shared_vars=self.shared_vars,
                                     init_weights=init_f,
                                     init_clock=init_c,
                                    )
            self.threads["trainer"] = trainer
            self.all_threads.append(trainer)
    def get_avg_runtime(self):
        ret = 0
        for t in self.shared_vars["run_time"]:
            ret += t
        return ret / len(self.shared_vars["run_time"])

    def run(self):
        if len(self.threads["workers"]) == 0:
            print("You have no workers employed. What do you want to run, even???");return
        for thread in self.all_threads:
            self.start_thread(thread)

    def start_thread(self, thread):
        print("Starting thread: {}".format(thread))
        thread.start()

    def join(self):
        # TODO: Make a watch dog thing here. (Check if workers die, if so: warning. Check if trainer dies, if so: terminate.)
        print("Tring to join...")
        done = False
        while not done:
            done = True
            for flag in self.shared_vars["run_flag"]:
                done = done and flag == 0
            time.sleep(1)
        print("join done!")
