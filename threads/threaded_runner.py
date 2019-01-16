import tensorflow as tf
import multiprocessing as mp
import time
import numpy as np
import aux.utils as utils
import threads
from threads.worker_thread import worker_thread
from threads.trainer_thread import trainer_thread
from threads.wrangler_thread import wrangler_thread

class threaded_runner:
    def __init__(self, settings=None):
        #Parse settings
        self.settings = utils.parse_settings(settings)
        patience = self.settings["process_patience"]
        if type(patience) is list: worker_patience, trainer_patience, self.patience = patience
        else: worker_patience = trainer_patience = self.patience = patience

        #Set up some shared variables to use for inter-thread communications (data transfer etc)
        manager = mp.Manager()
        self.shared_vars = {
                            #run_flag is up when a worker is running. run_time is the exectution-time of a worker.
                             "run_flag"            : mp.Array("i", [  0 for _ in range(settings["n_workers"])] ),
                             "run_time"            : mp.Array("d", [0.0 for _ in range(settings["n_workers"])] ),
                            #Time
                             "global_clock"        : mp.Value("i", 0),
                            #Weights
                             "update_weights"      : manager.dict(zip(["idx", "weights"], [0,None] ) ), #This means that the last issued weights is "None" with batch_no "0"
                             "update_weights_lock" : mp.Lock(),
                            #data_flag signals that a worker put something on it's data_bus
                             "data_queue"          : mp.Queue(),
                             "trainer_feed"        : mp.Queue(),
                             "trainer_return"      : mp.Queue(),
                            #some stats...
                             "trainer_stats"       : manager.dict(),
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
                                   patience=worker_patience,
                                  )
            thread.deamon = True
            self.threads["workers"].append(thread)

        if not self.settings["run_standalone"]:
            #Add 1 trainer
            trainer = trainer_thread(
                                     id=threads.TRAINER_ID,
                                     settings=settings,
                                     shared_vars=self.shared_vars,
                                     patience=trainer_patience,
                                    )
            self.threads["trainer"] = trainer
            #Add 1 wrangler
            wrangler = wrangler_thread(
                                       id=threads.WRANGLER_ID,
                                       settings=settings,
                                       shared_vars=self.shared_vars,
                                       patience=trainer_patience,
                                      )
            self.threads["wrangler"] = wrangler

    def get_avg_runtime(self):
        ret = 0
        for queue in self.shared_vars["run_time"]:
            ret += queue.get()
        return ret / len(self.shared_vars["run_time"])

    def run(self, steps):
        if len(self.threads["workers"]) == 0:
            print("You have no workers employed. What do you want to run, even???");return
        for thread in self.threads["workers"]:
            self.start_thread(thread)
        while self.shared_vars["run_flag"][0] == 0:
            time.sleep(1);
        if not self.settings["run_standalone"]:
            self.start_thread(self.threads["trainer"])
            self.start_thread(self.threads["wrangler"])

    def start_thread(self, thread):
        print("Starting thread: {}".format(thread))
        thread.start()

    def join_all_threads(self):

        ##TODO: This code segment was meant to exit if the trainer crashes... but it doesnt work :(
        # flag = True
        # while flag:
        #     time.sleep(self.patience)
        #     if not self.threads["trainer"].is_alive():
        #         if not self.threads["trainer"].running:
        #             print("TRAINER IS DEAD!")
        #             for thread in self.threads["workers"]:
        #                 thread.terminate()
        #         flag = False

        print("Tring to join...")
        threads = self.threads["workers"] if self.settings["run_standalone"] else [self.threads["trainer"], self.threads["wrangler"], *self.threads["workers"]]
        for thread in threads:
            thread.join()
        print("join done!")

    def join(self):
        self.join_all_threads()
