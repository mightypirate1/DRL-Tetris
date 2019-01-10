import tensorflow as tf
import multiprocessing
import time
import numpy as np
import aux.utils as utils
from threads.worker_thread import worker_thread
from threads.trainer_thread import trainer_thread

class threaded_runner:
    def __init__(self, settings=None):
        self.threads = []
        # runner_threads = []
        self.settings = utils.parse_settings(settings)
        patience = self.settings["process_patience"]
        if type(patience) is list: runner_patience, trainer_patience, self.patience = patience
        else: runner_patience = trainer_patience = self.patience = patience
        self.trajectory_queue = [multiprocessing.Queue() for _ in range(settings["n_workers"])]
        for i in range(settings["n_workers"]):
            thread = worker_thread(
                                   id=i,
                                   settings=settings,
                                   trajectory_queue=self.trajectory_queue[i],
                                   )
            self.threads.append(thread)
        # self.trainer = trainer_thread(
        #                                 id="trainer",
        #                                 settings=settings,
        #                                 session=session,
        #                              )
        # self.threads.append(trainer_thread)

    def sum_return_que(self):
        ret = 0
        while not self.trajectory_queue.empty():
            ret += self.trajectory_queue.get()
        return ret

    def run(self, steps):
        for thread in self.threads:
            thread.start()

    def join_all_threads(self):
        print("Tring to join...")
        for thread in self.threads:
            while thread.running:
                time.sleep(self.patience)
            thread.join()
        print("join done!")

    def join(self):
        self.join_all_threads()
