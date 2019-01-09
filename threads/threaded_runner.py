import tensorflow as tf
import multiprocessing
import time
import numpy as np
from threads.worker_thread import worker_thread
from threads.trainer_thread import trainer_thread

class threaded_runner:
    def __init__(self, settings=None):
        self.threads = []
        self.return_queue = multiprocessing.Queue()
        # runner_threads = []
        patience = settings["process_patience"]
        if type(patience) is list: runner_patience, trainer_patience, self.patience = patience
        else: runner_patience = trainer_patience = self.patience = patience
        for i in range(settings["n_workers"]):
            thread = worker_thread(
                                   i,
                                   settings=settings,
                                   return_queue=self.return_queue,
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
        while not self.return_queue.empty():
            ret += self.return_queue.get()
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
