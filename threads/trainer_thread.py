import tensorflow as tf
import multiprocessing
import time

class trainer_thread(multiprocessing.Process):
    def __init__(self):
        super().__init__(self,target=self, args=(1,2,3,))
    def __call__(self, *args):
        self.run(*args)
    def run(self, *args):
        print("EMPTY TRAINER-THREAD CALLED")
