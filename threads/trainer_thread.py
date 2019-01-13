import tensorflow as tf
import multiprocessing as mp
import time
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
        self.running = False
        self.shared_vars = shared_vars
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

    def run(self, *args):
        # self.var_test()
        '''
        Main code [TRAINER]:
        '''
        with tf.Session(config=tf.ConfigProto(log_device_placement=True,device_count={'GPU': self.gpu_count})) as session:
            #Initialize!
            self.trainer = self.settings["trainer_type"](
                                                         id=self.id,
                                                         mode=threads.ACTIVE,
                                                         sandbox=self.settings["env_type"](settings=self.settings),
                                                         session=session,
                                                         shared_vars=self.shared_vars,
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
                self.do_training()
                self.load_data()
                # self.print_stats()
                if self.trainer.n_train_steps % self.settings["weight_transfer_frequency"] == 0:
                    self.transfer_weights()
                    self.set_global_clock()

            #Report when done!
            print("trainer done")
            self.stats["t_stop"] = time.time()
            self.stats["t_total"] = self.stats["t_stop"] - self.stats["t_start"]
            runtime = self.stats["t_total"]

    def do_training(self):
        t = time.time()
        self.trainer.do_training()
        t = time.time() - t
        self.stats["t_training"] = t
        self.stats["t_training_total"] += t

    def load_data(self):
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
        self.trainer.receive_data(data)
        t = time.time() - t
        self.stats["t_loading"] = t
        self.stats["t_loading_total"] += t

    def print_stats(self):
        frac_load  = self.stats["t_loading_total"] / (self.stats["t_loading_total"] + self.stats["t_training_total"])
        frac_train = self.stats["t_training_total"] / (self.stats["t_loading_total"] + self.stats["t_training_total"])
        current_speed = self.stats["n_samples_total"] / (time.time() - self.stats["t_start"])
        print("-------trainer info-------")
        print("trained for {}s".format(self.stats["t_training"]))
        print("loaded  for {}s".format(self.stats["t_loading"]))
        print("fraction in training/loading: {} / {}".format(frac_train,frac_load))
        print("current speed: {} samples/sec".format(current_speed))

    def transfer_weights(self):
        #Get some data
        n, w = self.trainer.export_weights()
        #Make it the worlds
        self.shared_vars["update_weights_lock"].acquire()
        self.shared_vars["update_weights"]["idx"] = n
        self.shared_vars["update_weights"]["weights"] = w
        self.shared_vars["update_weights_lock"].release()

    def set_global_clock(self):
        with self.shared_vars["global_clock"].get_lock():
            self.shared_vars["global_clock"].value = self.trainer.global_clock


    def workers_running(self):
        for i in self.shared_vars["run_flag"]:
            if i == 1: return True
        return False

    def __str__(self):
        return "thread( type={}, ID={})".format(type(self), self.id)









#####
#####
#####
#####
#####
#####

    def var_test(self):
        import numpy as np
        def val(x):
            if isinstance(x, mp.queues.Queue):
                return x
            if isinstance(x, mp.sharedctypes.Synchronized):
                return x.value
            if isinstance(x, type(self.shared_vars["run_flag"])):
                return [v for v in x]
            return x
        idx = -1
        while True:
            idx += 1
            time.sleep(3)
            while True:
                idx += 1
                time.sleep(3)
                print("trainer vars (idx:{}):".format(idx))
                for x in self.shared_vars:
                    print("\t{} : {}".format(x,val(self.shared_vars[x])))
                if idx % 4 == 0:
                    self.shared_vars["update_weights_lock"].acquire()
                    self.shared_vars["update_weights"][1] = idx*np.random.random(size=(4,)).round(decimals=2)
                    self.shared_vars["update_weights"][2] = idx*np.random.random(size=(4,)).round(decimals=2)
                    self.shared_vars["update_weights"][3] = idx
                    time.sleep(5)
                    self.shared_vars["update_weights_lock"].release()
