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

class worker_thread(mp.Process):
    def __init__(self, id=id, settings=None, shared_vars=None, patience=0.1):
        mp.Process.__init__(self, target=self)
        self.settings = utils.parse_settings(settings)
        self.id = id
        self.patience = patience
        self.n_steps = self.settings["worker_steps"]
        self.shared_vars = shared_vars
        self.gpu_count = 0 if self.settings["worker_net_on_cpu"] else 1
        self.current_weights = 0 #I have really old weights
        self.last_global_clock = 0
        self.print_frequency = 10 * self.settings["n_workers"]
        self.last_print_out = 10 * (self.settings["n_workers"] - self.id - 1 )
        if self.id > 0:
            self.settings["render"] = False #At most one worker renders stuff...
        self.running = False


    def __call__(self, *args):
        self.running = True #These flag-changes are not seen from the outside of this process I think...
        self.thread_code(*args)
        self.running = False

    def join(self):
        while self.running:
            time.sleep(10)

    def thread_code(self, *args):
        '''
        Main code [WORKER]:
        '''
        myid=mp.current_process()._identity[0]
        np.random.seed(myid^struct.unpack("<L",os.urandom(4))[0])

        # Be Nice
        niceness=os.nice(0)
        os.nice(5-niceness)
        self.shared_vars["run_flag"][self.id] = 1
        with tf.Session(config=tf.ConfigProto(log_device_placement=False,device_count={'GPU': self.gpu_count})) as session:
            #Initialize!
            self.env = self.settings["env_vector_type"](
                                                        self.settings["n_envs_per_thread"],
                                                        self.settings["env_type"],
                                                        settings=self.settings
                                                       )
            self.agent = self.settings["agent_type"](
                                                     self.settings["n_envs_per_thread"],
                                                     id=self.id,
                                                     mode=threads.WORKER if not self.settings["run_standalone"] else threads.STANDALONE,
                                                     sandbox=self.settings["env_type"](settings=self.settings),
                                                     session=session,
                                                     settings=self.settings,
                                                    )

            #
            ##Run!
            #####
            self.t_thread_start = time.time()
            s_prime = self.env.get_state()
            self.quick_summary = quick_summary(settings=self.settings, session=session)
            current_player = np.random.choice([i for i in range(self.settings["n_players"])], size=(self.settings["n_envs_per_thread"])  )
            for t in range(0,self.n_steps):
                #Say hi!
                if t % 100 == 0: print("worker{}:{}".format(self.id,t))

                #Take turns...
                current_player = 1 - current_player
                state = s_prime

                #Get action from agent
                action_idx, action    = self.agent.get_action(state, player=current_player, training=True)
                # action = self.env.get_random_action(player=current_player)

                #Perform action
                reward, done = self.env.perform_action(action, player=current_player)
                s_prime = self.env.get_state()

                #Render?
                if self.settings["render"]:
                    self.env.render()

                #Store to memory
                experience = (state, action_idx, reward, s_prime, current_player ,done)
                self.agent.store_experience(experience)

                #Reset the envs that reach terminal states
                for i,d in enumerate(done):
                    if d:
                        self.env.reset(env=i)
                        self.agent.store_experience((s_prime[i], None, None, None, 1-current_player[i],d), env=i)
                        self.agent.ready_for_new_round(training=True, env=i)
                        current_player[i] = np.random.choice([0,1])

                #Periodically send data to the trainer thread!
                if (t+1) % self.settings["worker_data_send_fequency"] == 0:
                    self.send_training_data()

                #Look for new weights from the trainer!
                self.update_weights_and_clock()

                #Print
                self.print_stats()

            #Report when done!
            print("worker{} done".format(self.id))
            runtime = time.time() - self.t_thread_start
            self.shared_vars["run_flag"][self.id] = 0
            self.shared_vars["run_time"][self.id] = runtime

    def update_weights_and_clock(self):
        if self.settings["run_standalone"]:
            if self.agent.trainer.n_train_steps % 100 == 0 and self.agent.trainer.n_train_steps > self.current_weights:
                print("Saving weights...")
                self.agent.trainer.save_weights(*utils.weight_location(self.settings,idx=self.agent.trainer.n_train_steps))
                self.current_weights = self.agent.trainer.n_train_steps
            return
        if self.shared_vars["global_clock"].value > self.last_global_clock:
            self.last_global_clock = self.shared_vars["global_clock"].value
            self.agent.update_clock(self.last_global_clock)
        if self.shared_vars["update_weights"]["idx"] > self.current_weights:
            self.shared_vars["update_weights_lock"].acquire()
            w = self.shared_vars["update_weights"]["weights"]
            self.current_weights = self.shared_vars["update_weights"]["idx"]
            self.shared_vars["update_weights_lock"].release()
            self.agent.update_weights(w)

    def send_training_data(self):
        if self.settings["run_standalone"]:
            return
        data = self.agent.transfer_data()
        if len(data) > 0:
            self.shared_vars["data_queue"].put(data)

    def print_stats(self):
        if self.walltime() < self.last_print_out + self.print_frequency:
            return
        t = self.walltime()
        print("-------worker{} info-------".format(self.id))
        print("clock: {}".format(self.agent.clock))
        print("current weights: {}".format(self.current_weights))
        print("average trajectory length: {}".format(self.agent.avg_trajectory_length))
        print("Epsilon: {}".format(self.settings["epsilon"].get_value(self.agent.clock) * self.agent.avg_trajectory_length**(-1)))
        self.last_print_out = t
        if self.settings["run_standalone"]:
            s = {
                 "Average trajectory length" : self.agent.avg_trajectory_length,
                 "Epsilon"                   : self.settings["epsilon"].get_value(self.agent.clock) * self.agent.avg_trajectory_length**(-1),
                 "Current speed"             : self.agent.trainer.experience_replay.total_samples/t,
                }
            self.quick_summary.update(s, time=self.agent.clock)

    def walltime(self):
        return time.time() - self.t_thread_start

    def time():
        if self.settings["run_standalone"]:
            return self.agent.clock
        else:
            return self.shared_vars["global_clock"].value

    def __str__(self):
        return "thread( type={}, ID={})".format(type(self), self.id)
