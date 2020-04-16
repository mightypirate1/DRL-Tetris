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
        self.initial_weights = True
        self.last_global_clock = 0
        self.print_frequency = 10 * self.settings["n_workers"]
        self.last_print_out = 10 * (self.settings["n_workers"] - self.id - 1 )
        if self.id > 0:
            self.settings["render"] = False #At most one worker renders stuff...
        self.running = False


    def run(self, *args):
        try:
            self.shared_vars["run_flag"][self.id] = self.running = 1
            if not self.settings["run_standalone"]:
                self.await_trainer()
            self.thread_code(*args)
        except Exception as e:
            print("WORKER{} PANIC: aborting!\n-----------------")
            raise e
        else:
            print("worker{} done".format(self.id))
        finally:
            runtime = time.time() - self.t_thread_start
            self.shared_vars["run_flag"][self.id] = 0
            self.shared_vars["run_time"][self.id] = runtime

    def join(self):
        while self.running:
            time.sleep(10)

    def thread_code(self, *args):
        '''
        Main code [WORKER]:
        '''
        if not self.settings["run_standalone"]:
            myid=mp.current_process()._identity[0]
            np.random.seed(myid^struct.unpack("<L",os.urandom(4))[0])

        # Be Nice
        niceness=os.nice(0)
        os.nice(5-niceness)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.settings["trainer_gpu_fraction"])
        with tf.Session(config=tf.ConfigProto(log_device_placement=False,device_count={'GPU': self.gpu_count})) as session:
            #Initialize!
            self.env = self.settings["env_vector_type"](
                                                        self.settings["n_envs_per_thread"],
                                                        self.settings["env_type"],
                                                        settings=self.settings
                                                       )
            self.agent = self.settings["agent_type"](
                                                     self.settings["n_envs_per_thread"],
                                                     n_workers=self.settings["n_workers"],
                                                     id=self.id,
                                                     mode=threads.WORKER if not self.settings["run_standalone"] else threads.STANDALONE,
                                                     sandbox=self.settings["env_type"](settings=self.settings),
                                                     session=session,
                                                     settings=self.settings,
                                                    )

            #
            ##Run!
            #####
            self.quick_summary = quick_summary(settings=self.settings, session=session)
            self.t_thread_start = time.time()
            s_prime = self.env.get_state()

            current_player = np.random.choice([i for i in range(self.settings["n_players"])], size=(self.settings["n_envs_per_thread"])  )
            if not self.settings["single_policy"]:
                c = current_player[0]
                for idx in range(current_player.size):
                    current_player[idx] = c

            if not self.settings["single_policy"]:
                stash = {"experience" : None}

            for t in range(0,self.n_steps):
                #Say hi!
                if t % 100 == 0: print("worker{}:{}/{}".format(self.id,t,self.n_steps))

                #Take turns...
                current_player = 1 - current_player
                state = self.env.get_state()

                #Get action from agent
                action_idx, action = self.agent.get_action(state, player=current_player, training=True, random_action=self.initial_weights)
                # action = self.env.get_random_action(player=current_player)

                #Perform action
                reward, done = self.env.perform_action(action, player=current_player)
                s_prime = self.env.get_state()

                #Render?
                if self.settings["render"]:
                    self.env.render()

                #Record what just happened!
                experience = self.make_experience(state, action_idx, reward, s_prime, current_player ,done)
                if self.settings["single_policy"]:
                    #Store to memory
                    self.agent.store_experience(experience)
                else:
                    #Complete the previous player's experience-tuple and store that
                    if stash["experience"] is not None:
                        e    = stash["experience"] #In two-policy mode, each agent completes the other's experience by adding their action/result to it, then store
                        e[2] = [self.merge_rewards(r1,r2) for r1,r2 in zip(e[2], reward)]
                        e[3] = s_prime
                        e[5] = done
                        self.agent.store_experience(e)

                #Reset the envs that reach terminal states
                reset_list = [ idx for idx,d in [(_idx,_d) for _idx,_d in enumerate(done)] if d]
                ref = [(_idx,_d) for _idx,_d in enumerate(done)]
                self.env.reset(env=reset_list)
                for e_idx in reset_list: #TODO: come up with a way of writing this without the loop. (prettier)
                    if self.settings["single_policy"]:
                        self.agent.store_experience( self.make_experience([s_prime[e_idx]], [None], [None], [None], [1-current_player[e_idx]],[True]), env=e_idx)
                    else:
                        self.agent.store_experience( self.make_experience(state, action_idx, reward, s_prime, current_player ,done, env=e_idx), env=e_idx)
                self.agent.ready_for_new_round(training=True, env=reset_list)

                #Store this
                if not self.settings["single_policy"]:
                    #If there were resets, the stashed experience should get the new state
                    experience = self.experience_reset_update(experience, reset_list, self.env.get_state())
                    stash["experience"] = experience

                #Periodically send data to the trainer thread!
                if (t+1) % self.settings["worker_data_send_fequency"] == 0:
                    self.send_training_data()

                #Look for new weights from the trainer!
                self.check_thread_status()
                self.update_weights_and_clock()

                #Print
                self.print_stats()

            #Report when done!
            self.report_wasted_data()

    def await_trainer(self):
        while self.shared_vars["run_flag"][-1] == 0:
            print("worker{} waiting for a trainer to come online.".format(self.id), self.shared_vars["run_flag"][-1])
            time.sleep(2)
    def check_thread_status(self):
        if self.shared_vars["run_flag"][-1] == 0:
            raise Exception("worker{} thinks it's trainer is dead. ABORTING.".format(self.id))

    def update_weights_and_clock(self):
        if self.settings["run_standalone"]:
            if self.agent.trainer.n_train_steps["total"] % 100 == 0 and self.agent.trainer.n_train_steps["total"] > self.current_weights:
                print("Saving weights...")
                self.agent.trainer.save_weights(*utils.weight_location(self.settings,idx=self.agent.trainer.n_train_steps["total"]))
                self.current_weights = self.agent.trainer.n_train_steps["total"]
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
            self.initial_weights = False

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
        print("action temperature: {}".format(self.agent.theta))
        print("action entropy: {}".format(self.agent.action_entropy))
        print("Epsilon: {}".format(self.settings["epsilon"].get_value(self.agent.clock) * self.agent.avg_trajectory_length**(-1)))
        self.last_print_out = t
        if self.settings["run_standalone"]:
            s = {
                #"Avg. combo-reward"          : self.agent.env.tot_combo_reward / self.agent.rounds_played,
                 "Average trajectory length" : self.agent.avg_trajectory_length,
                 "Epsilon (adative)"         : self.settings["epsilon"].get_value(self.agent.clock) * self.agent.avg_trajectory_length**(-1),
                 "Action entropy"            : self.agent.action_entropy,
                 "Action temperature"        : self.agent.theta,
                }
            self.quick_summary.update(s, time=self.agent.clock)

    def walltime(self):
        return time.time() - self.t_thread_start

    def report_wasted_data(self):
        waste = sum([len(t) for t in self.agent.current_trajectory])
        print( "worker{} discards".format(self.id), waste, "samples")

    def time():
        if self.settings["run_standalone"]:
            return self.agent.clock
        else:
            return self.shared_vars["global_clock"].value

    def experience_reset_update(self, experience, reset_list, newstate):
        state, action_idx, reward, s_prime, current_player, done = experience
        _state          = [ ns if i   in reset_list else s for s,ns,i in zip(state,newstate,range(len(state))) ]
        _action_idxs    = [ 0  if idx in reset_list else a for idx, a in enumerate(action_idx)                 ]
        _reward         = [ 0  if idx in reset_list else r for idx, r in enumerate(reward)                     ]
        _s_prime        = [ None                           for _      in state                                 ]
        _current_player = [ c                              for c      in current_player                        ]
        _done           = [ False                          for _      in state                                 ]
        return self.make_experience(_state, _action_idxs, _reward, _s_prime, _current_player, _done)

    def make_experience(self, state, action_idx, reward, s_prime, current_player ,done, env=None):
        if env is None:
            return [state, action_idx, reward, s_prime, current_player, done]
        else:
            return [[state[env]], [action_idx[env]], [reward[env]], [s_prime[env]], [current_player[env]], [done[env]]]

    def merge_rewards(self,r1,r2):
        # flag = False
        # if r1 != 0 or r2 != 0: print("r1/r2: ",r1,r2, end=' -> ', flush=True); flag = True
        assert r1 > -1, "r1==-1 !!!!!!!!!!\n\n\n\n"
        if r1 > 0:
            assert r2 < 0, "if I win, you should lose\n\n\n\n"
            # if flag: print(r1, " (case1)")
            return r1
        # if flag: print(r1-r2, " (case2)")
        return r1-r2

    def __str__(self):
        return "thread( type={}, ID={})".format(type(self), self.id)
