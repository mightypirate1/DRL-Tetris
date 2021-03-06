import multiprocessing as mp
import tensorflow as tf
import numpy as np
import struct
import time
import os

from tools.tf_hooks import quick_summary
import tools.utils as utils
import threads

class worker_thread(mp.Process):
    def __init__(self, id=id, settings=None, shared_vars=None, init_weights=None, init_clock=0):
        mp.Process.__init__(self, target=self)
        self.settings = utils.parse_settings(settings)
        self.id = id
        self.n_steps = self.settings["worker_steps"]
        self.shared_vars = shared_vars
        self.init_weights = init_weights
        self.init_clock = init_clock
        self.t_thread_start = 0
        self.gpu_count = 1 if (not self.settings["worker_net_on_cpu"]) or self.settings["run_standalone"] else 0
        self.random_actions = init_weights is None #Speed up initial datagathering by making random moves.
        self.current_weights = 0 #I have really old weights
        self.current_weights_timestamp = None
        self.last_global_clock = 0
        self.stashed_experience = None
        self.old_reset_list = []
        self.print_frequency = 5 * self.settings["n_workers"]
        self.last_print_out = 5 * (self.settings["n_workers"] - self.id - 1 )
        if self.id > 0:
            self.settings["render"] = False #At most one worker renders stuff...
        self.running = False

    def thread_code(self, *args):
        '''
        Main code [WORKER]:
        '''
        if not self.settings["run_standalone"]: #run_standalone is a debug-mode where only one process exists. By default run_standalone is False.
            myid=mp.current_process()._identity[0]
            np.random.seed(myid^struct.unpack("<L",os.urandom(4))[0])
        # Be Nice
        niceness=os.nice(0) #os.nice increments niceness with the arg and returns the resulting niceness.
        os.nice(self.settings["worker_niceness"] - niceness)# Increment current niceness with worker_niceness (default 5)

        # I don't have the luxury of having multiple GPUs, so this code might not work as intended. It works as it should for single-GPU settings, where the GPU is reserved for the trainer.
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.settings["trainer_gpu_fraction"])
        config = tf.ConfigProto(log_device_placement=False,device_count={'GPU': self.gpu_count})
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as session:
            #Initialize env and agents!
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
                                                     init_weights=self.init_weights,
                                                     init_clock=self.init_clock,
                                                    )

            #
            ## Initialize training variables, and a quick_summary for TensorBoard stats.
            #####
            if self.settings["worker_summaries"]:
                self.quick_summary = quick_summary(settings=self.settings, session=session, suffix="-worker") if self.id == 0 else None
            self.t_thread_start = time.time()
            s_prime = self.env.get_state()
            current_player = np.random.choice([i for i in range(self.settings["n_players"])], size=(self.settings["n_envs_per_thread"])  )
            reset_list = [i for i in range(self.settings["n_envs_per_thread"])]
            if not self.settings["single_policy"]: # not single_policy means we have 2 agents playing each other.
                # The env is an abstraction hiding n different tetris games, and the agent is an abstraction potentially hiding multiple agents. Due to this we have to make it so it's the same player's turn each of the games whenever we have multiple agents.
                c = current_player[0]
                for idx in range(current_player.size):
                    current_player[idx] = c

            #
            ## Main loop!
            ###
            for t in range(0,self.n_steps):

                # Who's turn, and what state do they see?
                current_player = 1 - current_player
                state = self.env.get_state()

                # What action do they want to perform?
                action_idx, action = self.agent.get_action(state, player=current_player, training=True, random_action=self.random_actions)

                #Perform action!
                reward, done = self.env.perform_action(action, player=current_player)
                s_prime = self.env.get_state()

                #Record what just happened!
                experience = self.make_experience(state, action_idx, reward, s_prime, current_player, done)
                self.agent.store_experience(experience)

                #Render?
                self.env.render() #unless turned off in settings

                #If some game has reached termingal state, it's reset here. Agents also get a chance to update their internals...
                reset_list = self.reset_envs(done, reward, current_player)
                self.agent.ready_for_new_round(training=True, env=reset_list)

                # Get data back and forth to the trainer!
                self.send_training_data(t)
                self.check_thread_status()
                self.update_weights_and_clock()

                #Print
                self.print_stats(t)

            #Report when done!
            self.report_wasted_data()

    ###
    #  A few helper functions. They do what they say they do,
    #  but might look weird since they are written to funcion
    #  seamlessly across a range of taining paradigms.
    #
    #  All default behavior is when single_policy is True,
    #  so read that!
    #
    def reset_envs(self, done, reward, current_player):
        #Reset the envs that reach terminal states
        reset_list = [ idx for idx,d in enumerate(done) if d]
        self.env.reset(env=reset_list)
        if self.settings["single_policy"]:
            return reset_list
        #we delay the signal to the agent 1 time-step, since it's experiences are delayed as much.
        old_reset = self.old_reset_list
        self.old_reset_list = reset_list
        return old_reset

    def make_experience(self, state, action_idx, reward, s_prime, current_player, done, env=None):
        if env is None:
            experience = _e = [state, action_idx, reward, s_prime, current_player, done]
            if (not self.settings["single_policy"]):
                experience = self.merge_from_stash(experience, current_player[0])
                self.stashed_experience = _e
            return experience

    def merge_from_stash(self, new, player):
        # merge_experiences and stash is used when training 2 policies against each other.
        # It makes each agent see the other as part of the environment: (s0,s1,s2, ...) -> (s0,s2,s4,..), (s1,s3,s5,...) and similarly for s', r, done etc
        if self.stashed_experience is None:
            return [[None for _ in range(self.settings["n_envs_per_thread"])] for _ in range(6)]
        old_s, old_a, old_r, old_sp, old_p, old_d  = self.stashed_experience
        new_s,new_a,new_r,new_sp,new_p,new_d = new
        #In 2-policy-mode the current player completes the previous players experience-tuple, and reports it for them!
        experience = [
                      old_s,  #s_(t-1)
                      old_a,  #a_(t-1)
                      [old - new for old, new in zip(old_r,new_r)],  #r_(t-1) - r_t ::: Good news for current player shows as bad news for the previous one.
                      new_sp, #s_(t+1)
                      old_p,  #previous player
                      [x or y for x,y in zip(new_d,old_d)] #if either player's move caused the round to end, it should be seen  here!
                     ]
        return experience

    ##########
    ##### Thread communication functions (transfer of trajectories, weights and info)
    ##########
    def update_weights_and_clock(self):
    #Get latest weights and time from the trainer!
        if self.settings["run_standalone"]:#standalone is used mainly for debugging.
            #It is a single-process mode, this is the only thread and it has some extra responsibilities:
            if self.agent.trainer.n_train_steps["total"] % 100 == 0 and self.agent.trainer.n_train_steps["total"] > self.current_weights:
                print("Saving weights...")
                self.agent.trainer.save_weights(*utils.weight_location(self.settings,idx=self.agent.trainer.n_train_steps["total"]))
                self.current_weights = self.agent.trainer.n_train_steps["total"]
            return
        if self.shared_vars["global_clock"].value > self.last_global_clock:
            self.last_global_clock = self.shared_vars["global_clock"].value
            self.agent.update_clock(self.last_global_clock)
        if self.shared_vars["update_weights"]["idx"] > self.current_weights:
            #unlock the weight-box
            self.shared_vars["update_weights_lock"].acquire()
            #get the data
            w = self.shared_vars["update_weights"]["weights"]
            self.current_weights = self.shared_vars["update_weights"]["idx"]
            self.current_weights_timestamp = self.shared_vars["update_weights"]["timestamp"]
            #lock then update
            self.shared_vars["update_weights_lock"].release()
            self.agent.update_weights(w)
            self.random_actions = False
    def send_training_data(self,t):
        if self.settings["run_standalone"]:
            self.random_actions = t > self.settings["n_samples_each_update"]
            return #standalone doesn't send data!
        #Periodically send data to the trainer thread!
        if (t+1) % self.settings["worker_data_send_fequency"] == 0:
            data = self.agent.transfer_data()
            if len(data) > 0:
                self.shared_vars["data_queue"].put(data)
    def await_trainer(self):
        while self.shared_vars["run_flag"][-1] == 0:
            print("worker{} waiting for a trainer to come online.".format(self.id), self.shared_vars["run_flag"][-1])
            time.sleep(2)
    def check_thread_status(self):
        if self.shared_vars["run_flag"][-1] == 0:
            raise Exception("worker{} thinks it's trainer is dead. ABORTING.".format(self.id))

    ##########
    ##### Stats & print-outs
    ##########
    def print_stats(self, loop_time):
        if self.walltime() < self.last_print_out + self.print_frequency:
            return
        self.last_print_out = self.walltime()
        print("-------worker{} info-------".format(self.id))
        print("real time:", time.ctime())
        print("loop-time:", loop_time, "/",self.n_steps)
        print("clock: {}".format(self.agent.clock))
        print("current weights: {} : {}".format(self.current_weights, self.current_weights_timestamp))
        print(utils.progress_bar(loop_time, self.n_steps))
    def report_wasted_data(self):
        # Some people are curious to see how much data is still in the worker when the training is finnished.
        waste = sum([len(t) for t in self.agent.current_trajectory])
        print( "worker{} discards".format(self.id), waste, "samples")
    def time():
        if self.settings["run_standalone"]:
            return self.agent.clock
        else:
            return self.shared_vars["global_clock"].value
    def walltime(self):
        return time.time() - self.t_thread_start

    def __str__(self):
        return "thread( type={}, ID={})".format(type(self), self.id)

    ##########
    ##### Basic thread functionality
    ##########
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
