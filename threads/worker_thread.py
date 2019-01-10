import tensorflow as tf
import numpy as np
import multiprocessing
import time
from aux.settings import default_settings
import aux.utils as utils

class worker_thread(multiprocessing.Process):
    def __init__(self, id=id, settings=None, trajectory_queue=None):
        multiprocessing.Process.__init__(self, target=self)
        self.settings = utils.parse_settings(settings)
        self.id = id
        self.n_steps = self.settings["worker_steps"]
        self.trajectory_queue = trajectory_queue
        if self.id > 0:
            self.settings["render"] = False #At most one worker renders stuff...
        self.running = False

    def __call__(self, *args):
        self.run(*args)

    # def join(self):
    #     while self.running:
    #         time.sleep(self.patience)

    def run(self, *args):
        '''
        Main code:
        '''
        with tf.Session() as session:
            #Initialize!
            self.env = self.settings["env_vector_type"](
                                                    self.settings["n_envs_per_thread"],
                                                    self.settings["env_type"],
                                                    settings=self.settings
                                                    )
            self.agent = self.settings["agent_type"](
                                                self.settings["n_envs_per_thread"],
                                                id=self.id,
                                                sandbox=self.settings["env_type"](settings=self.settings),
                                                session=session,
                                                trajectory_queue=self.trajectory_queue,
                                                settings=self.settings,
                                                )

            #
            ##Run!
            #####
            #Thread-logics
            self.running = True
            #Init run
            t_thread_start = time.time()
            s_prime = self.env.get_state()
            current_player = np.random.choice([i for i in range(self.settings["n_players"])], size=(self.settings["n_envs_per_thread"])  )
            for t in range(0,self.n_steps):
                #Say hi!
                print("worker{}:{}".format(self.id,t))

                #Take turns...
                current_player = 1 - current_player
                state = s_prime

                #Get action from agent
                action_idx, action    = self.agent.get_action(state, player=current_player)
                # action = self.env.get_random_action(player=current_player)

                #Perform action
                reward, done = self.env.perform_action(action, player=current_player)
                s_prime = self.env.get_state()

                #Store to memory
                experience = (state, action_idx, reward, s_prime, current_player,done)
                self.agent.store_experience(experience)

                #Render?
                if self.settings["render"]:
                    self.env.render()

                #Reset the envs that reach terminal states
                for i,d in enumerate(done):
                    if d:
                        self.env.reset(env=[i])
                        # self.agent.ready_for_new_round(...)
                        current_player[i] = np.random.choice([0,1])

            #Report when done!
            print("worker{} done".format(self.id))
            runtime = time.time() - t_thread_start
            self.trajectory_queue.put(runtime)
