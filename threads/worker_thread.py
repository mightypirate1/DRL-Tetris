import tensorflow as tf
import numpy as np
import multiprocessing
import time
from aux.settings import default_settings

class worker_thread(multiprocessing.Process):
    def __init__(self, id, settings=None, return_queue=None):
        multiprocessing.Process.__init__(self, target=self, args=(return_queue,))
        self.id = id
        self.running = False
        self.settings = default_settings.copy()
        for x in settings:
            self.settings[x] = settings[x]
        if self.id > 0:
            self.settings["render"] = False #At most one worker renders stuff...
        self.return_queue = return_queue

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
                                                settings=self.settings,
                                                )
            self.n_steps = self.settings["worker_steps"]

            #
            ##Run!
            #####
            #Thread-logics
            T_thread_start = time.time()
            self.running = True
            #Init run
            s_prime = self.env.get_state()
            current_player = np.random.choice([i for i in range(self.settings["n_players"])], size=(self.settings["n_envs_per_thread"])  )
            for t in range(0,self.n_steps):
                #Say hi!
                print("worker{}:{}".format(self.id,t))

                #Take turns...
                current_player = 1 - current_player
                state = s_prime

                #Get action from agent
                _, action    = self.agent.get_action(state, player=current_player)
                #Perform action
                reward, done = self.env.perform_action(action, player=current_player)
                s_prime = self.env.get_state()

                #Store to memory
                experience = (state, reward, s_prime, current_player,done)
                self.agent.store_experience(experience)

                #Render?
                if self.settings["render"]:
                    self.env.render()

                #Reset the envs that reach terminal states
                for i,d in enumerate(done):
                    if d:
                        self.env.reset(env=[i])
                        current_player[i] = np.random.choice([0,1])

            #Report when done!
            print("worker{} done".format(self.id))
            T_thread_stop = time.time()
            self.return_queue.put(T_thread_stop-T_thread_start)
            self.running = False
