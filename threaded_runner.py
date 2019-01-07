import multiprocessing
import time
import numpy as np
import aux.misc

class trainer_thread:
    def __init__(self,trainer, runner_threads, train_epochs, patience=0.1):
        # multiprocessing.Process.__init__(self, target=self, args=())
        self.trainer = trainer
        self.runner_threads = runner_threads
        self.train_epochs = train_epochs
        self.running = False
        self.patience = patience
    def __call__(self, *x, **kx):
        self.run()
    def run(self):
        print("trainer bypassed...")
        return
        self.running = True
        # current_idx       = [0 for _ in runners]
        # current_iteration = [1 for _ in runners] #These are for the non-primitive solution...
        all_data = []
        # while ( np.array(current_iteration) < self.train_epochs ).any():
        for r in self.runner_threads:
            '''PRIMITIVE SOLUTION!'''
            while r.running:
                pass
            '''
            DESIRABLE: start processing the first epoch on the samples
            generated by the runner, as they are coming in
            '''
        #     all_data += r.agent.get_train_data()
        # self.trainer.do_training(all_data, self.train_epochs)
        self.running = False

    def join(self):
        while self.running:
            time.sleep(self.patience)

class runner_thread:
    def __init__(self, id, env, n_steps, agent, patience=0.1):
        # multiprocessing.Process.__init__(self, target=self, args=())
        self.id = id
        self.env = env
        self.n_steps = n_steps
        self.agent = agent
        self.running = False
        self.patience = patience
        self.current_player = 1
    def __call__(self):
        self.run()
    def join(self):
        while self.running:
            time.sleep(self.patience)
    def run(self):
        self.running = True
        s = self.env.get_state()
        for t in range(0,self.n_steps):
            print("worker{}:{}".format(self.id,t))
            self.current_player = 1 - self.current_player
            _,a = self.agent.get_action(s, player=self.current_player)
            ds = self.env.perform_action(a)
            s = self.env.get_state()
            for i,d in enumerate(ds):
                if d: self.env.reset(env=[i])
        print("worker{} done".format(self.id))
        self.running = False

class threaded_runner:
    def __init__(self, envs=None, n_steps=0, runners=None, trainer=None, train_epochs=3, patience=[0.1,0.1, 0.1]):
        self.threads = []
        runner_threads = []
        if type(patience) is list: runner_patience, trainer_patience, self.patience = patience
        else: runner_patience = trainer_patience = self.patience = patience
        for i,ae in enumerate(zip(runners,envs)):
            a,e = ae
            thread = runner_thread(i, e, n_steps, a, patience=runner_patience)
            self.threads.append(thread)
            runner_threads.append(thread)
        self.threads.append(trainer_thread(trainer,runner_threads, train_epochs, patience=trainer_patience))

    def run(self):
        p = multiprocessing.Pool(len(self.threads))

        print("pool created")
        p.map(lambda x:x(),self.threads)
        # for thread in self.threads:
        #     thread.start()

    def join_all_threads(self):
        for thread in self.threads:
            while thread.running:
                time.sleep(self.patience)
            thread.join()
        print("???")

    def join(self):
        self.join_all_threads()
