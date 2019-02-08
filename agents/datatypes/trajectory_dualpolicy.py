import numpy as np

class trajectory_dualpolicy:
    def __init__(self):
        self.length = 0
        self.s, self.sp, self.r, self.d, self.p = [], [], [], [], []

    def get_states(self):
        return self.s

    def add(self,e, end_of_trajectory=False):
        state,action,reward,s_prime,player,done = e
        if state is None:
            #If state is None, it means that the runner signaled that this experience is not "completed", and thus to be ignored. (The next experience coming will have all the data)
            return
        self.s.append(state)
        self.sp.append(s_prime)
        self.p.append(player)
        self.r.append(reward)
        self.d.append(done)
        self.length += 1

    def get_cumulative_reward(self, gamma_discount=0.999):
        return sum([x*gamma_discount**i for i,x in enumerate(self.r)])

    def process_trajectory(self, model, state_fcn, gamma_discount=0.99):
        v = model(self.s+self.sp, player=self.p+self.p)
        r = np.array(self.r).reshape((-1,1))
        d = np.array(self.d).reshape((-1,1))
        td_errors = -v[:self.length] + r + gamma_discount * v[self.length:] * (1-d)
        prios = np.abs(td_errors)
        s  = state_fcn(self.s, player=self.p)
        sp = state_fcn(self.sp, player=self.p)
        data = (s, sp,None,r,d)
        return data, prios

    def get_winner(self): #This function assumes that the reward is done correctly, and corresponds to winning only
        if self.r[-1] == 1:
            return self.p[-1]
        if self.r[-1] == -1:
            return 1-self.p[-1]
        print("IT HAPPENED ", self.r)

    def __len__(self):
        return self.length
