import numpy as np

class trajectory:
    def __init__(self):
        self.length = 0
        self.s, self.r, self.d, self.p = [], [], [], []

    def get_states(self):
        return self.s

    def add(self,e, end_of_trajectory=False):
        state,action,reward,s_prime,player,done = e
        self.s.append(state)
        self.p.append(player)
        if not end_of_trajectory:
            self.r.append(reward)
            self.d.append(done)
            self.length += 1

    def get_cumulative_reward(self, gamma_discount=0.999):
        return sum([x*gamma_discount**i for i,x in enumerate(self.r)])

    def process_trajectory(self, model, state_fcn, reward_shaper=None, gamma_discount=0.99):
        r = self.r if reward_shaper is None else reward_shaper(self.r)
        v = model(self.s, player=self.p)
        r = np.array(     r).reshape((-1,1))
        d = np.array(self.d).reshape((-1,1))
        td_errors = -v[:-1] + r -gamma_discount * v[1:] * (1-d)
        prios = np.abs(td_errors)
        s  = state_fcn(self.s[:-1], player=self.p[:-1])
        sp = state_fcn(self.s[1:], player=self.p[1:])
        data = (s, sp,None,r,d)
        return data, prios

    def get_winner(self): #This function assumes that the reward is done correctly, and corresponds to winning only
        if self.r[-1] == 1:
            return self.p[-1]
        if self.r[-1] == -1:
            return 1-self.p[-1]
        print("IT HAPPENED ", self.r[-1])

    def __len__(self):
        return self.length
