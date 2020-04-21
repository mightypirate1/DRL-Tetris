import numpy as np

class trajectory:
    def __init__(self):
        self.length = 0
        self.s, self.r, self.d, self.p = [], [], [], []
        self.winner = None
    def get_states(self):
        return self.s

    def add(self,e):
        if e is None or e[0] is None: #This is code for "This is a fake-experience - don't add it!"
            return
        state,action,reward,s_prime,player,done = e
        self.s.append(state)
        self.p.append(player)
        self.r.append(reward)
        self.d.append(done)
        self.length += 1
        if done:
            self.winner = 1 - player

    def get_cumulative_reward(self, gamma_discount=0.999):
        return sum([x*gamma_discount**i for i,x in enumerate(self.r)])

    def process_trajectory(self, model, state_fcn, reward_shaper=None, gamma_discount=0.99):
        _r = self.r if reward_shaper is None else reward_shaper(self.r)
        r = [x() for x in _r]
        r = np.array(     r).reshape((-1,1))
        d = np.array(self.d).reshape((-1,1))
        #Add a dummy state to the end of the trajectory. This doesnt matter since it's effect will be multiplied by 0 due to done
        _s = self.s + [self.s[-1]]
        _p = self.p + [self.p[-1]]
        prios = model((_s, r, d), player=_p)
        s  = state_fcn(self.s, player=self.p)
        sp = state_fcn(_s[1:], player=_p[1:])
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
