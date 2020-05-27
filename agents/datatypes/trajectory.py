import numpy as np

class trajectory:
    def __init__(self):
        self.length = 0
        self.s, self.p, self.a, self.r, self.d = [], [], [], [], []
        self.winner = None
    def get_states(self):
        return self.s
    def add(self,e):
        if e is None or e[0] is None: #This is code for "This is a fake-experience - don't add it!"
            return
        state,action,reward,s_prime,player,done = e
        self.s.append(state)
        self.p.append(player)
        self.a.append(action)
        self.r.append(reward)
        self.d.append(done)
        self.length += 1
        if done:
            self.winner = 1 - player
    def process_trajectory(self, model, state_fcn, reward_shaper=None, gamma_discount=0.99, k_steps=1):
        assert len(self) > 0, "dont process empty trajectories!"
        #Ready the data!
        _s = self.s + [self.s[-1]]
        _p = self.p + [self.p[-1]]
        a = np.array( self.a).reshape((-1,1))
        r = self.r if reward_shaper is None else reward_shaper(self.r)
        r = np.array( [R() for R in r]).reshape((-1,1))
        d = np.array( self.d          ).reshape((-1,1))
        #Prelinary prios! (They may differ from the "real" prios, since they are computed as a 1-step td-error without a reference net)
        vals = model(_s, player=_p)
        td_error = r+gamma_discount*vals[1:]*(1-d) - vals[:-1]
        prios = np.abs(td_error)
        s  = state_fcn(self.s, player=self.p)
        data = (s,a,r,d) # s[t,0,:] <- s_t, s[t,1,:] <- s_(t+1) etc
        return data, prios
    def get_winner(self): #This function assumes that the reward is done correctly, and corresponds to winning only. Draws are rare enough, so it should be good enough...
        if self.r[-1] == 1:
            return self.p[-1]
        if self.r[-1] == -1:
            return 1-self.p[-1]
    def __len__(self):
        return self.length

class ppo_trajectory(trajectory):
    def __init__(self):
        super().__init__()
    def process_trajectory(self, model, state_fcn, reward_shaper=None, gamma_discount=0.99, k_steps=1):
        assert len(self) > 0, "dont process empty trajectories!"
        a = np.array( self.a          ).reshape((-1,len(self.a[0])))
        r = self.r if reward_shaper is None else reward_shaper(self.r)
        r = np.array( [R() for R in r]).reshape((-1,1))
        d = np.array( self.d          ).reshape((-1,1))
        s = state_fcn(self.s, player=self.p)
        prios = np.zeros_like(r)
        data = (s,a,r,d)
        return data, prios

class q_trajectory(trajectory):
    def __init__(self):
        super().__init__()
    def process_trajectory(self, model, state_fcn, reward_shaper=None, gamma_discount=0.99, k_steps=1):
        assert len(self) > 0, "dont process empty trajectories!"
        #Ready the data!
        _s = self.s + [self.s[-1]]
        _p = self.p + [self.p[-1]]
        a = np.array( self.a          ).reshape((-1,len(self.a[0])))
        r = self.r if reward_shaper is None else reward_shaper(self.r)
        r = np.array( [R() for R in r]).reshape((-1,1))
        d = np.array( self.d          ).reshape((-1,1))
        prios = 4*np.ones_like(r) #Very large prio :)
        s  = state_fcn(self.s, player=self.p)
        data = (s,a,r,d) # s[t,0,:] <- s_t, s[t,1,:] <- s_(t+1) etc
        return data, prios
