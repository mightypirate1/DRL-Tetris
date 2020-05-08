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
        # _r = self.r if reward_shaper is None else reward_shaper(self.r)
        # r = [x() for x in _r]
        # r = np.array(     r).reshape((-1,1))
        # d = np.array(self.d).reshape((-1,1))
        # #Add a dummy state to the end of the trajectory. This doesnt matter since it's effect will be multiplied by 0 due to done
        # _s = self.s + [self.s[-1]]
        # _p = self.p + [self.p[-1]]
        # prios = model((_s, r, d), player=_p)
        # s  = state_fcn(self.s, player=self.p)
        # data = (s,0,r,d)
        # return data, prios
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

        # #Prelinary prios! (They may differ from the "real" prios, since they are computed as a 1-step td-error without a reference net)
        _Q, V, pieces = model((_s), player=_p)
        Qs = np.array([q[r,t,p] for (r,t,p),q in zip(self.a,_Q[:-1])]).reshape(-1,1)
        Vsp = V[1:]
        td_error = r + gamma_discount * Vsp - Qs
        prios = np.abs(td_error)
        s  = state_fcn(self.s, player=self.p)
        data = (s,a,r,d) # s[t,0,:] <- s_t, s[t,1,:] <- s_(t+1) etc
        return data, prios
