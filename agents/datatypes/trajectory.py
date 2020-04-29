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
        r = self.r if reward_shaper is None else reward_shaper(self.r)
        r = np.array( [R() for R in r]).reshape((-1,1))
        d = np.array( self.d          ).reshape((-1,1))
        #Add a dummy state to the end of the trajectory. This doesnt matter since it's effect will be multiplied by 0 due to done
        _s = self.s + [self.s[-1] for _ in range(k_steps)]
        _p = self.p + [self.p[-1] for _ in range(k_steps)]
        _a = self.a + [self.a[-1] for _ in range(k_steps)]
        _r = self.r + [self.r[-1] for _ in range(k_steps)]
        _d = self.d + [self.d[-1] for _ in range(k_steps)]
        assert False, "Fix this part: compute prios"
        prios = model((_s, _r, _d), player=_p)
        s  = state_fcn(self.s, player=self.p)
        data = (s,self.a,r,d) # s[t,0,:] <- s_t, s[t,1,:] <- s_(t+1) etc
        return data, prios

    def get_winner(self): #This function assumes that the reward is done correctly, and corresponds to winning only. Draws are rare enough, so it should be good enough...
        if self.r[-1] == 1:
            return self.p[-1]
        if self.r[-1] == -1:
            return 1-self.p[-1]

    def __len__(self):
        return self.length
