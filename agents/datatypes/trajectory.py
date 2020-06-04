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

class sventon_trajectory(trajectory):
    def __init__(self, *args, **kwargs):
        trajectory.__init__(self,*args, **kwargs)
    def process_trajectory(self, model, state_fcn, reward_shaper=None, compute_advantages=False, gamma_discount=0.99, gae_lambda=0.96, k_steps=1):
        assert len(self) > 0, "dont process empty trajectories!"
        a_env, a_int_raw = list(zip(*self.a))
        r = self.r if reward_shaper is None else reward_shaper(self.r)
        r = np.array( [R() for R in r]).reshape((-1,1))
        d = np.array( self.d, dtype=np.uint8).reshape((-1,1))
        s = state_fcn(self.s, player=self.p)
        #process!
        a_env_np = np.array(a_env) #r,t,p
        a_int_np = np.zeros((len(self),3)) #p, a, target_v
        a_int_data = np.array(a_int_raw)
        probabilities = a_int_data[:,0]
        v_piece = a_int_data[:,1]
        v_mean  = a_int_data[:,2]
        a_int_np[:,0] = probabilities
        if compute_advantages:
            a_int_np[:,1:] = self.adv_and_targets(v_piece, v_mean, r[:,0], d[:,0], gamma=gamma_discount, gae_lambda=gae_lambda)
        prios = 2*np.ones_like(r) #Very large prio :)
        a = (a_env_np, a_int_np)
        data = (s,a,r,d)
        return data, prios
    def adv_and_targets(self, v_mean, v_piece, r, d, gamma=0.98, gae_lambda=0.96):
        advantages = np.zeros(v_mean.shape)
        v_next = np.zeros(v_mean.shape)
        v_next[:-1] = v_mean[1:]
        td1s = r + gamma * v_next * (1-d) - v_mean
        A, W = 0.0, 0.0
        for i,td in reversed(list(enumerate(td1s))):
            A *= (gamma * gae_lambda)
            W *= gae_lambda
            A += td
            W += 1
            advantages[i] = (A + v_mean[i] - v_piece[i]) / W #Adjusts the advantage so that the specific piece affects the advantage at current time step, while all other time steps sees it as the average piece value
        targets = v_piece + advantages
        return np.concatenate([advantages[:,np.newaxis], targets[:,np.newaxis]], axis=1)
    # def adv_and_targets2(self, v_mean, v_piece, r, d, gamma=0.98, gae_lambda=0.96):
    #     advantages = np.zeros((len(self),1))
    #     weight = 0.0
    #     a_t = 0.0
    #     v_tplus1 = 0.0
    #     gl = gae_lambda * gamma
    #     for i, v_t, vp_t, r_t in reversed(list(zip(range(len(self)),v_mean, v_piece,r))):
    #         weight += 1.0
    #         a_t += r_t + gamma * v_tplus1 - vp_t
    #         advantages[i,0] = a_t / weight
    #         a_t += vp_t - v_t
    #         v_tplus1 = v_t
    #         weight *= gae_lambda
    #         a_t *= gl
    #     targets = (v_piece[:,np.newaxis] + advantages)
    #     return np.concatenate([advantages, targets], axis=1)

#
# class q_trajectory(trajectory):
#     def __init__(self):
#         super().__init__()
#     def process_trajectory(self, model, state_fcn, reward_shaper=None, gamma_discount=0.99, k_steps=1):
#         assert len(self) > 0, "dont process empty trajectories!"
#         #Ready the data!
#         _s = self.s + [self.s[-1]]
#         _p = self.p + [self.p[-1]]
#         a = np.array( self.a          ).reshape((-1,len(self.a[0])))
#         r = self.r if reward_shaper is None else reward_shaper(self.r)
#         r = np.array( [R() for R in r]).reshape((-1,1))
#         d = np.array( self.d          ).reshape((-1,1))
#         prios = 4*np.ones_like(r) #Very large prio :)
#         s  = state_fcn(self.s, player=self.p)
#         data = (s,a,r,d) # s[t,0,:] <- s_t, s[t,1,:] <- s_(t+1) etc
#         return data, prios
ppo_trajectory = q_trajectory = sventon_trajectory
