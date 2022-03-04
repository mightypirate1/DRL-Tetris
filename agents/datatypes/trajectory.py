import numpy as np
import tools.utils as utils

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
    def process_trajectory(self, model, state_fcn, reward_shaper=None, compute_advantages=False, gamma_discount=0.99, gae_lambda=0.96, k_steps=1, augment=False):
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
        v_piece = a_int_data[:,1, np.newaxis]
        v_mean  = a_int_data[:,2, np.newaxis]
        a_int_np[:,0] = probabilities
        if compute_advantages:
            a_int_np[:,1:] = self.adv_and_targets(
                v_piece,
                v_mean,
                r,
                d,
                gamma=gamma_discount,
                gae_lambda=gae_lambda,
                concatenate=True
            )
        a = (a_env_np, a_int_np)
        prios = 2*np.ones_like(r) #Very large prio :)
        data = (s,a,r,d)
        if augment:
            data, prios = self.augment_data(data, prios, state_fcn)
        return data, prios

    def augment_data(self, data, prios, state_fcn):
        piece_swap = (1,0,3,2,4,5,6) # permutation mapping piece-idxs on their refrected counterpart (L -> J, S -> Z, and the rest on themselves)
        s, a, r, d = data
        a_env_np, a_int_np = a
        # Mirrored states
        s2 = state_fcn(self.s, player=self.p, mirrored=True)
        # Mirrored actions
        a_env2_np = a_env_np.copy()
        a_env2_np[:,1] = 9-a_env2_np[:,1]
        for i,p in enumerate(a_env_np[:,2]):
            a_env2_np[i,2] = piece_swap[p]
        a2 = (a_env2_np, a_int_np)
        (vec1a,vec1b), (vis1a,vis1b), p1 = s
        (vec2a,vec2b), (vis2a,vis2b), p2 = s2
        f = lambda x,y : np.concatenate([x,y],axis=0)
        S = [ [f(vec1a,vec2a), f(vec1b,vec2b)], [f(vis1a,vis2a), f(vis1b,vis2b)], f(p1,p2)]
        A = [f(a[0],a2[0]), f(a[1],a2[1])]
        R = f(r,r)
        D = f(d,d)
        data  = (S,A,R,D)
        prios =  np.concatenate([prios, prios], axis=0)
        return data, prios

    def adv_and_targets(self, v_mean, v_piece, r, d, gamma=0.98, gae_lambda=0.96, gve_lambda=0.5, concatenate=False):
        # assumes td1s, v_mean and v_piece
        compute_advantages(lambda_value):
            estimates =  np.zeros_like(td1s)
            A, W = 0.0, 0.0
            for i,td in reversed(list(enumerate(td1s))):
                A *= (gamma * lambda_value)
                W *= lambda_value
                A += td
                W += 1
                estimates[i] = (A + v_mean[i] - v_piece[i]) / W #Adjusts the advantage so that the specific piece affects the advantage at current time step, while all other time steps sees it as the average piece value
            return estimates


        v_next = np.zeros(v_mean.shape)
        v_next[:-1] = v_mean[1:]
        td1s = r + gamma * v_next * (1-d) - v_mean
        advantages       = compute_advantages(td1s, gae_lambda)
        advantages_value = compute_advantages(td1s, gve_lambda)

        targets = v_piece + advantages_value
        if not concatenate:
            return advantages, targets
        return np.concatenate([advantages, targets], axis=1)

class sherlock_trajectory(sventon_trajectory):
    def process_trajectory(self, model, state_fcn, reward_shaper=None, compute_advantages=False, gamma_discount=0.99, gae_lambda=0.96, k_steps=1, augment=False):
        assert len(self) > 0, "dont process empty trajectories!"
        _a  = list(zip(*self.a))
        r = self.r if reward_shaper is None else reward_shaper(self.r)
        r = np.array( [R() for R in r]).reshape((-1,1))
        d = np.array( self.d, dtype=np.uint8).reshape((-1,1))
        s = state_fcn(self.s, player=self.p)
        #join up actions
        f = lambda x : np.concatenate(x, axis=0)
        a = [f(x) for x in _a]
        #process!
        if compute_advantages:
            v_piece, v_mean, = a[3], a[4]
            advantages, targets = self.adv_and_targets(v_piece, v_mean, r, d, gamma=gamma_discount, gae_lambda=gae_lambda)
            a[3], a[4] = advantages, targets
            #a = a_idx, piece, prob, adv, target_v, delta, delta_sum
        #bogus prios as always
        prios = 2*np.ones_like(r) #Very large prio :)
        data = (s,a,r,d)
        if augment:
            data, prios = self.augment_data(data, prios, state_fcn)
        return data, prios
