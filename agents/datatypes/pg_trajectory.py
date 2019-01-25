import numpy as np

class pg_trajectory:
    def __init__(self, n_actions, state_size):
        self.length = 0
        self.s, self.a, self.r, self.d, self.p = [], [], [], [], []
        self.future_state_size = (n_actions, *state_size[1:])
        self.future_state_mask_size = (n_actions,)

    def get_states(self):
        return self.s

    def add(self,e, end_of_trajectory=False):
        state,action,reward,s_prime,player,done = e
        self.s.append(state)
        self.p.append(player)
        if not end_of_trajectory:
            self.a.append(action)
            self.r.append(reward)
            self.d.append(done)
            self.length += 1

    def get_cumulative_reward(self, gamma_discount=0.999):
        return sum([x*gamma_discount**i for i,x in enumerate(self.r)])

    def process_trajectory(self, model, n_actions, state_fcn=None, sandbox=None, maskmaker=None, gamma_discount=0.999, lambda_discount=0.95):
        assert sandbox is not None and state_fcn is not None and maskmaker is not None, "You need to pass all kwargs"
        def future_state(s,p):
            sandbox.set(s)
            n_actions = len(sandbox.get_actions(player=int(p)))
            state_np = state_fcn(sandbox.simulate_all_actions(int(p)), player=[p for _ in range(n_actions)])
            return n_actions, state_np

        future_states      = np.zeros( (1+len(self), *self.future_state_size)      )
        future_states_mask = np.zeros( (1+len(self), *self.future_state_mask_size) )
        for i, sp in enumerate(zip(self.s,self.p)):
            s, p = sp
            n, state = future_state(s,p)
            future_states[i,:n,:]      = state
            future_states_mask[i,:] = maskmaker(n)

        _old_probs, v = model(future_states, future_states_mask)
        a = np.array(self.a).reshape((-1,1))
        r = np.array(self.r).reshape((-1,1))
        d = np.array(self.d).reshape((-1,1))
        td_errors = -v[:-1] + r -gamma_discount * v[1:] * (1-d)

        cum_adv = 0
        advantages = np.zeros((len(self),1))
        LG = lambda_discount * gamma_discount
        for t,td in reversed(list(enumerate(td_errors))):
            cum_adv = td - LG*cum_adv
            advantages[t,:] = cum_adv

        idxs = np.arange(len(self))
        old_probs =_old_probs[idxs,a[idxs,0]].reshape((-1,1))
        target_values = v[:-1] + advantages
        data = (future_states[:-1], future_states_mask[:-1] ,a ,old_probs, target_values, advantages)
        return data

    def __len__(self):
        return self.length
