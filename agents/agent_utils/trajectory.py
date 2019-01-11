class trajectory:
    def __init__(self, state_size):
        self.length = 0
        self.state_size = state_size
        self.s, self.r, self.d, self.p = [], [], [], []
    def get_states(self):
        return [self.s[i] for i,a in enumerate(self.a) if a is not None]
    def add(self,e, end_of_trajectory=False):
        state,action,reward,s_prime,player,done = e
        self.s.append(state)
        if not end_of_trajectory:
            self.r.append(reward)
            self.d.append(done)
            self.length += 1
    def get_length(self):
        return self.length
    def get_cumulative_reward(self, gamma_discount=0.99):
        return sum([x*gamma_discount**i for i,x in enumerate(self.r)])

    def process_trajectory(self, model, gamma_discount=0.99, lambda_discount=0.95):
        print("I PRETEND TO DO MATH!")
        n = 10
        return [i for i in range(n)], [["data...."] for _ in range(n)]
        advantages     = [0 for x in range(self.length)]
        td_errors      = [0 for x in range(self.length)]
        target_values  = [0 for x in range(self.length)]
        p,v = model(self.s)
        for i in range(self.length):
            td_errors[i] = -v[i] + self.r[i] + gamma_discount*v[i+1]*int(not self.d[i])
        for i in range(self.length):
            for j in range(i, self.length):
                advantages[i] += lambda_discount**(j-i) * td_errors[j]
        #ADNANTAGE METHOD
        target_values = [x+y for x,y in zip(v.tolist(),advantages)]
        #OTHER METHOD
        # for i in range(self.length):
        #     target_values[i] = self.r[i] + gamma_discount * v[i+1]
        old_probabilities = [[p[i,a]] for i,a in enumerate(self.a)]
        return advantages, target_values, old_probabilities
    def end_episode(self,sp,d):
        self.add((sp, None, None, d), end_of_trajectory=True)
    def __len__(self):
        return self.get_length()
