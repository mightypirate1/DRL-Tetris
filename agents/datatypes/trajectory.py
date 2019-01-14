import numpy as np
from agents.datatypes import processed_trajectory
class trajectory:
    def __init__(self):
        self.length = 0
        self.s, self.r, self.d, self.p = [], [], [], []

        self.max_len = 5

    def get_states(self):
        return [self.s[i] for i,a in enumerate(self.a) if a is not None]

    def add(self,e, end_of_trajectory=False):
        state,action,reward,s_prime,player,done = e
        self.s.append(state)
        self.p.append(player)
        if not end_of_trajectory:
            self.r.append(reward)
            self.d.append(done)
            self.length += 1

    def get_length(self): #TODO 1 should suffice!
        return self.length
    def __len__(self):
        return self.get_length()

    def get_cumulative_reward(self, gamma_discount=0.999):
        return sum([x*gamma_discount**i for i,x in enumerate(self.r)])

    def process_trajectory(self, model, gamma_discount=0.999, lambda_discount=0.9, player_adjusted=True, update=False):
        _v,_ = model(self.s, player=self.p)
        #Our default setting is that we invert the sign of every other value, since it represents the other players state...
        if player_adjusted:
            for i in range(len(_v)):
                if i%2==1:
                    _v[i] = -_v[i]
        #Raw data
        v = np.array(_v).ravel()
        r = np.array(self.r).ravel()
        d = np.array(self.d).ravel().astype(np.int)
        advantages = np.zeros(r.shape)
        target_values = np.zeros(r.shape)

        #Compute the easy stuff!
        td_errors = -v[:-1] + r * gamma_discount * v[1:] * d
        priorities = np.abs(td_errors)

        #This computes a truncated version of the GAE from https://arxiv.org/abs/1506.02438
        # but here we also adjust for the 2-player/1-policy setting by flipping sign of every 2nd advantage
        for i in range(len(self)):
            weight = 0
            for j in range(i, min(i+self.max_len, self.length)):
                w = lambda_discount**(j-i)
                weight += w
                advantages[i] += w * td_errors[j]
            if player_adjusted and i%1 == 0:
                advantages[i] = -advantages[i]
                v[i]          = -v[i]
            advantages[i] /= weight
        target_values = v[:-1] + advantages

        if update: #TODO: This if statement smuggles out these numbers for updates in process_trajectory objects. It would be nicer if the prior parts of this method was shared in a tidier way!
            return target_values[0], priorities[0]

        data = []
        for i in range(self.length):
            I = min(i+self.max_len, len(self)) #This makes sure that all processed pieces have a state-list that is one longer than the other data it holds (compare the row "td_errors = ...")
            d = processed_trajectory.processed_trajectory(
                                                          self.s[i:I+1],
                                                          self.r[i:I  ],
                                                          self.d[i:I  ],
                                                          self.p[i:I+1],
                                                          target_values[i],
                                                          priorities[i],
                                                         )
            data.append(d)
        return data, priorities
    # def end_episode(self,sp,d):
    #     self.add((sp, None, None, d), end_of_trajectory=True)
