import numpy as np
from agents.datatypes import processed_trajectory

DEBUG = False #<----------
import time  #<----------


class trajectory:
    def __init__(self):
        self.length = 0
        self.s, self.r, self.d, self.p = [], [], [], []
        self.max_len = 5



        self.last_print  = 0 #<----------




    def get_states(self):
        return [self.s[i] for i,a in enumerate(self.a) if a is not None]

    def add(self,e, end_of_trajectory=False):
        state,action,reward,s_prime,player,done = e
        if DEBUG: print(state,reward,player,done, "<- state,reward,player,done"); input()
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

    def process_trajectory(self, model, gamma_discount=0.99, lambda_discount=0.9, player_adjusted=True, update=False):
        _v,_ = model(self.s, player=self.p)
        #Our default setting is that we invert the sign of every other value, since it represents the other players state...

        # Raw data
        v = np.array(_v).ravel()
        r = np.array(self.r).ravel()
        d = np.array(self.d).ravel().astype(np.int)
        #To be filled in...
        advantages = np.zeros(r.shape)
        target_values = np.zeros(r.shape)

        # ''' - - - - - '''                                             #<----------
        # losetime = np.argmax(np.abs(r))                   #<----------
        # v = np.arange(v.size)/v.size                  #<----------
        # for i in range(losetime%2,v.size,2):                  #<----------
        #     v[i] = -v[i]                  #<----------
        # print(np.round(v,decimals=2))                 #<----------
        # print(r)                  #<----------
        # # exit()                  #<----------
        # ''' - - - - - '''                 #<----------



        #Compute the easy stuff!
        td_errors = -v[:-1] + r - gamma_discount * v[1:] * (1 - d)
        priorities = np.abs(td_errors)
        if DEBUG: print(td_errors)

        # ''' Multi-step advantages... maybe broken '''
        # #This computes a truncated version of the GAE from https://arxiv.org/abs/1506.02438
        # # but here we also adjust for the 2-player/1-policy setting by flipping sign of every 2nd advantage
        # sum = 0.0
        # LG = lambda_discount * gamma_discount
        # # L,G = lambda_discount, gamma_discount
        # weight_sum = 0.0
        # for i in reversed(range(len(self))):
        #     sum = sum * (-LG) + td_errors[i]
        #     weight_sum = LG*weight_sum+1
        #     advantages[i] = sum / weight_sum
        #     if DEBUG: print("!:",advantages[i],sum,td_errors[i], LG)
        # target_values = v[:-1] + advantages

        #DEBUG-print
        # if time.time() - 3 > self.last_print and type(self) is trajectory:
        #     self.last_print = time.time()
        #     for _s,_r,_d,_p,i in zip(self.s,self.r,self.d,self.p, range(len(self))):
        #         print(_s,_r,_d,_p,i,":",np.round(v[i],decimals=2),"\t->",np.round(target_values[i])," (",np.round(td_errors[i]),")")

        ''' 1-step avdantates... '''
        target_values = v[:-1] + td_errors

        if update: #TODO: This if statement smuggles out these numbers for updates in processed_trajectory objects. It would be nicer if the prior parts of this method was shared in a tidier way!
            return target_values[0], priorities[0]

        if DEBUG: print([(t,r,td,p) for t,r,td,p in zip(target_values,r,td_errors,self.p)],"<- target, reward, td_errs, player");input()

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
