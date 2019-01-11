from collections import deque
from math import ceil
import bisect
import numpy as np
import aux.utils as utils

'''
This class is made to store experiences (arbitrary data).
It is a simple implementation of https://arxiv.org/abs/1511.05952 and hopefully
correct and efficient :)

DESCRIPTION:
experience_replay is made to use prioritization and timestamps although both can
be turned off. (prioritization is turned off in the constructor but you still
need to pass some bogus number to the methods if the signature so demands.
time-stamps are "turned off" by always passing time_stamps=0 to add_samples.)

The class stores data, and allows for manipulation of it with the following
methods:

get_random_sample(n):
    short description: returns a random sample WITH replacement of n. It's
    possible to change alpha, gamma (see paper for details).

    longer description:
    the goal is to supply a random sample of size n, containing:
        data,
        priorities,
        importance sampling weights,
        time-stamps

    since probably want to replace the samples after updating the priorities, we
    use a filter which is thus also returned. The purpose of the filter is that
    the caller of get_random_sample updates the priorities, and then add back
    ALL samples together with the accompanying meta-data (prio/time-stamps) AND
    also passing the filter. This makes it so that we only add back samples that
    were sampled multiple times (replacement, remember..) are only added back
    once.

    the full return is thus: return data,prio,time-stamps,is-weights,filter

add_samples(data,prio,time_stamps=None, filter=None):
    This adds lists of samples to the data, while maintaining the list sorted
    ascendingly in priority.

    If no time stamps are specified, the buffer keeps a
    clock, and uses that for time-stamps. You can also give a filter (i.e. a
    list of indices) which if specified, discards all the samples not in the
    filter. This is to make replacement easier, since the caller does not need
    to figure out.

    Use-case 1: I want to add a new sample (e.g.fresh from the environment)!
        Just call
        ´´´
            experience_replay.add_samples(data, prio)
        ´´´
        to add a newly gathered data-point

    Use-case 2: I want to replace data that the experience_replay previously
    gave me!
        instead call:
            experience_replay.add_samples(
                                          data,
                                          updated_prio,
                                          time_stamps=time_stamps,
                                          filter=filter,
                                         )
    to replace a (data,prio,time_stamps,is_weights,filter)-tuple received from
    the get_random_sample method.

merge(experience_replay)
    merges self with experience_replay.
'''

class experience_replay:
    def __init__(self, size, prioritized=True, log=None):
        self.test_flag = False
        self.max_size = size
        self.log = log
        self.data = deque()
        self.prio = deque()
        self.time_stamps = deque()
        self.time = 0
        self.prioritized = prioritized

    def get_random_sample(self, n_samples, alpha=1.0, beta=1.0, remove_samples=True):
        ##Random sampling of indices, with replacement and custom-distribution
        n = len(self)
        all_indices = np.arange(n)

        #Create the sampling distribution (see paper for details)
        if self.prioritized:
            one_over_n = 1/n #Rank-based sampling
            one_over_surpriserank = (1/(n-all_indices))
            p_unnormalized = one_over_surpriserank**alpha
        else:
            p_unnormalized = np.ones((n,))
            p = p_unnormalized / p_unnormalized.sum()
            is_weights_all = np.ones((n,1))
        p = p_unnormalized / p_unnormalized.sum() #sampling distribution done
        is_weights_unnormalized = ((n*p)**(-beta))[:,np.newaxis] #Compute the importance sampling weights
        is_weights_all = is_weights_unnormalized/is_weights_unnormalized.reshape(-1).max()

        #Sample indices to make a batch out of!
        idx_dict = {}
        indices = np.random.choice(all_indices, size=n_samples, p=p).tolist()

        ##Data collection, and index-tracking
        sample_data, sample_prio, sample_time_stamps = [None]*n_samples, [None]*n_samples, [None]*n_samples
        for i, idx in enumerate(indices):
            idx_dict[idx]         = i #This makes sure that idx_dict keeps track of the position of the last occurance of each index in the sample
            sample_data[i]        = self.data[idx]
            sample_prio[i]        = self.prio[idx]
            sample_time_stamps[i] = self.time_stamps[idx]
        sample_is_weights = is_weights_all[indices,:]
        filter = list(idx_dict.values()) #The filter is a list containing the last occurance of each unique sample. This means, that if we add back as many samples as is in the filter, the number of samples in the buffer is unchanged.

        #If the caller did not specify that they will not give the samples back (by setting remove_samples=False), we assume they want to update priority etc. Since we do not have an update-prio fcn (this may be on the wish-list) we instead just remove the samples, and expect the caller to replace them later
        if remove_samples:
            indices_to_be_removed = list(idx_dict.keys())
            indices_to_be_removed.sort(reverse=True)
            for idx in indices_to_be_removed:
                del self.data[idx]
                del self.prio[idx]
                del self.time_stamps[idx]

        #We now give the sampled data,prio,time-stamps, is-weights to the caller.
        #The filter is provided, so that they can compute new priorities, and pass those back to us through the add_samples methodself.
        #IMPORTANT: pass the filter given here to the add_samples method when replacing samples, otherwise any data that occured n times in the sample given out, will be added n times back, which is NOT how you want to do it most likely...
        return sample_data, sample_prio, sample_time_stamps, sample_is_weights, filter

    def add_samples(self, data, prio, time_stamps=None, filter=None):
        #Add time_stamps if needed, and filter the input if we are told to!
        if time_stamps is None: time_stamps = [t for t in range(self.time,self.time+len(data),1)]
        if filter is not None:
            data        = [       data[x] for x in filter]
            prio        = [       prio[x] for x in filter]
            time_stamps = [time_stamps[x] for x in filter]

        #Now that we have time-stamped and filtered data, we add it one sample at a time, keeping all deques sorted!
        for i,d in enumerate(data):
            if self.prioritized:
                #What I think this does (and it seems to do) is:
                # 1: find the first idx s.t. self.prio[idx] < prio[i]
                # 2: insert entries e in deques s.t. e is and idx and subsequent entries are pushed to their previous index + 1
                idx = bisect.bisect_left(self.prio, prio[i]) #Find index to insert
                self.data.insert(idx, d      )
                self.prio.insert(idx, prio[i])
                self.time_stamps.insert(idx, time_stamps[i])
            else:
                self.data.append(d      )
                self.prio.append(prio[i])
                self.time_stamps.append(time_stamps[i])
        if len(self) > self.max_size:
            self.remove_old( ceil(self.max_size*0.1) )
        self.time += len(data)

    def remove_old(self, cut_off):
        if self.log is not None:
            self.log.debug("REMOVING {} OBJECTS".format(cut_off))
        self.time -= cut_off
        for idx in reversed(range(len(self))):
            self.time_stamps[idx] -= cut_off
            if self.time_stamps[idx] < 0:
                del self.data[idx]
                del self.prio[idx]
                del self.time_stamps[idx]

    def merge(self, exp_rep):
        def weave(x,y):
            idx_x = idx_y = 0
            ret = [None for _ in range(len(x)+len(y))]
            for i in range(len(ret)):
                if idx_x < len(x) and idx_y < len(y):
                    pick_x = x[idx_x] < y[idx_y]
                    ret[i] = x[idx_x] if pick_x else y[idx_y]
                    if pick_x: idx_x += 1
                    else: idx_y += 1
                elif idx_x < len(x):
                    ret[i] = x[idx_x]
                    idx_x += 1
                else:
                    ret[i] = y[idx_y]
                    idx_y += 1
            return ret

        a = deque((0 + 2*np.arange(10000000)).tolist())
        b = deque((1 + 2*np.arange(10000000)).tolist())
        import time
        t = time.time()
        # weave(a,b)
        sorted(a+b)
        print("weave: ",time.time()-t)
        from heapq import merge
        t = time.time()
        list(merge(a,b))
        print("merge:",time.time()-t);exit()

    def __len__(self):
        return len(self.data)
