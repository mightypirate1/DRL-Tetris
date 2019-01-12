import numpy as np
from math import ceil
from heapq import merge
import bisect
import aux.utils as utils
from .. import agent_utils

'''
This class is made to store experiences (arbitrary data basically).
It is a simple implementation of https://arxiv.org/abs/1511.05952 and hopefully
correct and efficient :)

You feed it lists with data, priorities and optional timestamp.
It creates a data_entry type object that it keeps sorted based on priority.

You can get a random sample (of n data_entry objects) with get_random_sample(n).
The caller "promises" to replace them later. The caller is responsible for
updating the priorities before replacing them if that is part of the algorithm.

Since the sampling is with replacement, get_random_sample also provides a filter.
Just save the filter, and pass it to replace_samples when you replace them (after
having updated the prio).

Most importand fcns to know:
get_random_sample(n)
    gives out a random (with replacement!) sample of data_entry objects.

add_samples(data,prio, time_stamps=None)
    if you have collected fresh data from the env. use this to get it in the buffer.

replace_samples(data, filter)
    if you got a sample from get_random_sample it came with a filter. pass that
    sample (update the priority value first if you plan to do that) together with
    the filter to replace them properly in the buffer.

merge_in(another_buffer)
    merges the data of another_buffer into this buffer, maintaining sorted order.
'''
class experience_replay:
    def __init__(self, size, prioritized=True, log=None):
        self.test_flag = False
        self.max_size = size
        self.log = log
        self.data = list()
        self.time = 0
        self.prioritized = prioritized
    def get_random_sample(self, n_samples, alpha=1.0, beta=1.0, remove_samples=True):
        #Create the sampling distribution (see paper for details)
        n = len(self)
        all_indices = np.arange(n)
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
        sample_data = [None]*n_samples
        for i, idx in enumerate(indices):
            idx_dict[idx]         = i #This makes sure that idx_dict keeps track of the position of the last occurance of each index in the sample
            sample_data[i]        = self.data[idx]
            sample_data[i].set_isw(is_weights_all[idx])
        filter = list(idx_dict.values()) #The filter is a list containing the last occurance of each unique sample. This means, that if we add back as many samples as is in the filter, the number of samples in the buffer is unchanged.

        #If the caller did not specify that they will not give the samples back (by setting remove_samples=False), we assume they want to update priority etc. Since we do not have an update-prio fcn (this may be on the wish-list) we instead just remove the samples, and expect the caller to replace them later
        if remove_samples:
            indices_to_be_removed = list(idx_dict.keys())
            indices_to_be_removed.sort(reverse=True)
            for idx in indices_to_be_removed:
                del self.data[idx]

        #We now give the sampled data to the caller.
        #The filter is provided so that they can compute new priorities, update the data_entries, and put them back through replace_samples.
        #IMPORTANT: pass the filter given here to the replace_samples method when replacing samples, otherwise any data that occured n times in the sample given out, will be added n times back, which is NOT how you want to do it most likely...
        return sample_data, filter

    def add_samples(self, data, prio, time_stamps=None, filter=None):
        #Add time_stamps if needed, and filter the input if we are told to!
        if time_stamps is None: time_stamps = [t for t in range(self.time,self.time+len(data),1)]
        if filter is not None:
            data        = [       data[x] for x in filter]
            prio        = [       prio[x] for x in filter]
            time_stamps = [time_stamps[x] for x in filter]
        #Now that we have time-stamped and filtered data, we add it one sample at a time, keeping all deques sorted!
        i = 0
        for d,p,t in zip(data,prio,time_stamps):
            self.replace_sample(agent_utils.data_entry(d,p,t))
        if len(self) > self.max_size:
            self.remove_old( ceil(self.max_size*0.1) )
        self.time += len(data)

    def replace_sample(self, entry):
        if self.prioritized:
            #What I think this does (and it seems to do) is:
            # 1: find the first idx s.t. self.prio[idx] < prio[i]
            # 2: insert entries e in deques s.t. e is and idx and subsequent entries are pushed to their previous index + 1
            idx = bisect.bisect_left(self.data, entry) #Find index to insert
            self.data.insert(idx, entry)
        else:
            self.data.append(entry)

    def replace_samples(self, entries, filter):
        for idx in filter:
            self.replace_sample(entries[idx])

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

    def merge_in(self, exp_rep, mode):
        return list(merge(self.data,exp_rep.data))

    def __len__(self):
        return len(self.data)
