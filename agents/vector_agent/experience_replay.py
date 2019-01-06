from collections import deque
from math import ceil
import bisect
import numpy as np

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

    def get_random_sample(self, n_samples, alpha=1.0, beta=1.0):
        ##Random sampling of indices, with replacement and custom-distribution
        n = len(self.data)
        all_indices = np.arange(n)
        if self.prioritized:
            one_over_n = 1/n
            one_over_surpriserank = (1/(1+all_indices))
            p_unnormalized = one_over_surpriserank**alpha
            p = p_unnormalized / p_unnormalized.sum()
            is_weights_unnormalized = ((n*p)**(-beta))[:,np.newaxis]
            is_weights_all = is_weights_unnormalized/is_weights_unnormalized.reshape(-1).max()
        else:
            p_unnormalized = np.ones((n,))
            p = p_unnormalized / p_unnormalized.sum()
            is_weights_all = np.ones((n,1))
        indices = np.random.choice(all_indices, size=n_samples, p=p).tolist()
        ##Data collection, and index-tracking
        idx_dict = {}
        sample_data, sample_prio, sample_time_stamps = [None]*n_samples, [None]*n_samples, [None]*n_samples
        for i, idx in enumerate(indices):
            idx_dict[idx] = i
            sample_data[i] = self.data[idx]
            sample_prio[i] = self.prio[idx]
            sample_time_stamps[i] = self.time_stamps[idx]
        ##Remove the samples from the experience replay, so that they can be added back via the add samples function, with the same or a different prio
        indices_to_be_removed = list(idx_dict.keys())
        indices_to_be_removed.sort(reverse=True)
        for idx in indices_to_be_removed:
            del self.data[idx]
            del self.prio[idx]
            del self.time_stamps[idx]
        return sample_data, [-s for s in sample_prio], sample_time_stamps, is_weights_all[indices,:], list(idx_dict.values())
    def add_samples(self, new_data, new_prio, filter=None, time_stamps=None):
        if filter is None:
            filter = [x for x in range(len(new_data))]
        if time_stamps is None:
            time_stamps = [self.time + x for x in range(len(new_data))]
        data = [ new_data[x] for x in filter]
        prio = [-new_prio[x] for x in filter]
        time_stamps = [time_stamps[x] for x in filter]
        for i,d in enumerate(data):
            idx = bisect.bisect_right(self.prio, prio[i]) #Find index to insert
            self.data.insert(idx,d)
            self.prio.insert(idx,prio[i])
            self.time_stamps.insert(idx, time_stamps[i])
        if len(self.data) > self.max_size:
            self.remove_old( ceil(self.max_size*0.1) )
        self.time += len(filter)

    def remove_old(self, cut_off):
        if self.log is not None:
            self.log.debug("REMOVING {} OBJECTS".format(cut_off))
        self.time -= cut_off
        for idx in reversed(range(len(self.data))):
            self.time_stamps[idx] -= cut_off
            if self.time_stamps[idx] < 0:
                del self.data[idx]
                del self.prio[idx]
                del self.time_stamps[idx]
    def __len__(self):
        return len(self.data)
