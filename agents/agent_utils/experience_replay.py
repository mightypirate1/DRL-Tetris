import numpy as np
from math import ceil
from heapq import merge
import bisect
import aux.utils as utils
from .. import agent_utils
from agents.datatypes import replaybuffer_entry

'''
This class is made to store experiences (arbitrary data basically).
It is a simple implementation of https://arxiv.org/abs/1511.05952 and hopefully
correct and efficient :)

You feed it lists with data and priorities.
It creates a replaybuffer_entry type object that it keeps sorted based on
priority.

You can get a random sample (of n replaybuffer_entry objects) with
get_random_sample(n). The data you put in is accessible in those entries:
For replaybuffer_entry e:
    e.data #Has the data
    e.prio #Has the priority
    e.is_weight #Has the importance sampling weight
    e.time_stamp #Has the time-stamp
Since e is a class instance; unless you did pickle your sample or something, you
can update the priority of your sample, and that updates the "source" in the
buffer.

Most importand fcns to know:
get_random_sample(n, remove=False)
    gives out a random (with replacement!) sample of replaybuffer_entry objects.
    (1) remove=True removes the samples entries from the buffer. In this mode,
    get_random_sample returns 2 lists. The first is the samples, the second is a
    filter, which is passed to the replace_samples function if you do want to
    put them back.
    (2) remove=False (default) leaves the samples in the buffer. As the samples
    are of type replaybuffer_entry, you can

add_samples(data,prio)
    if you have collected fresh data from the env. use this to get it in the
    buffer. If data is of arbitrary type, add it with priorities.

replace_samples(data, filter)
    If you got a sample from get_random_sample(n,remove=True) it came with a
    filter. pass that sample (update the priority value first if you plan to do
    that) together with the filter to replace them properly in the buffer. This
    is because the sampling was done WITH REPLACEMENT, so the entries in the
    sample is NOT UNIQUE. The filter keeps track of what entries needs to be
    put back (and if you do something other than just updating stuff, you should
    figure out how the filter works, and use it in your code to ensure
    correctness).

merge_in(another_buffer)
    merges the data of another_buffer into this buffer, maintaining sorted
    order.

TO MAKE IT WORK IN A MULTI-PROCESS SETTING:
The buffer keeps its own clock to keep track of what samples are to be removed.
The remove-method of the class (that is used to regulate the size of the buffer)
assumes that these indices are unique and in order. Thus merge_in is very
distruptive since it just "pours in" new samples. The fix for this when used in
a multi-worker/single-trainer setting is that the workers send data frequently
enough so that they NEVER have to empty their own buffer EVER. On top of that,
a time_fcn is passed to the constructor, which maps local time to a unique
global time (see below). If one then synchs the local/global times with the
set_time / get_time functions frequently enough, the outcome should be near
perfect. Example time_fcn is lambda x : a*x+b where a is total number of
workers, and b is worker id (assuming workers are numbered
0,1,...,total_number_of_workers).
'''
class experience_replay:
    def __init__(self, size, prioritized=True, log=None, time_fcn=lambda x:x):
        self.max_size = size
        self.log = log
        self.data = list()
        self.time = 0
        self.time_fcn = time_fcn
        self.time_increment = time_fcn(1)-time_fcn(0)
        self.prioritized = prioritized
        #
        self.total_removed = 0  #Total entries removed ever
        self.remove_count = 0.1 * self.max_size

    def get_random_sample(self, n_samples, alpha=1.0, beta=1.0, remove=False):
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

        #If the caller did not specify that they will not give the samples back (by setting remove=False), we assume they want to update priority etc. Since we do not have an update-prio fcn (this may be on the wish-list) we instead just remove the samples, and expect the caller to replace them later
        if remove:
            indices_to_be_removed = list(idx_dict.keys())
            indices_to_be_removed.sort(reverse=True)
            for idx in indices_to_be_removed:
                del self.data[idx]
            #We now give the sampled data to the caller.
            #The filter is provided so that they can compute new priorities, update the data_entries, and put them back through replace_samples.
            #IMPORTANT: pass the filter given here to the replace_samples method when replacing samples, otherwise any data that occured n times in the sample given out, will be added n times back, which is NOT how you want to do it most likely...
        return sample_data, filter

    def add_samples(self, data, prio):
        #Add time_stamps if needed, and filter the input if we are told to!
        time_stamps = [self.time_fcn(t) for t in range(self.time,self.time+len(data),1)]
        for d,p,t in zip(data,prio,time_stamps):
            if issubclass(type(d), replaybuffer_entry):
                d.prio, d.time_stamp = p, t
                self.replace_sample(d)
            else:
                self.replace_sample(replaybuffer_entry(d,p,t))
        self.test_length()
        self.time += self.time_increment * len(data)

    def replace_sample(self, entry):
        assert issubclass(type(entry), replaybuffer_entry), "replace_sample called with type:{}  -- replaybuffer_entry or subclass was expected"
        if self.prioritized:
            #What I think this does (and it seems to do) is:
            # 1: find the first idx s.t. self.prio[idx] < prio[i]
            # 2: insert entries e in deques s.t. e is and idx and subsequent entries are pushed to their previous index + 1
            idx = bisect.bisect_left(self.data, entry) #Find index to insert
            self.data.insert(idx, entry)
        else:
            self.data.append(entry)

    def replace_samples(self, entries, filter=None):
        for idx in (filter if filter is not None else range(len(entries))):
            self.replace_sample(entries[idx])

    def remove_old(self):
        threshold = self.total_removed + self.remove_count
        count = 0
        for i, d in reversed(list(enumerate(self.data))):
            if d.time_stamp < threshold:
                del self.data[i]
                count += 1
        self.total_removed += count
        if self.log is not None:
            self.log.debug("REMOVED {} OBJECTS ({} of total)".format(count,count/len(self)))

    def merge_in(self, data):
        if type(data) is experience_replay:
            data = data.data
        self.data = list(merge(self.data,data))
        self.test_length()

    def test_length(self):
        if len(self) > self.max_size:
            self.remove_old()

    def reset(self):
        self.__init__(prioritized=self.prioritized,log=self.log, time_fcn=self.time_fcn)

    def clear_buffer(self):
        self.total_removed += len(self)
        self.data = list()

    def set_time(self, t):
        self.time = t

    def get_time(self):
        return self.total_removed + len(self) #This should count all elements that ever passed through the buffer

    def print_out(self):
        for d in self.data:
            print(d)

    def __len__(self):
        return len(self.data)
