import numpy as np
import scipy.stats
from .. import agent_utils

class experience_replay:
    def __init__(self, max_size=None, state_size=None, log=None):
        self.log        = log
        self.max_size   = max_size
        self.vector_state_size, self.visual_state_size = state_size
        self.vector_states   = [np.zeros((max_size,*s[1:])) for s in self.vector_state_size]
        self.visual_states   = [np.zeros((max_size,*s[1:])) for s in self.visual_state_size]
        self.vector_s_primes = [np.zeros((max_size,*s[1:])) for s in self.vector_state_size]
        self.visual_s_primes = [np.zeros((max_size,*s[1:])) for s in self.visual_state_size]
        self.rewards  = np.zeros((max_size,1))
        self.dones    = np.zeros((max_size,1))
        self.prios    = -np.ones((max_size,1))
        self.current_size  = 0
        self.current_idx   = 0
        self.total_samples = 0

    def get_random_sample(self, n_samples, alpha=1.0, beta=1.0, remove=False, compute_stats=False):
        #Create the sampling distribution (see paper for details)
        n = self.current_size
        all_indices = np.arange(n)
        #make ranking
        rank = 1+n-scipy.stats.rankdata(self.prios[:n].ravel(), method='ordinal')
        #make a ranking-based probability disribution (pareto-ish)
        one_over_rank = 1/rank #Rank-based sampling
        p_unnormalized = one_over_rank**alpha
        p = p_unnormalized / p_unnormalized.sum() #sampling distribution done
        is_weights_unnormalized = ((n*p)**(-beta))[:,np.newaxis] #Compute the importance sampling weights
        is_weights_all = is_weights_unnormalized/is_weights_unnormalized.reshape(-1).max()

        #Sample indices to make a batch out of!
        idx_dict = {}
        indices = np.random.choice(all_indices, replace=True, size=n_samples, p=p).tolist()

        ##Data collection, and index-tracking
        for i, idx in enumerate(indices):
            idx_dict[idx] = i #This makes sure that idx_dict keeps track of the position of the last occurance of each index in the sample

        #Return values!
        filter = idx_dict
        is_weights = is_weights_all[indices]
        data = (
                ([vs[indices,:] for vs in self.vector_states],   [vs[indices,:] for vs in self.visual_states]),
                ([vs[indices,:] for vs in self.vector_s_primes], [vs[indices,:] for vs in self.visual_s_primes]),
                None,
                self.rewards[indices,:],
                self.dones[indices,:],
                )

        #Stats?
        if compute_stats:
            iwu = is_weights_unnormalized.ravel()
            stats = {
                     "ExpRep-iwu_max"  : iwu.max(),
                     "ExpRep-iwu_mean" : iwu.mean(),
                     "ExpRep-iwu_min"  : iwu.min(),
                     "ExpRep-prio_max"  : (-self.prios).max(),
                     "ExpRep-prio_mean" : (-self.prios).mean(),
                     "ExpRep-prio_min"  : (-self.prios).min(),
                     "ExpRep-size"      : self.current_size,
                    }
        else:
            stats = {}
        return data, is_weights, filter, stats

    def add_samples(self, data, prio):
        s, sp,_,r,d = data
        vec_s, vis_s   = s
        vec_sp, vis_sp = sp
        n = prio.size
        idxs = [x%self.max_size for x in range(self.current_idx, self.current_idx+n)]
        for i,vs in enumerate(self.vector_states):
            vs[idxs,:]   = vec_s[i]
        for i,vs in enumerate(self.visual_states):
            vs[idxs,:]   = vis_s[i]
        for i,vs in enumerate(self.vector_s_primes):
            vs[idxs,:]   = vec_sp[i]
        for i,vs in enumerate(self.visual_s_primes):
            vs[idxs,:]   = vis_sp[i]
        self.rewards[idxs,:]  = r
        self.dones[idxs,:]    = d
        self.prios[idxs,:]    = prio
        self.current_idx += n
        self.current_size = min(self.current_size+n, self.max_size)
        self.total_samples += n

    def update_prios(self, new_prios, filter):
        self.prios[list(filter.keys()),:] = new_prios[list(filter.values()),:]

    def __len__(self):
        return self.current_size
