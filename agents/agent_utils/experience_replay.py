import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.stats
from .. import agent_utils

class experience_replay:
    def __init__(self, max_size=None, state_size=None, k_step=1, log=None, sample_mode='rank'):
        self.log        = log
        self.k_step, self.state_len = k_step, k_step+1
        self.vector_state_size, self.visual_state_size = state_size

        #Underlying data
        self._max_size = max_size
        self._vector_states   = [np.zeros((self._max_size,*s[1:]), dtype=np.uint8) for s in self.vector_state_size]
        self._visual_states   = [np.zeros((self._max_size,*s[1:]), dtype=np.uint8) for s in self.visual_state_size]
        self._actions  = np.zeros((self._max_size,1), dtype=np.uint8)
        self._dones    = np.zeros((self._max_size,1), dtype=np.uint8)
        self._rewards  = np.zeros((self._max_size,1), dtype=np.float32)

        #Presented data
        self.max_size = max_size - k_step
        self.vector_states = [agent_utils.k_step_view(_v, k_step+1) for _v in self._vector_states]
        self.visual_states = [agent_utils.k_step_view(_v, k_step+1) for _v in self._visual_states]
        self.actions       =  agent_utils.k_step_view(self._actions, k_step+1)
        self.dones         =  agent_utils.k_step_view(self._dones,   k_step+1)
        self.rewards       =  agent_utils.k_step_view(self._rewards, k_step+1)
        self.prios    = -np.ones((self.max_size,1), dtype=np.float32)

        #Inner workings...
        self.current_size  = 0
        self.current_idx   = 0
        self.total_samples = 0
        self.sample_mode = sample_mode
        self.resort_fraction = 0.5

    def get_random_sample(self, n_samples, alpha=1.0, beta=1.0, remove=False, compute_stats=False):
        #Create the sampling distribution (see Schaul et al. for details)
        n = self.current_size
        all_indices = np.arange(n)
        if self.sample_mode == 'rank':
            #make ranking
            rank = 1+n-scipy.stats.rankdata(self.prios[:n].ravel(), method='ordinal')
            #make a ranking-based probability disribution (pareto-ish)
            one_over_rank = 1/rank #Rank-based sampling
            p_unnormalized = one_over_rank**alpha
        else:
            #Proportional
            p_unnormalized = (self.prios[:n].ravel() + 0.0001)**alpha

        p = p_unnormalized / p_unnormalized.sum() #sampling distribution done
        is_weights_unnormalized = ((n*p)**(-beta))[:,np.newaxis] #Compute the importance sampling weights
        is_weights_all = is_weights_unnormalized/is_weights_unnormalized.reshape(-1).max()

        #Sample indices to make a batch out of!
        idx_dict = {}
        indices = np.random.choice(all_indices, replace=True, size=n_samples, p=p).tolist()

        ##Data collection, and index-tracking
        i = 0
        for idx in indices:
            idx_dict[idx] = i #This makes sure that idx_dict keeps track of the position of the last occurance of each index in the sample
            i += 1

        #Return values!
        filter = idx_dict #This keeps track of which internal index
        is_weights = is_weights_all[indices]
        data = self.retrieve_samples_by_idx(indices)

        #Stats?
        if compute_stats:
            iwu = is_weights_unnormalized[:self.current_size].ravel()
            stats = {
                     "ExpRep-iwu_max"  : iwu.max(),
                     "ExpRep-iwu_mean" : iwu.mean(),
                     "ExpRep-iwu_min"  : iwu.min(),
                     "ExpRep-prio_max"  : self.prios[:n].max(),
                     "ExpRep-prio_mean" : self.prios[:n].mean(),
                     "ExpRep-prio_min"  : self.prios[:n].min(),
                     "ExpRep-sample_size"     : self.current_size,
                    }
        else:
            stats = {}
        return data, is_weights, filter, stats

    def add_samples(self, data, prio):
        s, a, r, d = data
        vec_s, vis_s   = s
        n = prio.size
        idxs = self.add_indices(n)
        for i,vs in enumerate(self._vector_states):
            vs[idxs,:]   = vec_s[i]
        for i,vs in enumerate(self._visual_states):
            vs[idxs,:]   = vis_s[i]
        self._rewards[idxs,:]  = r
        self._dones[idxs,:]    = d
        self._actions[idxs,:]    = a
        self.prios[idxs,:]    = prio
        self.current_idx += n
        self.current_size = min(self.current_size+n, self.max_size)
        self.total_samples += n

    def add_indices(self, n):
        if self.current_idx + n < self._max_size:
            return slice(self.current_idx, self.current_idx+n,1)
        self.current_size = self.current_idx
        self.prios[self.current_idx:] = -1
        self.current_idx = 0
        return slice(0,n,1)

    def update_prios(self, new_prios, filter):
        self.prios[list(filter.keys()),:] = new_prios[list(filter.values()),:]

    def retrieve_samples_by_idx(self, indices):
        data = (
                ([vs[indices,:] for vs in self.vector_states],   [vs[indices,:] for vs in self.visual_states]),
                self.actions[indices,:],
                self.rewards[indices,:],
                self.dones[indices,:],
                )
        return data

    def __len__(self):
        return self.current_size
