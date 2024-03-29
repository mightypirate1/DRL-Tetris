import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.stats
from .. import agent_utils

class experience_replay:
    def __init__(self, max_size=None, action_size=1, state_size=None, k_step=1, log=None, sample_mode='rank'):
        self.logger = log
        self.stats = {}
        self.vector_state_size, self.visual_state_size = state_size
        self.actions_not_list = False
        if type(action_size) not in [list, tuple]:
            self.actions_not_list = True
            action_size = [[action_size]]
        self.action_size = action_size
        self.n_actions = len(self.action_size)
        self._max_size = max_size
        self.max_size = max_size - k_step
        self.k_step = k_step
        self.sample_mode = sample_mode
        self.total_samples = 0
        self.initialize()

    def initialize(self):
        #Underlying data
        self._vector_states   = [np.zeros((self._max_size,*s[1:]), dtype=np.uint8) for s in self.vector_state_size]
        self._visual_states   = [np.zeros((self._max_size,*s[1:]), dtype=np.uint8) for s in self.visual_state_size]
        self._actions  = [np.zeros((self._max_size,*a_size), dtype=np.float32) for a_size in self.action_size]
        self._dones    =  np.zeros((self._max_size,1), dtype=np.uint8)
        self._rewards  =  np.zeros((self._max_size,1), dtype=np.float32)
        #Presented data
        self.vector_states = [agent_utils.k_step_view(_v, self.k_step+1) for _v in self._vector_states]
        self.visual_states = [agent_utils.k_step_view(_v, self.k_step+1) for _v in self._visual_states]
        self.actions       = [agent_utils.k_step_view(_a, self.k_step+1) for _a in self._actions      ]
        self.dones         =  agent_utils.k_step_view(self._dones,   self.k_step+1)
        self.rewards       =  agent_utils.k_step_view(self._rewards, self.k_step+1)
        self.prios    = -np.ones((self.max_size,1), dtype=np.float32)
        #Inner workings...
        self.current_size  = 0
        self.current_idx   = 0

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
        elif sample_mode == 'proportional':
            #Proportional
            p_unnormalized = (self.prios[:n].ravel() + 0.0001)**alpha
        else:
            raise Exception("get_random_sample called, but exp-rep set to non-random")
        p = p_unnormalized / p_unnormalized.sum() #sampling distribution done
        is_weights_unnormalized = ((n*p)**(-beta))[:,np.newaxis] #Compute the importance sampling weights
        is_weights_all = is_weights_unnormalized/is_weights_unnormalized.reshape(-1).max()

        #Sample indices to make a batch out of!
        idx_dict = {}
        indices = np.random.choice(all_indices, replace=False, size=n_samples, p=p).tolist()

        ##Index-tracking to make prio-updates easy
        i = 0
        for idx in indices:
            idx_dict[idx] = i #This makes sure that idx_dict keeps track of the position of the last occurance of each index in the sample
            i += 1

        #Return values!
        data = self.retrieve_samples_by_idx(indices)
        filter = idx_dict #This keeps track of which internal index
        is_weights = is_weights_all[indices]

        #Stats?
        if compute_stats:
            iwu = is_weights_unnormalized[:self.current_size].ravel()
            self.stats = {
                          "ExpRep-iwu_max"  : iwu.max(),
                          "ExpRep-iwu_mean" : iwu.mean(),
                          "ExpRep-iwu_min"  : iwu.min(),
                          "ExpRep-prio_max"  : self.prios[:n].max(),
                          "ExpRep-prio_mean" : self.prios[:n].mean(),
                          "ExpRep-prio_min"  : self.prios[:n].min(),
                          "ExpRep-sample_size"     : self.current_size,
                          }
        return data, is_weights, filter


    def retrieve_and_clear(self, compute_stats=False):
        data = self.retrieve_all()
        self.initialize()
        return data
        
    def retrieve_all(self, compute_stats=False):
        #retrieve
        all_indices = np.arange(self.current_size)
        data = self.retrieve_samples_by_idx(all_indices)
        #Stats?
        if compute_stats:
            self.stats = {
                          "ExpRep-sample_size"     : self.current_size,
                          "ExpRep-total_samples"   : self.total_samples,
                          }
        return data

    def add_samples(self, data, prio, retrieve_samples=True):
        s, a, r, d = data
        vec_s, vis_s   = s[:2]
        n = prio.size
        idxs = self.add_indices(n)
        for i,vs in enumerate(self._vector_states):
            vs[idxs,:]   = vec_s[i]
        for i,vs in enumerate(self._visual_states):
            vs[idxs,:]   = vis_s[i]
        self._rewards[idxs,:]  = r
        self._dones[idxs,:]    = d
        if self.actions_not_list:
            self._actions[idxs,:]  = a
        else:
            for i,A in enumerate(self._actions):
                A[idxs,:]   = a[i]
        self.prios[idxs,:]    = prio
        #Counter for stats
        self.total_samples += n
        if retrieve_samples:
            return self.retrieve_samples_by_idx(idxs)

    def add_indices(self, n):
        if self.current_idx + n > self.max_size:
            self.current_size = self.current_idx
            self.current_idx = 0
        #This counter is basically our write-cursor
        self.current_idx += n
        #This is the border to the unused part of the exprep
        self.current_size = max(self.current_size, self.current_idx)
        return slice(self.current_idx-n, self.current_idx,1)

    def update_prios(self, new_prios, filter):
        self.prios[list(filter.keys()),:] = new_prios[list(filter.values()),:]

    def retrieve_samples_by_idx(self, indices):
        data = (
                ([vs[indices,:] for vs in self.vector_states],   [vs[indices,:] for vs in self.visual_states]),
                self.actions[indices,:] if self.actions_not_list else [a[indices,:] for a in self.actions],
                self.rewards[indices,:],
                self.dones[indices,:],
                )
        return data

    def __len__(self):
        return self.current_size
