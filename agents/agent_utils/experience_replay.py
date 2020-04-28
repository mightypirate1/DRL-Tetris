import numpy as np
import scipy.stats
from .. import agent_utils

class experience_replay:
    def __init__(self, max_size=None, state_size=None, experience_type="single_experience_tuple",log=None, sample_mode='rank' ,forget_mode='oldest'):
        self.log        = log
        self.max_size   = max_size
        self.vector_state_size, self.visual_state_size = state_size
        self.vector_states   = [np.zeros((max_size,*s[1:]), dtype=np.uint8) for s in self.vector_state_size]
        self.visual_states   = [np.zeros((max_size,*s[1:]), dtype=np.uint8) for s in self.visual_state_size]
        self.vector_s_primes = [np.zeros((max_size,*s[1:]), dtype=np.uint8) for s in self.vector_state_size]
        self.visual_s_primes = [np.zeros((max_size,*s[1:]), dtype=np.uint8) for s in self.visual_state_size]
        self.dones    = np.zeros((max_size,1), dtype=np.uint8)
        self.rewards  = np.zeros((max_size,1), dtype=np.float32)
        self.prios    = -np.ones((max_size,1), dtype=np.float32)
        if experience_type == "trajectory":
            self.trajectory = np.full((max_size,), None)
            self.trajectory_prios = -np.ones((max_size,1), dtype=np.float32)
            self.current_n_trajectories = 0
        self.current_size  = 0
        self.current_idx   = 0
        self.total_samples = 0
        self.sample_mode = sample_mode
        self.forget_mode = forget_mode
        self.experience_type = experience_type
        self.resort_fraction = 0.5

    def get_random_sample(self, n_samples, alpha=1.0, beta=1.0, remove=False, compute_stats=False):
        if self.experience_type == "single_experience_tuple":
            sample_prios = self.prios
            n = self.current_size
        elif self.experience_type == "trajectory":
            sample_prios = self.trajectory_prios
            n = self.current_n_trajectories
        all_indices = np.arange(n)

        #Create the sampling distribution (see paper for details)
        if self.sample_mode == 'rank':
            #make ranking
            rank = 1+n-scipy.stats.rankdata(sample_prios[:n].ravel(), method='ordinal')
            #make a ranking-based probability disribution (pareto-ish)
            one_over_rank = 1/rank #Rank-based sampling
            p_unnormalized = one_over_rank**alpha
        else:
            p_unnormalized = (self.prios + 0.0001)**alpha

        p = p_unnormalized / p_unnormalized.sum() #sampling distribution done
        is_weights_unnormalized = ((n*p)**(-beta))[:,np.newaxis] #Compute the importance sampling weights
        is_weights_all = is_weights_unnormalized/is_weights_unnormalized.reshape(-1).max()

        #Sample indices to make a batch out of!
        idx_dict = {}
        indices = np.random.choice(all_indices, replace=True, size=n_samples, p=p).tolist()

        ##Data collection, and index-tracking
        i = 0
        for idx in indices:
            if self.experience_type == "single_experience_tuple":
                idx_dict[idx] = i #This makes sure that idx_dict keeps track of the position of the last occurance of each index in the sample
                i += 1
            elif self.experience_type == "trajectory":
                idx_dict[idx] = slice(i, i+len(self.trajectory[idx])) #This makes sure that idx_dict keeps track of the position of the last occurance of each index in the sample
                i += len(self.trajectory[idx])

        #Return values!
        filter = idx_dict
        is_weights = is_weights_all[indices]

        if self.experience_type == "single_experience_tuple":
            data = self.retrieve_samples_by_idx(indices)
            trajectory_idxs = None
        elif experience_type == "trajectory":
            data, trajectory_idxs = self.retrieve_trajectories_by_idx(indices)

        #Stats?
        if compute_stats:
            iwu = is_weights_unnormalized[:self.current_size].ravel()
            stats = {
                     "ExpRep-iwu_max"  : iwu.max(),
                     "ExpRep-iwu_mean" : iwu.mean(),
                     "ExpRep-iwu_min"  : iwu.min(),
                     "ExpRep-prio_max"  : sample_prios[:n].max(),
                     "ExpRep-prio_mean" : sample_prios[:n].mean(),
                     "ExpRep-prio_min"  : sample_prios[:n].min(),
                     "ExpRep-sample_size"     : self.current_size,
                     "ExpRep-trajectory_size" : self.current_n_trajectories,
                    }
        else:
            stats = {}
        return data, trajectory_idxs, is_weights, filter, stats

    def add_samples(self, data, prio):
        s, sp,_,r,d = data
        vec_s, vis_s   = s
        vec_sp, vis_sp = sp
        n = prio.size
        idxs = self.add_indices(n)
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
        if self.forget_mode != 'oldest':
            self.resort()

    def add_indices(self, n):
        idxs = [x%self.max_size for x in range(self.current_idx, self.current_idx+n)]
        if self.forget_mode != 'oldest':
            idxs = self.forget_order[idxs]
        return idxs

    def resort(self):
        if self.forget_mode == 'lowest_prio':
            if self.current_idx < self.max_size * self.resort_fraction:
                return
            ### If we have added many new samples, we re-sort them to make sure we over-write low-prio samples!
            self.forget_order = np.argsort(self.prios.ravel())
            self.current_idx = 0

    def resort(self):
        if self.forget_mode == 'highest_prio':
            if self.current_idx < self.max_size * self.resort_fraction:
                return
            ### If we have added many new samples, we re-sort them to make sure we over-write low-prio samples!
            self.forget_order = np.argsort(self.prios.ravel())[::-1]
            self.current_idx = 0

        elif self.forget_mode == 'uniform_prio':
            if self.current_idx > self.max_size:
                self.forget_order = np.random.permutation(self.max_size)
                self.current_idx = 0

    def update_prios(self, new_prios, filter):
        if self.experience_type == "single_experience_tuple":
            self.prios[list(filter.keys()),:] = new_prios[list(filter.values()),:]
        elif self.experience_type == "trajectory":
            for t_idx, f_idx in zip(filter.keys(), filter.vals()):
                self.prios[ self.trajectory[t_idx], :] = new_prios[f_idx,:]
                self.trajectory_prios[t_idx,:] = np.mean(new_prios[f_idx,:].ravel())

    def retrieve_samples_by_idx(self, indices):
        data = (
                ([vs[indices,:] for vs in self.vector_states],   [vs[indices,:] for vs in self.visual_states]),
                ([vs[indices,:] for vs in self.vector_s_primes], [vs[indices,:] for vs in self.visual_s_primes]),
                None,
                self.rewards[indices,:],
                self.dones[indices,:],
                )
        return data

    def retrieve_trajectories_by_idx(self, idxs):
        t_idx_list = self.trajectory[indices]
        data = (
                #s
                (
                [np.concatenate([vs[i,:] for i in t_idx],axis=0) for vs in self.vector_states],
                [np.concatenate([vs[i,:] for i in t_idx],axis=0) for vs in self.visual_states]
                ),
                #s'
                (
                [np.concatenate([vs[i,:] for i in t_idx],axis=0) for vs in self.vector_s_primes],
                [np.concatenate([vs[i,:] for i in t_idx],axis=0) for vs in self.visual_s_primes]
                ),
                #a
                None,
                #r
                np.concatenate([self.rewards[i,:] for i in t_idx ], axis=0),
                #d
                np.concatenate([self.dones[i,:] for i in t_idx ], axis=0)
               )
        t_idx, start = [None for _ in t_idx_list], 0
        for i, t in t_idx_list:
            t_idx[i] = slice(start, start+len(t))
            start += len(t)
        return data, t_idx

    def __len__(self):
        return self.current_size
