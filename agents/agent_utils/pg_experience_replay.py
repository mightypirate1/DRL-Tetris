import numpy as np
import scipy.stats
import aux.utils as utils
from .. import agent_utils

class pg_experience_replay:
    def __init__(self, max_size=None, n_actions=None, state_size=None, log=None):
        self.log        = log
        self.max_size   = max_size
        self.n_actions = n_actions
        self.state_size = state_size
        self.states         = np.zeros((max_size, n_actions,*state_size))
        self.states_mask    = np.zeros((max_size, n_actions))
        self.actions        = np.zeros((max_size,1))
        self.old_probs      = np.zeros((max_size,1))
        self.target_values  = np.zeros((max_size,1))
        self.advantages     = np.zeros((max_size,1))
        self.current_size  = 0
        self.current_idx   = 0
        self.total_samples = 0

    def get_random_sample(self, n_samples, alpha=1.0, beta=1.0, remove=False):
        #Create the sampling distribution (see paper for details)
        n = self.current_size
        #Sample indices to make a batch out of!
        idx_dict = {}
        indices = np.random.choice(all_indices, replace=True, size=n_samples).tolist()

        #Return values!
        data = (
                self.states[indices,:],
                self.states_mask[indices,:],
                self.actions[indices,:],
                self.old_probs[indices,:],
                self.target_values[indices,:],
                self.advantages[indices,:],
                )
        return data

    def get_all_samples(self):
        return (self.states, self.states_mask, self.actions, self.old_probs, self.target_values, self.advantages)

    def add_samples(self, data):
        s,m,a,old_probs,target_values,advantages = data
        n = a.shape[0]
        idxs = [x%self.max_size for x in range(self.current_idx, self.current_idx+n)]
        self.states[idxs,:]   = s
        self.states_mask[idxs,:]   = m
        self.actions[idxs,:]   = a
        self.old_probs[idxs,:]  = old_probs
        self.target_values[idxs,:]  = target_values
        self.advantages[idxs,:]  = advantages
        self.current_idx += n
        self.current_size = min(self.current_size+n, self.max_size)
        self.total_samples += n

    def clear(self):
        self.__init__(
                      n_actions=self.n_actions,
                      max_size=self.max_size,
                      state_size=self.state_size,
                      log=self.log
                     )

    def __len__(self):
        return self.current_size
