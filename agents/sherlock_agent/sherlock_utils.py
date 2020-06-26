import numpy as np
from scipy.special import softmax

import tools.utils as utils
import environment.data_types as edt

action_dim_exception = ValueError("action_[...] argument shape needs to be (r,t); usually (4,10).")

def generate_deltas(states, players, sandbox):
    deltas_all = pad_and_concat([ deltas(s,p,sandbox)[np.newaxis,:] for s,p in zip(states, players)])
    delta_sums = np.sum(deltas_all, axis=-1, keepdims=True)
    return deltas_all, delta_sums
def deltas(state, player, env):
    results = env.simulate_all_actions(state, player=player, finalize=False)
    def d(result):
        ret = result[player]["field"] - state[player]["field"]
        if ret.sum() < 4.0:
            ret = np.full_like(ret, 1e-3)
        return ret
    return np.concatenate([ d(result)[:,:,np.newaxis] for result in results ], axis=2)
def pad_to_shape(x, shape):
    ret = np.zeros(shape)
    a,b,c,d = x.shape
    ret[:a,:b,:c,:d] = x
    return ret
def pad_and_concat(deltas):
    w_max = 0
    for d in deltas:
        a,b,c,w = d.shape
        w_max = max(w,w_max)
    ret = np.concatenate([pad_to_shape(d, (a,b,c,w_max)) for d in deltas], axis=0)
    return ret

def value_piece(eval, piece):
    if len(eval.shape) == 0:
        return eval
    pos = len(eval.shape) -1
    if eval.shape[pos] > 1:
        assert eval.size == eval.shape[pos]
        return eval.squeeze()[piece]
    return eval.squeeze()

def value_mean(eval):
    return np.mean(eval)

def action_distribution(p):
    if len(p.shape) is not 1:
        raise action_dim_exception
    if np.isnan(p).any():
        print("WARNING: NaN encountered in action_distribution!")
        return np.array(0), 0.0
    a_idx = np.random.choice(np.arange(p.size),p=p)
    entropy = utils.entropy(p)
    return a_idx, entropy

def action_argmax(p):
    if len(p.shape) is not 1:
        raise action_dim_exception
    if np.isnan(p).any():
        return np.array(0), 0.0
    a_idx = np.argmax(p)
    entropy = 0.0
    return a_idx, entropy
