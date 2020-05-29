import numpy as np
from scipy.special import softmax

import aux.utils as utils
import environment.data_types as edt

action_dim_exception = ValueError("action_[...] argument shape needs to be (r,t); usually (4,10).")

def make_q_action(rotation, translation):
    action = [8 for _ in range(rotation)] \
            + [2] + [3 for _ in range(translation)] \
            + [7]
    return edt.action(action)

def action_argmax(A):
    if len(A.shape) is not 2:
        raise action_dim_exception
    x = A.ravel()
    a_idx = np.argmax(x)
    (r, t) = np.unravel_index(a_idx, A.shape)
    return (r, t), 0

def action_epsilongreedy(A, epsilon):
    if len(A.shape) is not 2:
        raise action_dim_exception
    if np.random.rand() < epsilon:
        r = np.random.choice(np.arange(A.shape[0]))
        t = np.random.choice(np.arange(A.shape[1]))
    else:
        (r,t), _ = action_argmax(A)
    e = min(1,epsilon)
    _entropy = e*np.full(A.size, 1/A.size)
    _entropy[0] += (1-e)
    entropy = utils.entropy(_entropy)
    return (r,t), entropy

def action_pareto(A,theta):
    if len(A.shape) is not 2:
        raise action_dim_exception
    x = A.ravel()
    p = utils.pareto(x, temperature=theta)
    a_idx = np.random.choice(np.arange(A.size),p=p)
    (r,t) = np.unravel_index(a_idx, A.shape)
    entropy = utils.entropy(p)
    return (r,t), entropy

def action_boltzman(A,theta):
    if len(A.shape) is not 2:
        raise action_dim_exception
    x = A.ravel()
    p = softmax(theta*x)
    np.random.choice(np.arange(A.size),p=p)
    (r,t) = np.unravel_index(amax, A.shape)
    entropy = utils.entropy(p)
    return (r,t), entropy

def action_distribution(A):
    # print(A);input()
    if len(A.shape) is not 2:
        raise action_dim_exception
    p = A.ravel()
    a_idx = np.random.choice(np.arange(A.size),p=p)
    (r,t) = np.unravel_index(a_idx, A.shape)
    entropy = utils.entropy(p+10**-6)
    return (r,t), entropy

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
