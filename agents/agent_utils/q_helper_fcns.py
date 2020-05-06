import numpy as np
import environment.data_types as edt

def make_q_action(rotation, translation):
    action = [8 for _ in range(rotation)] \
            + [2] + [3 for _ in range(translation)] \
            + [6]
    return edt.action(action)

def action_argmax(A):
    if len(A.shape) is not 2:
        raise ValueError("action_argmax argument shape needs to be (?,r,t) or (r,t). recieved {}".format(A.shape))
    x = A.ravel()
    amax = np.argmax(x)
    amax2d = np.unravel_index(amax, A.shape)
    return amax2d
