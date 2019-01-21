import numpy as np
import aux.utils as utils

# # # # #
# Helper fcns to get make a state-type object (environment.data_types.state)
# using the state-dict statemaker (default) into numpy-arrays.
#
# Two different flavours exist: states_from_perspective puts the specified
# players state first in the vector. states_to_vectors lists the states in the
# backend order, so player 0 first, then 1 etc.

# # #
def states_from_perspective( states, player):
    p_list = utils.parse_arg(player, [0, 1])
    return states_to_vectors(states, player_lists=[[p, 1-p] for p in p_list])

def state_from_perspective( state, player):
    return state_to_vector(state, [player, 1-player])

def state_to_vector( state, player_list=None):
    if isinstance(state,np.ndarray):
        assert False, "This should not happen"
        return state
    def collect_data(state_dict):
        tmp = []
        for x in state_dict:
            tmp.append(state_dict[x].reshape((1,-1)))
        return np.concatenate(tmp, axis=1)
    if player_list is None: #If None is specified, we assume we want the state of all players!
        player_list = [0, 1] #only 2player mode as of yet!
    data = [collect_data(state[player]) for player in player_list]
    ret = np.concatenate(data, axis=1)
    return ret

def states_to_vectors( states, player_lists=None):
    assert len(player_lists) == len(states), "player_list:{}, states:{}".format(player_lists, states)
    if type(states) is list:
        ret = [state_to_vector(s, player_list=player_lists[i]) for i,s in enumerate(states)]
    else:
        #Assume it is a single state if it is not a list
        ret = [state_to_vector(states, player_list=player_lists)]
    return np.concatenate(ret, axis=0)
