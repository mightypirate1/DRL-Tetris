import numpy as np
'''
A state_processor is a function that maps back-end states to front-end states.
The back-end state contains a lot of information, and it's structure is just
made to efficiently get information from the game-engine and out, so it is not
structured to be easily understood. Therefore, one may want to apply a
state_processor that turns it into something more understandable, or something
more compact. A state_processor can also present the state differently depending
on what player perspective is chosen. This is uselful if training with only one
policy playing against itself.

NOTE: a state_processor must take as argument: 1) the backend state
                                               2) the player-id
                                               3) a variable length param-list.

The parameter list will be demanded by registering the state_processor at the
bottom of this file.

This is just here for customizability, you can probably leave it as is!
'''

#x is type State as defined by the c++/python interface.
def state_dict(x, player, *parameters):
    col_code = { 1 : 5, 2 : 4, 3 : 1, 4 : 0, 5 : 2, 6 : 6, 7 : 3} #The engine codes pieces differently in different places....

    # piece_set = parameters[0][0] #For the compact representation
    if parameters[0][1]: #if settings["old_state_dict"]
        piece_set = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        piece_set = [0, 1, 2, 3, 4, 5, 6]

    ret =   {
                "field" : (np.array(x.states[player].field)>0).astype(np.uint8),
		        "piece" : np.array([int(p==col_code[x.states[player].piece.max()]) for p in piece_set]).astype(np.uint8),
		        "x" : np.array(x.states[player].x.copy(), dtype=np.uint8),
		        "y" : np.array(x.states[player].y.copy(), dtype=np.uint8),
		        "incoming_lines" : np.array(x.states[player].inc_lines),
		        "combo_time" : np.array( min(25000,x.states[player].combo_time+50)//100, dtype=np.uint8),
		        "combo_count" : np.array(x.states[player].combo_count, dtype=np.uint8),
		        "nextpiece" : np.array([int(p==x.states[player].nextpiece) for p in piece_set], dtype=np.uint8),
            }
    if parameters[0][2]: #settings["state_processor_separate_piece"]
        ret["piece_idx"] = col_code[x.states[player].piece.max()]
    if parameters[0][3]: #settings["augment_data"]
        swap_col_code = { 4 : 1, 3 : 0, 5 : 3, 7 : 2, 2 : 4, 1 : 5, 6 : 6,}
        piece_swap = [1,0,3,2,4,5,6]
        aug = {
                "field"     : ret["field"][:,::-1], #horizontal flip
                "piece"     : np.array([int(p==swap_col_code[x.states[player].piece.max()]) for p in piece_set]).astype(np.uint8),
                "nextpiece" : np.array([ piece_swap[int(p==x.states[player].nextpiece)] for p in piece_set], dtype=np.uint8),
                "piece_idx" : swap_col_code[x.states[player].piece.max().astype(np.uint8)],
              }
        ret["aug"] = aug
    return ret

def raw(x, player, *parameters):
    return x[player]

''' # # # # # # # # # # '''
''' # REGISTRY BELOW! # '''
''' # # # # # # # # # # '''

''' Register state_processor by entring it to the dictionary below '''
#Use this dictionary to register your state_processor so that it is accessible through its name.
'''Entries have the form: "<name>" : (<function_pointer>, [<s1>, <s2>, ... , <sn>]) '''
# where each <si> is a dictionary key to the settings dictionary.
# The <si> are used to request parameters from the environment...
# NOTE: The parameters are set up when the environment is created
func_dict = {
                "raw" : (raw, []),
                "state_dict": (state_dict, ["pieces", "old_state_dict", "state_processor_separate_piece", "augment_data"]),
            }
''' # # # # # # # # # # '''
''' # # # # # # # # # # '''
''' # # # # # # # # # # '''



''' Dont poke at this thing below unless you know what you are doing '''
class state_processor:
    def __init__(self,func,parameters=[]):
        self.func = func
        self.parameters = parameters
    def __call__(self,x,player):
        return self.func(x, player, self.parameters)
