import numpy as np
import tools.utils as utils

class unpacker:
    def __init__(
                 self,
                 state,
                 state_from_perspective=True,
                 observation_mode='vector', #'vector' / 'separate'
                 player_mode='vector', #'vector' / 'separate'
                 # piece_mode='1hot',
                 separate_piece=False,
                 piece_in_statevec=False,
                 ):
        self.state_fcn = self.states_from_perspective if state_from_perspective else self.states_to_vectors
        self.collect_data = self.collect_all_data if observation_mode == 'vector' else self.collect_separate_data
        self.joiner = self.concat if player_mode == 'vector' else self.join_separate
        self.state_from_perspective = state_from_perspective
        self.observation_mode = observation_mode
        self.player_mode = player_mode
        # self.piece_mode = piece_mode
        self.separate_piece = separate_piece
        self.piece_in_statevec = piece_in_statevec
        self.state_size = self.get_shapes(state=state)

    ##Frontend
    def get_shapes(self, state=None):
        if state is None: return self.state_size
        if not self.separate_piece:
            vec, vis = self([state], [0])
        else:
            vec, vis, piece = self([state], [0])
        return [v.shape for v in vec], [v.shape for v in vis]

    def states_from_perspective(self, states, player, mirrored=False):
        p_list = utils.parse_arg(player, [0, 1])
        return self.states_to_vectors(states, player_lists=[[p, 1-p] for p in p_list], mirrored=mirrored)
    def state_from_perspective(self, state, player, mirrored=False):
        return self.state_to_vector(state, [player, 1-player])

    def state_to_vector(self, state, player_list=None, mirrored=False):
        if isinstance(state,np.ndarray):
            assert False, "This should not happen"
            return state
        if player_list is None: #If None is specified, we assume we want the state of all players!
            player_list = [0, 1] #only 2player mode as of yet!
        data = [self.collect_data(state[player], mirrored=mirrored) for player in player_list]
        ret = self.joiner(data, player_list)
        return ret

    def states_to_vectors(self, states, player_lists=None, mirrored=False):
        assert len(player_lists) == len(states), "player_list:{}, states:{}".format(player_lists, states)
        if type(states) is list:
            ret = self.pack([self.state_to_vector(s, player_list=player_lists[i], mirrored=mirrored) for i,s in enumerate(states)], player_lists)
        else:
            #Assume it is a single state if it is not a list
            ret = self.pack([self.state_to_vector(states, player_list=player_lists, mirrored=mirrored)], player_lists)
        return ret

    ##Backend
    def concat(self, data, player_list, axis=-1):
        _vector, _visual = zip(*data)
        vector = np.concatenate(_vector, axis=axis) if _vector[0] is not None else None
        visual = np.concatenate(_visual, axis=axis) if _visual[0] is not None else None
        return vector, visual

    def join_separate(self, data, player_list):
        return zip(*data)

    def collect_all_data(self,state_dict, mirrored=False):
        tmp = []
        if mirrored:
            aug = state_dict.pop("aug", None)
            state_dict.update(aug)
        for x in state_dict:
            if x == "aug":
                continue
            if x in ['piece', 'piece_idx'] and self.separate_piece:
                piece = np.array(state_dict['piece_idx']).reshape((1))
                if not self.piece_in_statevec or x in ['piece_idx']:
                    continue
            tmp.append(state_dict[x].reshape((1,-1)))
        vector = np.concatenate(tmp, axis=1)
        visual = None
        ret = vector, visual if not self.separate_piece else vector, visual, piece
        return ret

    def collect_separate_data(self, state_dict, mirrored=False):
        tmp = []
        if mirrored:
            aug = state_dict.pop("aug", None)
            state_dict.update(aug)
        for x in state_dict:
            if x == "aug":
                continue
            if x in ['piece', 'piece_idx'] and self.separate_piece:
                piece = np.array(state_dict['piece_idx']).reshape((1))
                if not self.piece_in_statevec or x in ['piece_idx']:
                    continue
            elif x != 'field':
                tmp.append(state_dict[x].reshape((1,-1)))
        vector = np.concatenate(tmp, axis=1)
        visual = state_dict['field'][None,:,:,None]
        ret = (vector, visual) if not self.separate_piece else (vector, visual, piece)
        return ret

    def pack(self, data_list, player_lists):
        if not self.separate_piece:
            _vector, _visual = zip(*data_list)
        else:
            _vector, _visual, _piece = zip(*data_list)
        if self.player_mode == 'vector':
            if _visual[0] is None:
                visual = []
            else:
                visual = [np.concatenate(_visual, axis=0)]
            if _vector[0] is None:
                vector = []
            else:
                vector = [np.concatenate(_vector, axis=0)]

        else:
            if _vector[0] is None:
                vector = []
            else:
                vector = [np.concatenate([v[p] for v in _vector], axis=0) for p in range(2)]
            if _visual[0] is None:
                visual = []
            else:
                visual = [np.concatenate([v[p] for v in _visual], axis=0) for p in range(2)]

        if self.separate_piece:
            piece = [np.concatenate([piece[p] for piece in _piece], axis=0) for p in range(2)]
            ret = vector, visual, piece
        else:
            ret = vector, visual
        return ret

    def __call__(self, states, player=None, mirrored=False):
        return self.state_fcn(states,player, mirrored=mirrored)
