import threads
from agents.agent_utils import state_unpack

class template_agent_base:
    def __init__(
                  self,
                  id=0,
                  name="base_type!",
                  session=None,
                  sandbox=None,
                  settings=None,
                  mode=threads.STANDALONE
                 ):

        #Some basic core functionality (leaving it here so there's an example of how to use the unpacker-util)
        self.sandbox = sandbox.copy()
        self.unpack = state_unpack.unpacker(
                                            self.sandbox.get_state(),
                                            observation_mode='separate' if self.settings["field_as_image"] else 'vector',
                                            player_mode='separate' if self.settings["players_separate"] else 'vector',
                                            state_from_perspective=self.settings["relative_state"],
                                            separate_piece=False,
                                            ) #this hides all the vectorization in and out of the vectorized environment.
        '''
        usage:
         (vis_p1, vis_p2), (vec_p1, vec_p2), [(piece)]= self.unpack([s0,s1,...,sn], player=[p0, p1,...,pn]) #sequence of states, and the player who's perspective you want to see it from
         vis_x and vec_x are the visual- and vector-components of the state.
         vis_x is an np.array of shape [n,W,H] where W,H are the dimentions of the play field
         vec_x is an np.array of shape [n,K] iirc K==30, but you can check that....
         p1 is the player who's perspective you view the state from, and p2 is the opponent
        '''

    def update_clock(self, clock):
        ''' if you want a clock that synchs with the rest '''
        #self.clock = clock
        pass

    def save_weights(self, folder, file, verbose=False): #folder is a sub-string of file!  e.g. folder="path/to/folder", file="path/to/folder/file"
        ''' useful but not mandatory '''
        pass

    def load_weights(self, folder, file):
        ''' useful but not mandatory '''
        pass

    def update_weights(self, weight_list): #As passed by the trainer's export_weights-fcn..
        ''' when the trainer sends you weights '''
        pass
