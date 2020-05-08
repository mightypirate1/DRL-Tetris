import threads
from agents.agent_utils import state_unpack
''' import your base-class '''

class template_agent(template_agent_base):
    def __init__(
        def __init__(
                     self,
                     n_envs,                    # How many envs in the vector-env?
                     n_workers=1,               # How many workers run in parallel? If you don't know, guess it's just 1
                     id=0,                      # What's this trainers name?
                     session=None,              # The session to operate in
                     sandbox=None,              # Sandbox to play in!
                     mode=threads.STANDALONE,   # What's our role?
                     settings=None,             # Settings-dict passed down from the ancestors
                     init_weights=None,
                     init_clock=0,
                    ):

        #Some general variable initialization etc...
        super().__init__(self, id=id, name="worker{}".format(id), session=session, sandbox=sandbox, settings=settings, mode=mode)

        def get_action(self, state_vec, player=None, random_action=False, training=False, verbose=False):
            ''' you get a vector of states, you provide a vector of actions (actions) and how you want your action recorded (this will be the content in the experience-tuple) '''
            return #action_idxs, actions

        def ready_for_new_round(self, training=False, env=None):
            ''' when round ends, this function is called '''
            pass

        def store_experience(self, experience, env=None):
            '''
            you get experiences sent to you.
            they are on the form (s,a,r,s',done,p)
            a is the action_idx from get_action
            p is the playr performing it
            all elements in the experince tuple is a list, since the environmnt is vectorized
            '''

            '''
            PRO-TIP:
            agents.datatypes.trajectory contains a trajectory-type that can be useful to you. inherit off of it to maintain other agent's functionality intact!
            '''
            pass

        def transfer_data(selfe):
            '''
            this is how the worker sends data. trainers receive data as it's sent here
            '''
            return #data
