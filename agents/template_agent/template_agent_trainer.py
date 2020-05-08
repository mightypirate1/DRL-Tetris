''' import the baseclass-file (not done here since you are going to move it probably) '''

class template_agent_trainer(template_agent_base):
    def __init__(
                 self,
                 id=0,                      # What's this trainers name?
                 session=None,              # The session to operate in
                 sandbox=None,              # Sandbox to play in!
                 mode=threads.ACTIVE,       # What's our role?
                 settings=None,             # Settings-dict passed down from the ancestors
                 init_weights=None,
                 init_clock=0,
                ):

        #Some general variable initialization etc...
        super().__init__(self, id=id, name="trainer{}".format(id), session=session, sandbox=sandbox, settings=settings, mode=mode)
        '''
        set yourself up!

        you probably need like a neural-net and stuff
        '''
        pass

    #What if someone just sends us some experiences?! :D
    def receive_data(self, data_list):
        ''' you need to be able to get data, you get it on the format as the workers sends it in '''
        return #n_samples_recieved, avg_length_of_data

    def do_training(self, policy=None):
        ''' do your training (policy=None means you have only 1 policy in your agent, this is default. In 2-policy mode, policy is 0 or 1)'''
        return

    def export_weights(self):
        ''' export weights so that workers can get them '''
        return #weight_index, weights
