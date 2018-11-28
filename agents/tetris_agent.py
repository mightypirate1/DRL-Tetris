class tetris_agent:
    training = False
    id = None
    def __init__(self,a,s=None):
        print("I am a base-class!")

    def get_action(self, state, training=True):
        pass

    def get_evaluation(self, state, training=True):
        pass

    def store_experience(self, experience):
        pass

    def process_trajectory(self):
        pass

    def is_ready_for_training(self):
        pass

    def do_training(self):
        pass
