import logging
import random
from agents.tetris_agent import tetris_agent

default_settings = {
                    "option" : "entry",
                    "time_to_training" : 100,
                    }

class random_agent(tetris_agent):
    def __init__(self, id=0, session=None, sandbox=None, training=False, settings=None):
        self.log = logging.getLogger("agent")
        self.log.debug("Test agent created!")
        self.id = id
        self.sandbox = sandbox
        self.training = training
        self.settings = default_settings.copy()
        self.session = session
        if settings is not None:
            for x in settings:
                self.settings[x] = settings[x]
        settings_ok = self.process_settings() #Checks so that the settings are not conflicting
        assert settings_ok, "Settings are not ok! See previous error messages..."

        self.time_to_training = self.settings['time_to_training']
        self.current_trajectory = []

    # # # # #
    # Agent interface fcns
    # # #
    def get_action(self, state, training=True):
        self.sandbox.set(state)
        a_list = self.sandbox.get_actions(player=self.id)
        a_idx = random.randrange(len(a_list))
        return a_idx, a_list[a_idx]

    def get_evaluation(self, state, training=True):
        pass

    def store_experience(self, experience):
        print("agent[{}] stored experience {}".format(self.id, experience))
        self.current_trajectory.append(experience)
        self.time_to_training -= 1

    def process_trajectory(self):
        print("agent[{}] processed a trajectory".format(self.id))
        self.current_trajectory.clear()

    def is_ready_for_training(self):
        return (self.time_to_training < 0)

    def do_training(self):
        self.time_to_training = self.settings["time_to_training"]
        print("agent[{}] doing training".format(self.id))

    def init_training(self):
        pass

    # # # # #
    # Training fcns
    # # #
    def get_next_states(self, state):
        self.sandbox.set(state.backend_state)
        self.sandbox.simulate_all_actions(self.id)

    # # # # #
    # Memory management fcns
    # # #
    def save_weights(self):
        pass

    def load_weights(self):
        pass

    def save_memory(self):
        pass

    def load_memory(self):
        pass
    # # # # #
    # Init fcns
    # # #
    def create_model(self):
        pass

    def create_training_ops(self):
        pass

    def create_saver(self):
        pass
    # # # # #
    # Helper fcns
    # # #

    def process_settings(self):
        print("process_settings not implemented yet!")
        return True
