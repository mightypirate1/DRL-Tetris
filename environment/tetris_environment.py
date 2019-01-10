import logging
import time
import numpy as np
# import aux
import aux.utils as utils
from aux.settings import default_settings
from environment.game_backend.modules import tetris_env
import environment.env_utils.state_processors as state_processors
import environment.env_utils.draw_tetris as draw_tetris
import environment.data_types as data_types

class tetris_environment:
    def __init__(self, id=None, settings=None, init_env=None):

        #Set up settings
        self.settings = utils.parse_settings(settings)
        settings_ok = self.process_settings() #Checks so that the settings are not conflicting
        assert settings_ok, "Settings are not ok! See previous error messages..."

        #Set up logging
        self.debug = self.settings["environment_logging"]
        if self.debug: self.log = logging.getLogger("environment")

        #Set up the environment
        self.id = id
        if init_env is None:
            pieces = self.generate_pieces()
            tetris_env.set_pieces(pieces)
            self.backend = tetris_env.PythonHandle(
                                                self.settings["n_players"],
                                                self.settings["game_size"]
                                              )
            #Upon agreement with backend, we always reset once.
            self.backend.reset()
        else:
            if type(init_env) is tetris_environment:
                self.backend = init_env.backend.copy()
            elif type(init_env) is tetris_env.PythonHandle:
                self.backend = init_env.copy()
            else:
                assert False, "Invalid init_env: type is {}".format(type(init_env))
        self.player_idxs = [p_idx for p_idx in range(self.settings["n_players"])]
        self.done = False

        #Say hi!
        if self.debug:
            self.log.info("Created tetris_environment!")
            self.log.info(self)

    # # # # #
    # Env interface fcns
    # # #
    def get_random_action(self, player=None):
        assert type(player)  is int,  "tetris_environment.get_random_action(player=int) was called with type(player)={}".format(type(player))
        actions = self.get_actions(player=player)
        idx = np.random.randint(low=0, high=len(actions))
        return actions[idx]

    def reset(self):
        self.backend.reset()

    def get_actions(self,player=None):
        if self.debug: self.log.debug("get_action invoked: player={}".format(player))
        assert type(player)  is int,  "tetris_environment.get_actions(int player) was called with type(player)={}".format(type(player))
        if self.settings["action_type"] is "place_block":
            self.backend.get_actions(player) #This ensures that the backend is in a sound state (i.e. did not currupt due to pickling. If you think you can fix pickling better, please contact me //mightypirate1)
            return data_types.action_list(self.backend.masks[player].action, remove_null=self.settings["bar_null_moves"])
        if self.settings["action_type"] is "press_key":
            return action_list([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]) #This is not error-tested (but "should work")

    def simulate_actions(self,actions, player=None):
        if self.debug: self.log.debug("simulate_actions invoked: actions={}, player={}".format(actions,player))
        assert type(player)  is int,                    "tetris_environment.simulate_actions(action_list actions,int player) was called with type(player)={}".format(type(player))
        assert type(actions) is data_types.action_list, "tetris_environment.simulate_actions(action_list actions,int player) was called with type(actions)={}".format(type(actions))
        ret = [None for _ in range(len(actions))]
        anchor = self.backend.copy()
        for i,a in enumerate(actions):
            self.backend.set(anchor)
            self.perform_action(a, player=player, render=False)
            ret[i] = self.get_state()
        self.backend.set(anchor)
        if self.settings["render_simulation"]:
            draw_tetris.drawAllFields([r.backend.states[0].field for r in ret[1:5]])
        return ret

    def perform_action(self, action, player=None, render=False):
        if self.debug: self.log.debug("executing action {} for player {}".format(action, player))
        assert type(player) is int,               "tetris_environment.perform_action(action a,int p) was called with type(player)={}".format(type(player))
        assert type(action) is data_types.action, "tetris_environment.perform_action(action a,int p) was called with type(action)={}".format(type(action))
        a         = [data_types.null_action for _ in self.player_idxs]
        a[player] = action
        self.done = self.backend.action(a,self.settings["time_elapsed_each_action"])
        reward = self.get_state()[player]["reward"]
        done   = self.done
        return reward, done

    def get_winner(self):
        if not self.done:
            return None
        for i in range(self.settings["n_players"]):
            if not self.backend.states[i].dead:
                return i
        if self.debug: self.log.warning("get_winner: env returned done=True, no one was alive to win game. I returned winner=666 in the hopes that this will be noticed...")
        return 666 #This should never happen.

    def simulate_all_actions(self, player):
        actions = self.get_actions(player=player)
        return self.simulate_actions(actions, player)

    def get_state(self):
        return data_types.state(self.backend, self.state_processor)

    # # # # #
    # Somewhat hacky fcns for env handling
    # # #
    def copy(self):
        return tetris_environment(settings=self.settings, init_env=self.backend)

    def set(self, e):
        if isinstance(e,tetris_env.PythonHandle):
            self.backend.set(e)
        elif isinstance(e,tetris_environment):
            self.backend.set(e.backend)
        elif isinstance(e,data_types.state):
            self.backend.set(e.backend_state)
        else:
            if self.debug: self.log.error("tetris_environment.set was called with an unrecognized argument!")

    # # # # #
    # Helper functions
    # # #
    def get_fields(self):
        # self.backend.recreate_state()
        return [self.backend.states[x].field for x in range(len(self.backend.states))]
    def render(self):
        if self.settings["render"]:
            self.renderer.drawAllFields(
                                        [self.get_fields()],
                                        force_rescale=False,
                                        pause_on_event=self.settings["pause_on_keypress"],
                                        )

    def generate_pieces(self):
        p = self.settings["pieces"]
        return (p*7)[:7]

    def process_settings(self):
        if type(self.settings["state_processor"]) is str:
            func, parameter_list = state_processors.func_dict[self.settings["state_processor"]]
            self.state_processor = state_processors.state_processor(func, [self.settings[x] for x in parameter_list])
        else:
            self.state_processor = state_processors.state_processor(self.settings["state_processor"])
        if self.settings["render"]:
            self.renderer = draw_tetris.get_global_renderer(resolution=self.settings["render_screen_dims"])

        assert self.settings["action_type"] in ["place_block", "press_key"]
        return True

    def __str__(self):
        ret = "tetris_environment settings:\n"
        length = max([len(x) for x in self.settings])
        for x in self.settings:
            ret += "\t{:{}}\t{}\n".format(x,length,self.settings[x])
        return ret

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'log' in d:
            d['log'] = d['log'].name
        if 'renderer' in d:
            del d['renderer']
        return d

    def __setstate__(self, d):
        if 'log' in d:
            d['log'] = logging.getLogger(d['log'])
            d['renderer'] = draw_tetris.get_global_renderer()
        self.__dict__.update(d)
