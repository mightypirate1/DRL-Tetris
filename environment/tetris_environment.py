import logging
import time
import numpy as np
from environment.game_backend.modules import tetris_env
import environment.env_utils.state_processors as state_processors
from environment.data_types.action_list import action_list
from environment.data_types.state import state
import aux

class tetris_environment:
    def __init__(self, id=None, settings=None, init_env=None):
        #Set up logging
        self.log = logging.getLogger("environment")
        #Set up settings
        self.settings = aux.settings.default_settings.copy()
        self.id = id
        if settings is not None:
            for x in settings:
                self.settings[x] = settings[x]
        settings_ok = self.process_settings() #Checks so that the settings are not conflicting
        assert settings_ok, "Settings are not ok! See previous error messages..."
        #Set up the environment
        if init_env is None:
            pieces = self.generate_pieces()
            tetris_env.set_pieces(pieces)
            self.backend = tetris_env.PythonHandle(
                                                self.settings["n_players"],
                                                self.settings["game_size"]
                                              )
            self.backend.reset()
        else:
            self.backend = init_env.copy()
        self.done = False

        #Say hi!
        self.log.info("Created tetris_environment!")
        self.log.info(self)

    # # # # #
    # Env interface fcns
    # # #
    def reset(self):
        self.backend.reset()

    def get_actions(self,player=None):
        self.log.debug("get_action invoked: player={}".format(player))
        if player not in range(self.settings["n_players"]) and player is not None:
            self.log.warning("get_actions called with player={}. This may be fatal. Expected 0<=player<{}.".format(player,self.settings["n_players"]))
            return None
        if self.settings["action_type"] is "press_key":
            available_actions = [0,1,2,3,4,5,6,7,8,9,10]
            return available_actions if player is not None else [available_actions]*self.settings["n_players"]
        elif self.settings["action_type"] is "place_block":
            if player is None:
                for p in range(self.settings["n_players"]):
                    self.backend.get_actions(p)
                return [action_list(self.backend.masks[p].action, remove_null=self.settings["bar_null_moves"]) for p in range(self.settings["n_players"])]
            else:
                self.backend.get_actions(player)
                return action_list(self.backend.masks[player].action, remove_null=self.settings["bar_null_moves"])
        else:
            self.log.warning("get_actions called with action_type={}. This may be fatal. Expected action_type \"press_key\" or \"place_block\"".format(self.settings["action_type"]))
            return None

    def get_random_action(self, player=None):
        if player is None: p_list = [p for p in range(self.settings["n_players"])]
        else : p_list = player
        if type(p_list) is not list: p_list = [p_list]
        ret = []
        actions = self.get_actions()
        for i in range(self.settings["n_players"]):
            if i in p_list:
                idx = np.random.choice(np.arange(len(actions[i])))
                ret.append(actions[i][idx])
            else:
                ret.append([0])
        return ret

    def simulate_actions(self,actions, player=None):
        self.log.debug("simulate_actions invoked: actions={}, player={}".format(actions,player))
        ret = []
        anchor = self.backend.copy()
        for a in actions:
            self.backend.set(anchor)
            self.perform_action(a, player=player, render=False)
            ret.append(state(self.backend, self.state_processor))
        self.backend.set(anchor)
        if self.settings["render_simulation"]:
            draw_tetris.drawAllFields([r.backend.states[0].field for r in ret[1:5]])
        return ret

    def perform_action(self, action, player=None, render=None):
        self.log.debug("executing action {} for player {}".format(action, player))
        # print(action,player,render);exit()
        if player is None:
            a = action
        else:
            a = [[0]]*self.settings["n_players"]
            a[player] = action
        self.done = self.backend.action(a,self.settings["time_elapsed_each_action"])
        # print(a, type(a));exit()
        if render:
            self.render()
        return self.done

    def get_winner(self):
        if not self.done:
            return None
        for i in range(self.settings["n_players"]):
            if not self.backend.states[i].dead:
                return i
        self.log.warning("get_winner: env returned done=True, no one was alive to win game. I returned winner=666 in the hopes that this will be noticed...")
        return 666 #This should never happen.

    def simulate_all_actions(self, player):
        actions = self.get_actions(player=player)
        return self.simulate_actions(actions, player)

    def get_state(self, player=None):
        return state(self.backend, self.state_processor)

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
        elif isinstance(e,state):
            self.backend.set(e.backend_state)
        else:
            self.log.error("tetris_environment.set was called with an unrecognized argument!")

    # # # # #
    # Helper functions
    # # #
    def render(self):
        if not self.settings["render"]:
            return
        self.draw_tetris.drawAllFields([self.backend.states[x].field for x in range(len(self.backend.states))])
        #Pausing capability
        if self.draw_tetris.pollEvents():
            print("----------------------")
            print("--------PAUSED--------")
            print("----------------------")
            while not self.draw_tetris.pollEvents():
                time.sleep(1.0)

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
            import environment.env_utils.draw_tetris as draw_tetris
            self.draw_tetris = draw_tetris
        return True

    def __str__(self):
        ret = "tetris_environment settings:\n"
        length = max([len(x) for x in self.settings])
        for x in self.settings:
            ret += "\t{:{}}\t{}\n".format(x,length,self.settings[x])
        return ret
