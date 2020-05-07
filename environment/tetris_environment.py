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
        self.player_idxs = [p_idx for p_idx in range(self.settings["n_players"])]

        #Set up logging
        self.debug = self.settings["environment_logging"]
        if self.debug: self.log = logging.getLogger("environment")

        ## Stats
        self.rounds_played = None
        self.tot_reward = None
        self.round_reward = None
        self.reward = [None, None]
        self.last_reward = [None, None]

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
            self.reset()
        else:
            if type(init_env) is tetris_environment:
                self.backend = init_env.backend.copy()
            elif type(init_env) is tetris_env.PythonHandle:
                self.backend = init_env.copy()
            else:
                assert False, "Invalid init_env: type is {}".format(type(init_env))

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
        self.done         = False
        self.round_reward = [self.reward_fcn(p_idx) for p_idx in self.player_idxs]
        self.reward       = [self.reward_fcn(p_idx) for p_idx in self.player_idxs]
        if self.tot_reward is None:
            self.tot_reward = self.reward
        if self.rounds_played is None:
            self.rounds_played = 0
        self.rounds_played += 1

    def get_actions(self, state, player=None):
        if self.debug: self.log.debug("get_action invoked: player={}".format(player))
        assert type(player)  is int,  "tetris_environment.get_actions(int player) was called with type(player)={}".format(type(player))
        self.set(state)
        if self.settings["action_type"] == "place_block":
            self.backend.get_actions(player) #This ensures that the backend is in a sound state (i.e. did not currupt due to pickling. If you think you can fix pickling better, please contact me //mightypirate1)
            return data_types.action_list(self.backend.masks[player].action, remove_null=self.settings["bar_null_moves"])
        if self.settings["action_type"] == "press_key":
            return data_types.action_list([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]) #This is not error-tested (but "should work")

    def simulate_actions(self,actions, player=None):
        if self.debug: self.log.debug("simulate_actions invoked: actions={}, player={}".format(actions,player))
        assert type(player)  is int,                    "tetris_environment.simulate_actions(action_list actions,int player) was called with type(player)={}".format(type(player))
        assert type(actions) is data_types.action_list, "tetris_environment.simulate_actions(action_list actions,int player) was called with type(actions)={}".format(type(actions))
        ret = [None for _ in range(len(actions))]
        anchor = self.backend.copy()
        for i,a in enumerate(actions):
            self.backend.set(anchor)
            self.perform_action(a, player=player, simulate=True)
            ret[i] = self.get_state()
        self.backend.set(anchor)
        if self.settings["render_simulation"]:
            self.renderer.drawAllFields([[r.backend_state.states[player].field for r in ret]],force_rescale=True,)
        return ret

    def perform_action(self, action, player=None, simulate=False):
        if self.debug: self.log.debug("executing action {} for player {}".format(action, player))
        assert type(player) is int,               "tetris_environment.perform_action(action a,int p) was called with type(player)={}".format(type(player))
        assert type(action) is data_types.action, "tetris_environment.perform_action(action a,int p) was called with type(action)={}".format(type(action))
        a         = [data_types.null_action for _ in self.player_idxs]
        a[player] = action
        self.done = self.backend.action(a,self.settings["time_elapsed_each_action"])
        if not simulate:
            reward = self.last_reward[player] = self.reward_fcn(player)
            self.round_reward[player] += reward
            self.tot_reward[player] += reward
        done   = self.done
        return None if simulate else reward, done

    def get_winner(self):
        if not self.done:
            return None
        for i in range(self.settings["n_players"]):
            if not self.backend.info[i].dead:
                return i
        if self.debug: self.log.warning("get_winner: env returned done=True, no one was alive to win game. I returned winner=666 in the hopes that this will be noticed...")
        return 666 #This should never happen.

    def simulate_all_actions(self, player):
        actions = self.get_actions(player=player)
        return self.simulate_actions(actions, player)

    def get_state(self):
        #The state_processor is responsible for extracting the data from the backend that is part of the state
        return data_types.state(self.backend, self.state_processor)

    def reward_fcn(self, player):
        base = 0
        if self.done:
            # base: me dead -> -1, you dead -> 1, both dead (maybe possible?)
            medead, youdead = int(self.backend.states[player].dead[0]), int(self.backend.states[1-player].dead[0])
            base = youdead - medead
            if medead and youdead:
                base = -1
        if not self.settings["extra_rewards"]:
            return data_types.maingoal_reward([base])
        #Auxiliary goals... [ upcoming research :-) ]
        w_base, w_combo = self.settings["reward_ammount"]
        combo = int(self.backend.states[player].combo_count[0])
        r = self.reward[player] = data_types.maingoal_reward([w_base*base, w_combo*combo])
        return r

    def get_info(self):
        ret = {}
        # speed = [self.backend.info[i].speed for i in range(self.settings["n_players"])]
        ret["is_dead"] = [self.backend.states[i].dead[0] for i in range(self.settings["n_players"])]
        # ret["speed"] = speed
        ret["reward"] = self.reward
        ret["tot_reward"] = self.tot_reward
        ret["round_reward"] = self.round_reward
        ret["rounds_played"] = self.rounds_played
        return ret

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

if __name__ == "__main__":
    ### Backend-unit test!
    np.random.seed(0)
    settings = default_settings.copy().update({"render" : True})
    env, p = tetris_env(settings=default_settings), 0
    #Set backend seed -> reset
    while not env.done:
        a = env.get_random_action(player=p)
        env.perform_action(a, player=p)
        env.render()
        p = 1-p
    print("dead:",[env.backend.state[p].dead for p in range(2)])
    input("OK?")
