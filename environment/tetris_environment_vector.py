import logging
import time
from environment.tetris_environment import tetris_environment
import environment.env_utils.state_processors as state_processors
import environment.env_utils.draw_tetris as draw_tetris
from environment.data_types.action_list import action_list
from environment.data_types.state import state
import aux.settings

class tetris_environment_vector:
    def __init__(self, n_envs, env_type, init_envs=None, settings={}):
        if type(init_envs) is not list: init_envs = [init_envs for _ in range(n_envs)]
        #Set up settings
        self.settings = aux.settings.default_settings.copy()
        if settings is not None:
            for x in settings:
                self.settings[x] = settings[x]
        self.n_envs = n_envs
        self.envs = [env_type(id=i, settings=settings, init_env=e) for i,e in enumerate(init_envs)]

    # def __getattr__(self, attr):
    #     class wrapper:
    #         def __init__(self,envs,attr):
    #             self.envs, self.attr = envs, attr
    #             self.fs = [getattr(e,attr) for e in envs]
    #         def argparser(self, *args, **kwargs):
    #             #Distribute list-tye  arguments across the env_list
    #             _args = [[]]*len(self.envs)                      #These hold the stuff passed to the function eventually
    #             _kwargs = [{}]*len(self.envs)
    #             for j,_ in enumerate(self.envs):
    #                 for i,a in enumerate(args):
    #                         if type(a) is list and len(a)>0:
    #                             assert len(a) == len(self.envs), "{} != {}".format(len(a), len(self.envs))
    #                             _args[j].append(a[j])
    #                         else:
    #                             _args[j].append(a)
    #                 for x in kwargs:
    #                     print(x)
    #                     if type(kwargs[x]) is list:
    #                         assert len(kwargs[x]) == len(self.envs), "{} != {}".format(len(kwargs[x]), len(self.envs))
    #                         _kwargs[j][x] = kwargs[x][j]
    #                     else:
    #                         _kwargs[j] = kwargs[x]
    #             return _args, _kwargs
    #         def __call__(self,*args, **kwargs):
    #             _args, _kwargs = self.argparser(*args,*kwargs)
    #             return [ f(*a,**kwargs) for f,a in zip(self.fs,_args)]
    #     return wrapper(self.envs, attr)

    # # # # #
    # Env interface fcns
    # # #
    def reset(self, env=None):
        env_list = self.envs if env is None else [self.envs[i] for i in env]
        return [e.reset() for e in env_list]

    def get_actions(self,env=None,player=None):
        env_list = self.envs if env is None else [self.envs[i] for i in env]
        return [e.get_actions(player=player) for e in env_list]

    def get_random_action(self, env=None, player=None):
        env_list = self.envs if env is None else [self.envs[i] for i in env]
        return [e.get_random_action(player=player) for e in env_list]

    def simulate_actions(self,actions, env=None, player=None):
        env_list = self.envs if env is None else [self.envs[i] for i in env]
        return [e.simulate_actions(a, player=player) for e,a in zip(env_list, actions)]

    def perform_action(self,actions, env=None, player=None, render=None):
        env_list = self.envs if env is None else [self.envs[i] for i in env]
        return [e.perform_action(a, player=player, render=render) for  e,a in zip(env_list, actions)]

    def get_winner(self, env=None):
        env_list = self.envs if env is None else [self.envs[i] for i in env]
        return [e.get_winner(actions, player=player) for e in env_list]

    def simulate_all_actions(self,actions, env=None, player=None):
        env_list = self.envs if env is None else [self.envs[i] for i in env]
        return [e.simulate_all_actions(actions, player=player) for e in env_list]

    def get_state(self, env=None, player=None):
        env_list = self.envs if env is None else [self.envs[i] for i in env]
        return [e.get_state(player=player) for e in env_list]

    # # # # #
    # Somewhat hacky fcns for env handling
    # # #
    def get_winner(self, env=None, player=None):
        env_list = self.envs if env is None else [self.envs[i] for i in env]
        return [e.get_winner(player=player) for e in env_list]
    def copy(self):
        exit("!NOT IMPLEMENTED")
        return tetris_environment(settings=self.settings, init_env=self.backend)
    def copy(self, env=None):
        env_list = self.envs if env is None else [self.envs[i] for i in env]
        return [e.copy(actions, player=player) for e in env_list]

    def set(self, target, env=None):
        env_list = self.envs if env is None else [self.envs[i] for i in env]
        if type(target) is list:
            return [e.set(t) for e,t in zip(env_list, target)]
        else:
            return [e.set(target) for e in env_list]

    # # # # #
    # Helper functions
    # # #
    def render(self, env=0):
        self.envs[env].render()

    def generate_pieces(self, env=None):
        env_list = self.envs if env is None else [self.envs[i] for i in env]
        return [e.generate_pieces(actions, player=player) for e in env_list]

    def __str__(self, env=None):
        env_list = self.envs if env is None else [self.envs[i] for i in env]
        ret = "<tetris_vector_env>"
        for x in [e.__str__() for e in env_list]:
            ret += x
        ret += "</tetris_vector_env>"
        return ret
