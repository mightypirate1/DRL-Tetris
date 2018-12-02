import logging
import random
import time
import tensorflow as tf
from environment.tetris_environment import tetris_environment
from agents.my_first_agent import my_agent

# # # # #
# Configuration:
# # #
''' - - - - - - - - - - - - - - '''
default_settings = {
                    "n_players" : 2,
                    "env" : tetris_environment,
                    "agent" : my_agent,
                    "max_actions" : 50,
                    "gamma" : 0.99,
                    "session" : tf.Session(),
                    "render" : True,
                    "max_round_time" : None,
                    "time_elapsed_each_action" : 100,
                    "balance_winrate" : True,
                    "winrate_tolerance" : 1.2,
                    "winrate_learningrate" : 0.04,
                    }
''' - - - - - - - - - - - - - - '''

class game_controller:
    def __init__(self, session, settings=None):
        logging.basicConfig(filename='logs/scratchpaper.log',level=logging.DEBUG)
        self.settings = default_settings.copy()
        if settings is not None:
            for x in settings:
                self.settings[x] = settings[x]
        settings_ok = self.process_settings() #Checks so that the settings are not conflicting
        self.env = self.settings['env'](settings=self.settings)
        self.agent = [ self.settings['agent'](id=x, session=session, sandbox=self.env.copy(), settings=self.settings) for x in range(self.settings['n_players'])]
        #IT FEELS UGLY TO HAVE THIS HERE?
        session.run(tf.global_variables_initializer())

    class session_stats:
        def __init__(self, controller):
            self.n_games = 0
            self.scores = {}
            self.exp_scores = {}
            self.gamma = controller.settings["winrate_learningrate"]
            for a in controller.agent:
                self.scores[a] = []
                self.exp_scores[a] = 1/controller.settings["n_players"]
            self.controller = controller
        def get_winrate(self,a):
            return sum(self.scores[a])/len(self.scores[a])
        def get_exp_weighted_winrate(self,a):
            return self.exp_scores[a]
        def print_winrate(self, exp=False):
            f, s = (self.get_exp_weighted_winrate, "exp_winrate") if exp else (self.get_winrate, "winrate")
            print("{}:{}".format(s,[f(a) for a in self.controller.agent]))
        def report_winner(self,a):
            for x in self.scores:
                if x.id == a.id:
                    self.scores[x].append(1)
                    self.exp_scores[x] = (1-self.gamma)*self.exp_scores[x] + self.gamma#* 1
                else:
                    self.scores[x].append(0)
                    self.exp_scores[x] = (1-self.gamma)*self.exp_scores[x]#+ self.gamma * 0

    def process_settings(self):
        print("game_controller_BOUNCE process_settings not implemented yet...")
        return True

    def time_out(self, time):
        if self.settings["max_round_time"] is None or time < self.settings["max_round_time"]:
            return False
        return True

    def train(self, n_steps=0, wait=True):
        def reset_trainer_variables():
            #Returns a bunch of constants to make code tidier below...
            return random.randrange(self.settings['n_players']),\
                    [True]*self.settings['n_players'],\
                    [None, None]*self.settings['n_players'],\
                    [None, None]*self.settings['n_players'],\
                    [False, False]*self.settings['n_players'],\
                    0
        #Initialize some variables, then go!
        self.env.reset() #reset env
        session_stats = self.session_stats(self)
        next_state = self.env.get_state()
        for a in self.agent:
            a.init_training() #"reset" agents
        current_player, first_move, last_state, last_action, player_done, round_time = reset_trainer_variables()

        for t in range(n_steps):
            if t%100 == 0:
                print("step {}".format(t))
                print("---")
            state = next_state #state s
            #Store transition
            if not first_move[current_player]:
                experience = (last_state[current_player], last_action[current_player], state, player_done[current_player], {"time" : t})
                self.agent[current_player].store_experience(experience)
            #Reset sometimes
            if self.time_out(t) or player_done[current_player]: #(this happens when the game is over, and every agent has noticed)
                session_stats.report_winner(self.agent[self.env.get_winner()])
                for a in self.agent:
                    a.ready_for_new_round(training=True)
                self.env.reset() #first reset env, then all relevant trainer-variables
                current_player, first_move, last_state, last_action, player_done, round_time = reset_trainer_variables()
                next_state = self.env.get_state()
                continue
            #Take an action
            action_idx, action = self.agent[current_player].get_action(state, training=True)
            player_done[current_player] = self.env.perform_action(action, player=current_player, render=self.settings["render"])
            next_state = self.env.get_state()
            #Training sometimes
            if self.agent[current_player].is_ready_for_training():
                #Unless (we try and keep winrates even AND this agent wins too much), we train!
                if not (self.settings["balance_winrate"] and session_stats.get_exp_weighted_winrate(self.agent[current_player]) > self.settings["winrate_tolerance"]/self.settings["n_players"]):
                    session_stats.print_winrate(exp=True)
                    self.agent[current_player].do_training()
            #Take notes of what s,a were so that we know that when it's our turn again, and we get to know s'
            last_state[current_player] = state
            last_action[current_player] = action_idx
            #Change active agent (take turns!)
            first_move[current_player] = False
            current_player = (current_player+1)%self.settings['n_players']


    def test(self, n_steps=0, wait=True, pause=False):
        def reset_tester_variables():
            #Returns a bunch of constants to make code tidier below...
            return random.randrange(self.settings['n_players']), False
        #Initialize some variables, then go!
        self.env.reset() #reset env
        session_stats = self.session_stats(self)
        next_state = self.env.get_state()
        for a in self.agent:
            a.ready_for_new_round() #"reset" agents
        current_player, done = reset_tester_variables()
        for t in range(n_steps):
            state = next_state #state s
            #Reset sometimes
            if self.time_out(t) or done: #(this happens when the game is over, or time-limit is reached)
                for a in self.agent:
                    a.ready_for_new_round()
                self.env.reset() #first reset env, then all relevant trainer-variables
                current_player, done = reset_tester_variables()
                next_state = self.env.get_state()
                continue
            #Take an action
            action_idx, action = self.agent[current_player].get_action(state, training=False, verbose=False)
            done = self.env.perform_action(action, player=current_player, render=self.settings["render"])
            next_state = self.env.get_state()
            if done:
                session_stats.report_winner(self.agent[self.env.get_winner()])
                print("agent{} won!!!".format(self.env.get_winner()))
                session_stats.print_winrate(exp=False)
                session_stats.print_winrate(exp=True)
                if pause:
                    input("Enter to start new round!")
            #Wait so the humans can keep up!
            if wait:
                time.sleep(self.settings["time_elapsed_each_action"]/1000)
            #Change active agent (take turns!)
            current_player = (current_player+1)%self.settings['n_players']
    def restore(self):
        ''' TODO! '''
        pass
