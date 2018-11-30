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

    def process_settings(self):
        print("game_controller_BOUNCE process_settings not implemented yet...")
        return True

    def time_out(self, time):
        if self.settings["max_round_time"] is None or time < self.settings["max_round_time"]:
            return False
        return True

    def train(self, n_steps=0):
        def reset_trainer_variables():
            #Returns a bunch of constants to make code tidier below...
            return random.randrange(self.settings['n_players']),\
                    [True]*self.settings['n_players'],\
                    [None, None]*self.settings['n_players'],\
                    [None, None]*self.settings['n_players'],\
                    [False, False]*self.settings['n_players'],\
                    0
        #Initialize some variables, then go!
        next_state = self.env.get_state()
        current_player, first_move, last_state, last_action, player_done, round_time = reset_trainer_variables()
        for t in range(n_steps):
            state = next_state #state s
            #Store transition
            if not first_move[current_player]:
                experience = (last_state[current_player], last_action[current_player], state, player_done[current_player], {"time" : t})
                self.agent[current_player].store_experience(experience)
            #Reset sometimes
            if self.time_out(t) or player_done[current_player]: #(this happens when the game is over, and every agent has noticed)
                for a in self.agent:
                    a.ready_for_new_round()
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
                self.agent[current_player].do_training()
            #Take notes of what s,a were so that we know that when it's our turn again, and we get to know s'
            last_state[current_player] = state
            last_action[current_player] = action_idx
            #Change active agent (take turns!)
            first_move[current_player] = False
            current_player = (current_player+1)%self.settings['n_players']

    def test(self, n_steps=0):
        def reset_tester_variables():
            #Returns a bunch of constants to make code tidier below...
            return random.randrange(self.settings['n_players']), False
        #Initialize some variables, then go!
        next_state = self.env.get_state()
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
            action_idx, action = self.agent[current_player].get_action(state, training=False)
            done = self.env.perform_action(action, player=current_player, render=self.settings["render"])
            next_state = self.env.get_state()
            #Wait so the humans can keep up!
            time.sleep(self.settings["time_elapsed_each_action"]/1000)
            #Change active agent (take turns!)
            current_player = (current_player+1)%self.settings['n_players']
    def restore(self):
        ''' TODO! '''
        pass
