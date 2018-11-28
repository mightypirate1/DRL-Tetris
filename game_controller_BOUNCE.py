import logging
import random
from environment.tetris_environment import tetris_environment
from agents.test_agent import test_agent
# # # # #
# Configuration:
# # #
''' - - - - - - - - - - - - - - '''
default_settings = {
                    "n_players" : 2,
                    "env" : tetris_environment,
                    "agent" : test_agent,
                    }
''' - - - - - - - - - - - - - - '''

class game_controller:
    def __init__(self, settings=None):
        logging.basicConfig(filename='logs/scratchpaper.log',level=logging.DEBUG)
        self.settings = default_settings.copy()
        if settings is not None:
            for x in settings:
                self.settings[x] = settings[x]
        settings_ok = self.process_settings() #Checks so that the settings are not conflicting
        self.env = self.settings['env'](settings=self.settings)
        self.agent = [ self.settings['agent'](id=x, sandbox=self.env.copy(), settings=self.settings) for x in range(self.settings['n_players'])]

    def process_settings(self):
        print("game_controller_BOUNCE process_settings not implemented yet...")
        return True

    def train(self, n_steps=0):
        def reset_trainer_variables():
            #Returns a bunch of constants to make code tidier below...
            return random.randrange(self.settings['n_players']),\
                    [True]*self.settings['n_players'],\
                    [None, None]*self.settings['n_players'],\
                    [None, None]*self.settings['n_players'],\
                    [False, False]*self.settings['n_players']
        #Initialize some variables, then go!
        next_state = self.env.get_state()
        current_player, first_move, last_state, last_action, player_done = reset_trainer_variables()
        for t in range(n_steps):
            state = next_state #state s
            #Store transition
            if not first_move[current_player]:
                experience = (last_state[current_player], last_action[current_player],t, {})
                self.agent[current_player].store_experience(experience)
            #Reset sometimes
            if player_done[current_player]: #(this happens when the game is over, and every agent has noticed)
                for a in self.agent:
                    a.process_trajectory()
                self.env.reset() #first reset env, then all relevant trainer-variables
                current_player, first_move, last_state, last_action, player_done = reset_trainer_variables()
                next_state = self.env.get_state()
                continue
            #Take an action
            action_idx, action = self.agent[current_player].get_action(state)
            player_done[current_player] = self.env.perform_action(action, player=current_player)
            next_state = self.env.get_state()
            #Training sometimes
            if self.agent[current_player].is_ready_for_training():
                self.agent[current_player].do_training()
            #Take notes of what s,a were so that we know that when it's our turn again, and we get to know s'
            last_state[current_player] = t#state[current_player]
            last_action[current_player] = action_idx
            #Change active agent (take turns!)
            first_move[current_player] = False
            current_player = (current_player+1)%self.settings['n_players']

    def restore(self):
        ''' TODO! '''
        pass

    def test(self):
        ''' TODO! '''
        pass
