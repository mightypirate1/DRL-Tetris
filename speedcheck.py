from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.vector_agent.vector_agent import vector_agent
import tensorflow as tf
import docopt
import time

t_steps = 1000
n_envs  = 64

docoptstring = \
'''Speedcheck!
Usage:
  Speedcheck.py [--n N] [--steps S]  [--no_rendering]

Options:
    --n N      N envs. [default: 16]
    --steps S  Run S environments steps in total. [default: 1000]
'''
docoptsettings = docopt.docopt(docoptstring)
t_steps = int(docoptsettings["--steps"])
n_envs = int(docoptsettings["--n"])
render = not docoptsettings["--no_rendering"]

settings = {
            "render" : render,
           }

print("Speedcheck:")
for x in docoptsettings:
    print("\t{} : {}".format(x,docoptsettings[x]))

with tf.Session() as session:
    envs = tetris_environment_vector(n_envs, tetris_environment, settings=settings)
    agent = vector_agent(
                         n_envs,
                         session=session,
                         sandbox=tetris_environment(settings=settings),
                         settings=settings,
                        )
    t0 = time.time()
    current_player = 1
    s = envs.get_state()
    print("Go!")
    for t in range(t_steps // n_envs):
        current_player = 1 - current_player
        _,a = agent.get_action(s, player=current_player)
        ds = envs.perform_action(a)
        s = envs.get_state()
        for i,d in enumerate(ds):
            if d: envs.reset(env=[i])
        if render:
            envs.render(env=0)
    delta_t = time.time() - t0
    print("{} steps in {} secs. ({} steps/sec)".format(t_steps, delta_t, t_steps/delta_t))








''' CCCC '''
    # for t in range(n_steps):
    #     if t%100 == 0:
    #         print("step {}".format(t))
    #         print("---")
    #     state = next_state #state s
    #     #Store transition
    #     if not first_move[current_player]:
    #         experience = [last_state[current_player], last_action[current_player], state, player_done[current_player], {"time" : t}]
    #         self.agent[current_player].store_experience(experience)
    #     #Reset sometimes
    #     if self.time_out(t) or player_done[current_player]: #(this happens when the game is over, and every agent has noticed)
    #         winner = self.env.get_winner()
    #         if winner not in [None, 666]: #FAI-LPROOF AGAINST BACKEND-BUG!
    #             session_stats.report_winner(self.agent[winner])
    #         for a in self.agent:
    #             a.ready_for_new_round(training=True)
    #         self.env.reset() #first reset env, then all relevant trainer-variables
    #         current_player, first_move, last_state, last_action, player_done, round_time = reset_trainer_variables()
    #         next_state = self.env.get_state()
    #         continue
    #     #Take an action
    #     action_idx, action = self.agent[current_player].get_action(state, training=True)
    #     player_done[current_player] = self.env.perform_action(action, player=current_player, render=self.settings["render"])
    #     next_state = self.env.get_state()
    #     #Training sometimes
    #     if self.agent[current_player].is_ready_for_training():
    #         #Unless (we try and keep winrates even AND this agent wins too much), we train!
    #         if not (self.settings["balance_winrate"] and session_stats.get_exp_weighted_winrate(self.agent[current_player]) > self.settings["winrate_tolerance"]/self.settings["n_players"]):
    #             session_stats.print_winrate(exp=True)
    #             self.agent[current_player].do_training()
    #     #Take notes of what s,a were so that we know that when it's our turn again, and we get to know s'
    #     last_state[current_player] = state
    #     last_action[current_player] = action_idx
    #     #Change active agent (take turns!)
    #     first_move[current_player] = False
    #     current_player = (current_player+1)%self.settings['n_players']
