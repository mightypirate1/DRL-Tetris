import tensorflow as tf
import docopt
import numpy as np
import time
import sys

from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.vector_agent import vector_agent, vector_agent_trainer
import threads.threaded_runner
from aux.settings import default_settings
import aux.utils as utils
import threads

#####
##  This is just a script with a main-loop for evaluations.
##
##  TODO: remake to use the code from the worker_thread!
#####

def adjust_settings(S):
    if run_settings["--null"]:
        S["bar_null_moves"] = False
    if run_settings["--nonull"]:
        S["bar_null_moves"] = True
    if run_settings["--all-pieces"]:
        S["pieces"] = default_settings["pieces"]
    if not run_settings["--res"]:
        S["render_screen_dims"] = default_settings["render_screen_dims"]
    S["tau_learning_rate"] = 0.995
    return S

docoptstring = \
'''
Eval

Usage:
    eval.py <weights> <weights> ... [--no-reload] [--all-pieces] [--nonull|--null] [--fast] [--debug] [--res]
'''
run_settings = docopt.docopt(docoptstring)
settingsfiles = map(utils.find_weight_settings, run_settings["<weights>"])
settings =      list(map(utils.load_settings,settingsfiles))

#Wedge some settings in...
assert utils.test_setting_compatibility(*settings), "Incompatible settings :("
s = settings[0].copy()
s["render"] = True
s = adjust_settings(s)
with tf.Session(config=tf.ConfigProto(log_device_placement=False,device_count={'GPU': 1})) as session:
    n_envs = len(settings) // 2
    assert n_envs == 1, "More or less than 2 players is not yet implemented"
    #Initialize env!
    env = s["env_vector_type"](
                               n_envs,
                               s["env_type"],
                               settings=s,
                               )
    agent, scoreboard = list(), dict()
    #Initialize agents!
    for i, setting, weight in zip(range(len(settings)), settings, run_settings["<weights>"]):
        setting = adjust_settings(setting)
        a = setting["agent_type"](
                                    n_envs,
                                    id=i,
                                    mode=threads.WORKER,
                                    sandbox=setting["env_type"](settings=setting),
                                    session=session,
                                    settings=setting,
                                )
        a.load_weights(*utils.weight_location(weight))
        agent.append(a)
        scoreboard[i] = 0

    #Initialize run!
    trajectory_start = 0
    s_prime = env.get_state()
    current_player = np.random.choice([i for i in range(s["n_players"])], size=n_envs )

    # Game loop!
    for t in range(0,5000):
        #Take turns...
        current_player = 1 - current_player
        state = s_prime

        #Get action from agent
        action_idx, action    = agent[current_player[0]].get_action(state, player=current_player[0], training=False)
        # action = env.get_random_action(player=current_player)

        #Perform action
        reward, done = env.perform_action(action, player=current_player)
        s_prime = env.get_state()

        #Store to memory
        experience = (state, action_idx, reward, s_prime, current_player ,done)
        agent[current_player[0]].store_experience(experience)

        #Debug-prints:
        if run_settings["--debug"]:
            print("player", current_player[0], action_idx, " -> reward :", reward[0](), "(total", env.envs[0].round_reward,")", done)
            print("---")
        #Render?
        if s["render"]:
            env.render()
            if not run_settings["--fast"]:
                time.sleep(0.05)

        #Reset the envs that reach terminal states
        for i,d in enumerate(done):
            if d:
                assert n_envs == 1, "Scoreboard will not be correct unless n_envs == 1"
                for p,dead in enumerate(env.envs[0].get_info()["is_dead"]):
                    if not dead:
                        scoreboard[p] += 1
                for a,w in zip(agent, run_settings["<weights>"]):
                    if not run_settings["--no-reload"]:
                        if run_settings["--debug"]:
                            print("agent loaded:",w,"(",a,")")
                        a.load_weights(*utils.weight_location(w))
                    a.ready_for_new_round(training=False, env=i)
                print("Round ended. {} steps (avg: {}), score: {}-{}".format(t-trajectory_start, agent[0].avg_trajectory_length, scoreboard[0], scoreboard[1]))
                current_player[i] = np.random.choice([0,1])
                env.reset(env=i)
                round_reward = [0,0]
                trajectory_start = t+1
