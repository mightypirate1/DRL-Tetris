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
from aux.scoreboard import scoreboard
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
    if run_settings["--no-null"]:
        S["bar_null_moves"] = True
    if run_settings["--all-pieces"]:
        S["pieces"] = default_settings["pieces"]
    if not run_settings["--res"]:
        S["render_screen_dims"] = default_settings["render_screen_dims"]
    S["render"] = not run_settings["--no-rendering"]
    return S

def random_match(agents, names):
    if len(all_agents) > 2:
        # random_player_1 vs random_player_2
        agent_idxs = np.random.choice(np.arange(len(agents)), 2, replace=False)
    elif len(all_agents) == 1:
        # me against myself
        agent_idxs = np.array([0,0])
    else:
        #me against you, same positions
        agent_idxs = np.array([0,1])
    a = [agents[i]  for i in agent_idxs]
    # w = [weights[i] for i in agent_idxs]
    n = [names[i]   for i in agent_idxs]
    if len(all_agents) > 2:
        print(n[0], "vs", n[1])
    return a, n

docoptstring = \
'''
Eval

Usage:
    eval.py <weights> ... [options] [--no-null | --null]

Options:
    --reload            Attempt to reload weights on reset. [default: False]
    --all-pieces        Force play with all pieces. [default: False]
    --no-null           Forces disabling of null-moves [default: False]
    --null              Forces enabling of null-moves [default: False]
    --fast              Go full speed, potentially faster than real-time [default: False]
    --debug             Lot's of prints [default: False]
    --res               Use the resolution used when training. Default behavior is to use the global default. [default: False]
    --no-rendering      Disables rendering [default: False]
    --steps S           Number of steps [default: 5000]
    --frac              Print scoreboard with fractions instead of floats. [default: False]
'''
run_settings = docopt.docopt(docoptstring)
total_steps = int(run_settings["--steps"])
if len(run_settings["<weights>"]) < 2:
    run_settings["<weights>"] += run_settings["<weights>"]
settingsfiles = map(utils.find_weight_settings, run_settings["<weights>"])
settings =      list(map(utils.load_settings,settingsfiles))
####
####
####
####
####
####
####
####
####
####
####
####

#Wedge some settings in...
assert utils.test_setting_compatibility(*settings), "Incompatible settings :("
s = adjust_settings(settings[0].copy())
frac, weights_str, debug, fast, reload_weights, render = run_settings["--frac"], run_settings["<weights>"], run_settings["--debug"], run_settings["--fast"], run_settings["--reload"], s["render"]

with tf.Session(config=tf.ConfigProto(log_device_placement=False,device_count={'GPU': 1})) as session:
    n_envs = 1
    #Initialize env!
    env = s["env_vector_type"](
                               n_envs,
                               s["env_type"],
                               settings=s,
                               )
    all_agents, all_names = list(), list()
    #Initialize agents!
    for i, setting, weight in zip(range(len(settings)), settings, weights_str):
        setting = adjust_settings(setting)
        a = setting["agent_type"](
                                    n_envs,
                                    id=i,
                                    mode=threads.WORKER,
                                    sandbox=setting["env_type"](settings=setting),
                                    session=session,
                                    settings=setting,
                                )
        w = utils.weight_location(weight)
        a.load_weights(*w)
        all_agents.append(a)
        n = setting["run-id"]
        if n in all_names: n += str(all_names.count(n))
        all_names.append(n)
        # all_weights.append(w)

    #Initialize run!
    trajectory_start, current_player = 0, np.array([1])
    s_prime = env.get_state()
    game_score = scoreboard(all_names, width=120)

    agent, name = random_match(all_agents, all_names)
    game_score.set_current_players(name)
    # Game loop!
    for t in range(0,total_steps):
        #Take turns...
        current_player = 1 - current_player
        state = s_prime

        #Get action from agent
        action_idx, action    = agent[current_player[0]].get_action(state, player=current_player[0], training=False)
        # action = env.get_random_action(player=current_player)

        #Perform action
        reward, done = env.perform_action(action, player=current_player)
        s_prime = env.get_state()

        #Debug-prints:
        if debug:
            print("player", current_player[0], action_idx, " -> reward :", reward[0](), "(total", env.envs[0].round_reward,")", done)
            print("---")

        #Render?
        if render:
            env.render()
            if not fast:
                time.sleep(s["time_elapsed_each_action"]/1000)

        #Reset the envs that reach terminal states
        for i,d in enumerate(done):
            if d:
                #End current round, resets etc
                assert n_envs == 1, "Scoreboard will not be correct unless n_envs == 1"
                for p,dead in enumerate(env.envs[0].get_info()["is_dead"]):
                    if not dead:
                        game_score.declare_winner(name[p])
                print(game_score.score_table(frac=frac))
                print("{} Round ended. {} steps.".format(utils.progress_bar(t,total_steps),t-trajectory_start))
                env.reset(env=i)
                #Prepare next round!
                agent, name = random_match(all_agents, all_names) #change who's go it is!
                game_score.set_current_players(name)
                for a,w in zip(agent, weight):
                    if reload_weights:
                        if debug:
                            print("agent loaded:",w,"(",a,")")
                        try:
                            a.load_weights(*weight)
                        except:
                            print("Failed to re-load weights...")
                current_player[i] = np.random.choice([0,1])
                round_reward = [0,0]
                trajectory_start = t+1
