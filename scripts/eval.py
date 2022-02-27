import tensorflow.compat.v1 as tf
import docopt
import numpy as np
import time

import experiments.presets
from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from tools.scoreboard import scoreboard
import tools.utils as utils
import threads

#######
### Ad-hoc nn visuals
#####

def visualize_nn_output(raw, player=0):
    # utils.progress_bar(t,total_steps),t-trajectory_start)
    length = 30
    blank = ' ' * length
    pref = blank if player else ''
    for prob_rtp, piece_value, piece in zip(*raw):
        pi = prob_rtp[:,:,piece]
        unif = np.ones_like(pi) / (pi.shape[0] * pi.shape[1])
        entropy = utils.entropy(pi)
        max_entropy = utils.entropy(unif)
        entr_vis = utils.progress_bar(entropy,max_entropy, length=length)
        print(pref + entr_vis, piece_value[0,0,piece])


#####
##  This is just a script with a main-loop for evaluations.
##
##  TODO: remake to use the code from the worker_thread!
#####

def adjust_settings(s):
    S = s.copy()
    if run_settings["--null"]:
        S["bar_null_moves"] = False
    if run_settings["--no-null"]:
        S["bar_null_moves"] = True
    if run_settings["--all-pieces"]:
        S["pieces"] = experiments.presets.presets["default"]["pieces"]
    if not run_settings["--res"]:
        S["render_screen_dims"] = experiments.presets.presets["default"]["render_screen_dims"]
    if run_settings["--argmax"]:
        S["eval_distribution"] = "argmax"
    S["render"] = not run_settings["--no-rendering"]
    S["worker_net_on_cpu"] = not run_settings["--gpu"]
    return S

def random_match(agents, names, weights):
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
    w = [weights[i] for i in agent_idxs]
    n = [names[i]   for i in agent_idxs]
    if len(all_agents) > 2:
        print(n[0], "({})".format(w[0][1]), "vs", n[1], "({})".format(w[1][1]))
    return a, n, w

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
    --debug             Lots of prints [default: False]
    --res               Use the resolution used when training. Default behavior is to use the global default. [default: False]
    --no-rendering      Disables rendering [default: False]
    --steps S           Number of steps [default: 5000]
    --frac              Print scoreboard with fractions instead of floats. [default: False]
    --width W           With of the score crosstable [default: 120]
    --argmax            Force evals to use argmax, regardless of project setting. [default: False]
    --gpu               Run on GPU. [default: False]
    --solo              Play like it's a 1-player game (only P1 plays) [default: False]
    --wait              Wait on input before restarting
'''
run_settings = docopt.docopt(docoptstring)
total_steps = int(run_settings["--steps"])
if len(run_settings["<weights>"]) < 2:
    run_settings["<weights>"] += run_settings["<weights>"]
settingsfiles = map(utils.find_weight_settings, run_settings["<weights>"])
settings =      list(map(utils.load_settings,settingsfiles))
####

#Wedge some settings in...
assert utils.test_setting_compatibility(*settings), "Incompatible settings :("
s = adjust_settings(settings[0].copy())
frac, weights_str, debug, fast, reload_weights, solo, render, score_width, wait = run_settings["--frac"], run_settings["<weights>"], run_settings["--debug"], run_settings["--fast"], run_settings["--reload"], run_settings["--solo"], s["render"], int(run_settings["--width"]), run_settings["--wait"]

with tf.Session(config=tf.ConfigProto(log_device_placement=False,device_count={'GPU': 1})) as session:
    n_envs = 1
    #Initialize env!
    env = s["env_vector_type"](
        n_envs,
        s["env_type"],
        settings=s,
    )
    all_agents, all_names, all_weights, name_count = list(), list(), list(), dict()
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
        if n not in name_count:
            name_count[n] = 0
        else:
            name_count[n] += 1
            n += "_"+str(name_count[n])
        all_names.append(n)
        all_weights.append(w)
        print("registering:", w, "-> ", n)
    #Initialize run!
    trajectory_start, current_player = 0, np.array([1])
    s_prime = env.get_state()
    game_score = scoreboard(all_names, width=score_width)

    agent, name, _ = random_match(all_agents, all_names, all_weights)
    game_score.set_current_players(name)
    # Game loop!
    for t in range(0,total_steps):
        #Take turns...
        current_player = 1 - current_player
        state = s_prime

        if solo:
            current_player= np.array([0])#
            env.envs[0].backend.states[1].field[:,:] = 0

        #Get action from agent
        action_idx, action, raw = agent[current_player[0]].get_action(state, player=current_player[0], training=False, verbose=debug)
        visualize_nn_output(raw, player=current_player)

        #Perform action
        reward, done = env.perform_action(action, player=current_player)
        s_prime = env.get_state()

        #Debug-prints:
        if debug:
            print("player", current_player[0], action_idx, " -> reward :", reward[0](), "(total", env.envs[0].round_reward,")", done)

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
                if wait:
                    input('<enter> to start the next round')
                env.reset(env=i)
                #Prepare next round!
                agent, name, weight = random_match(all_agents, all_names, all_weights) #change who's go it is!
                game_score.set_current_players(name)
                for a,w in zip(agent, weight):
                    if reload_weights:
                        if debug:
                            print("[*]agent loaded:",w,"(",a,")")
                        else:
                            print("[*]", end='')
                        try:
                            a.load_weights(*w)
                        except Exception as e:
                            print("Failed to re-load weights...", e)
                current_player[i] = np.random.choice([0,1])
                round_reward = [0,0]
                trajectory_start = t+1
