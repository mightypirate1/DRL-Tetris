from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.vector_agent.vector_agent import vector_agent
import threaded_runner

import tensorflow as tf
import docopt
from time import time

total_steps = 1000
n_envs  = 64

docoptstring = \
'''Speedcheck!
Usage:
  Speedcheck.py [--n N] [--m M] [--steps S]  [--no-rendering]

Options:
    --n N      N envs per thread. [default: 16]
    --m M      M runners. [default: 16]
    --steps S  Run S environments steps in total. [default: 1000]
'''
docoptsettings = docopt.docopt(docoptstring)
total_steps = int(docoptsettings["--steps"])
n_runners = int(docoptsettings["--m"])
n_envs_per_thread = int(docoptsettings["--n"])
n_envs = n_runners * n_envs_per_thread
render = not docoptsettings["--no-rendering"]

settings = {
            "render" : render,
           }

print("Speedcheck:")
for x in docoptsettings:
    print("\t{} : {}".format(x,docoptsettings[x]))

with tf.Session() as session:
    envs = [tetris_environment_vector(n_envs, tetris_environment, settings=settings) for _ in range(n_runners)]
    agents = [vector_agent(
                         n_envs,
                         id=thread_n,
                         session=session,
                         sandbox=tetris_environment(settings=settings),
                         settings=settings,
                        ) for thread_n in range(n_runners) ]
    agent = vector_agent(
                         n_envs,
                         id="trainer",
                         session=session,
                         sandbox=tetris_environment(settings=settings),
                         settings=settings,
                        )

    #Run stuff
    pool = threaded_runner.threaded_runner(envs=envs, runners=agents, trainer=agent)
    print("Starting pool")
    T_thread_start = time()
    pool.run(total_steps // n_runners)
    pool.join()
    T_thread_stop = time()
    print("Pool finished")
    #Done!


    ##Init SERIAL
    print("Starting serial")
    T_serial_start = time()
    current_player = 1
    s = envs[0].get_state()
    print("Go!")
    for t in range(total_steps // n_envs):
        current_player = 1 - current_player
        _,a = agent.get_action(s, player=current_player)
        ds = envs[0].perform_action(a)
        s = envs[0].get_state()
        for i,d in enumerate(ds):
            if d: envs[0].reset(env=[i])
        if render:
            envs[0].render(env=0)
        print("Step {}".format((t+1)*n_envs))
    T_serial_stop = time()
    print("Serial done")
    #Done!

#Show what we collected (parallel)!
print("[Parallel] Collected {} trajectories:".format(-1))
print("Total: {} collected in {} seconds ({} steps/second)".format(total_steps, T_thread_stop - T_thread_start, total_steps/(T_thread_stop - T_thread_start) ))

print("[Serial] Collected {} trajectories:".format(-1))
print("Total: {} collected in {} seconds ({} steps/second)".format(total_steps, T_serial_stop - T_serial_start, total_steps/(T_serial_stop - T_serial_start) ))
