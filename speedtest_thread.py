from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.vector_agent.vector_agent import vector_agent
import threaded_runner

import tensorflow as tf
import docopt
import time
import sys

total_steps = 1000
n_envs  = 64

docoptstring = \
'''Speedcheck!
Usage:
  Speedcheck.py [--n N] [--m M] [--steps S]  [--no-rendering]

Options:
    --n N      N envs per thread. [default: 16]
    --m M      M workers. [default: 16]
    --steps S  Run S environments steps in total. [default: 1000]
'''
docoptsettings = docopt.docopt(docoptstring)
total_steps = int(docoptsettings["--steps"])
n_workers = int(docoptsettings["--m"])
n_envs_per_thread = int(docoptsettings["--n"])
n_envs = n_workers * n_envs_per_thread
render = not docoptsettings["--no-rendering"]

settings = {
            "worker_steps"      : total_steps // n_envs,
            "n_envs_per_thread" : n_envs_per_thread,
            "render"            : render,
            "n_workers"         : n_workers,
            "env_vector_type"   : tetris_environment_vector,
            "env_type"          : tetris_environment,
            "agent_type"        : vector_agent,
            "process_patience"  : [0.1,0.1, 0.1],
           }

print("Speedcheck:")
for x in docoptsettings:
    print("\t{} : {}".format(x,docoptsettings[x]))


#Init PARALLEL
###########################
pool = threaded_runner.threaded_runner(
                                        settings=settings
                                      )
print("Starting pool")
pool.run(total_steps // n_envs )
pool.join()
print("Pool finished")
#Done!

sys.stdout.flush()
T_parallel = pool.sum_return_que() / n_workers
###########################


#
##Init SERIAL
with tf.Session() as session:
    env = settings["env_vector_type"](
                                      1,
                                      settings["env_type"],
                                      settings=settings
                                     )
    agent = settings["agent_type"](
                                   1,
                                   id=0,
                                   sandbox=settings["env_type"](settings=settings),
                                   session=session,
                                   settings=settings,
                                  )

    session.run(tf.global_variables_initializer())
    agent.reinitialize_model()

    print("Starting serial")
    T_serial_start = time.time()
    current_player = 1
    s = env.get_state()
    print("Go!")
    for t in range(total_steps):
        current_player = 1 - current_player
        _,a = agent.get_action(s, player=current_player)
        # a = env.get_random_action()
        ds = env.perform_action(a)
        s = env.get_state()
        for i,d in enumerate(ds):
            if d: env.reset(env=[i])
        if render:
            env.render(env=0)
        print("Step {}".format((t+1)))
    T_serial_stop = time.time()
    print("Serial done")
    #Done!

#Show what we collected (parallel)!
print("[Parallel] Collected {} trajectories:".format(None))
print("Total: {} collected in {} seconds ({} steps/second)".format(total_steps, T_parallel, total_steps/T_parallel))

print("[Serial] Collected {} trajectories:".format(None))
print("Total: {} collected in {} seconds ({} steps/second)".format(total_steps, T_serial_stop - T_serial_start, total_steps/(T_serial_stop - T_serial_start) ))
