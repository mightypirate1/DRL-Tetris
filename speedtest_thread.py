from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.vector_agent import vector_agent, vector_agent_trainer
from agents.pg_vector_agent import pg_vector_agent, pg_vector_agent_trainer
from aux.parameter import *
import threads.threaded_runner

import sys
import time
import docopt
import tensorflow as tf

total_steps = 1000
n_envs = 64

docoptstring = \
'''Speedcheck!
Usage:
  Speedcheck.py [--n N] [--m M] [--steps S]  [--no-rendering] [--debug] [--pg]

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
debug = docoptsettings["--debug"]


settings = {
            #Project
            "run-id" : "threads_04",

            #Train parameters
            "n_samples_each_update"    : 8192 if not docoptsettings["--pg"] else 16384,
            "minibatch_size"           : 128 if not docoptsettings["--pg"] else 1024,
            "epsilon"                  : constant_parameter(1.0),
            "value_lr"                 : linear_parameter(1e-6, final_val=1e-7, time_horizon=total_steps),
            "prioritized_replay_alpha" : constant_parameter(0.7),
            "prioritized_replay_beta"  : linear_parameter(0.5, final_val=1.0, time_horizon=total_steps),
            "experience_replay_size"   : 10**6 if not docoptsettings["--pg"] else 2*10**4,
            "alternating_models"       : False,
            "time_to_training"         : 10**3 if not docoptsettings["--pg"] else 1,

            #Dithering
            "dithering_scheme"    : "distribution_boltzman",
            "action_temperature"  : exp_parameter(1.0, final_val=16, time_horizon=total_steps),

            #Game settings
            # "game_size"                : [10,5],
            # "pieces"                   : [4,6],
            "time_elapsed_each_action" : 400,
            #Types
            "env_vector_type"   : tetris_environment_vector,
            "env_type"          : tetris_environment,
            "agent_type"        : vector_agent.vector_agent                 if not docoptsettings["--pg"] else pg_vector_agent.pg_vector_agent,
            "trainer_type"      : vector_agent_trainer.vector_agent_trainer if not docoptsettings["--pg"] else pg_vector_agent_trainer.pg_vector_agent_trainer,

            #Threading
            "run_standalone"    : docoptsettings["--debug"],
            "n_workers"         : n_workers,
            "n_envs_per_thread" : n_envs_per_thread,
            "worker_steps"      : total_steps // n_envs,
            "process_patience"  : [0.1,0.1, 10.0], #runner/trainer/process_manager
            "worker_net_on_cpu" : True,
            "trainer_net_on_cpu": False,
            #Communication
            "trainer_thread_save_freq"  : 100,
            "n_train_epochs_per_update" : 15,
            "worker_data_send_fequency" : 50,
            "weight_transfer_frequency" : 5 if not docoptsettings["--pg"] else 1,

            #Misc.
            "render"            : render,
            "bar_null_moves"    : True,

            ###
            ###  PG
            ###
            "n_actions"              : 25,
            "state_head_n_hidden"    : 2,
            "state_head_hidden_size" : 512,
            "state_head_output_size" : 10,
            "joined_n_hidden"        : 2,
            "joined_hidden_size"     : 512,
            "weight_loss_policy"     : 1.0,
            "weight_loss_entropy"    : 0.1,
            "weight_loss_value"      : 0.5,
            "clipping_parameter"     : constant_parameter(0.2),
           }

print("Speedcheck:")
for x in docoptsettings:
    print("\t{} : {}".format(x,docoptsettings[x]))


process_manager = threads.threaded_runner.threaded_runner(settings=settings)

##
#Thread debugger
#We get better error messages if we run just one process. Activate with "--debug"
if debug:
    print("Executing only thread_0:")
    process_manager.threads["workers"][0]()
    print("___")
    exit("thread debug run done.")

#Init PARALLEL
###########################
print("Creating processes")
process_manager.run(total_steps // n_envs )
process_manager.join()
T_parallel = process_manager.get_avg_runtime()
sys.stdout.flush()
print("Multi-processes finished")
#Done!

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
    print("Starting serial")
    T_serial_start = time.time()
    current_player = 1
    s = env.get_state()
    print("Go!")
    for t in range(total_steps):
        current_player = 1 - current_player
        _,a = agent.get_action(s, player=current_player)
        # a = env.get_random_action()
        r, ds = env.perform_action(a, player=current_player)
        s = env.get_state()
        for i,d in enumerate(ds):
            if d: env.reset(env=i)
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
