from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.vector_agent import vector_agent, vector_agent_trainer
from agents.pg_vector_agent import pg_vector_agent, pg_vector_agent_trainer
from agents.agent_utils.reward_shapers import *
from aux.parameter import *
import threads.threaded_runner

import sys
import time
import docopt
import tensorflow as tf

docoptstring = \
'''Threaded trainer!
Usage:
  thread_train.py [--n N] [--m M] [--steps S]  [--no-rendering] [--debug]

Options:
    --n N      N envs per thread. [default: 16]
    --m M      M workers. [default: 16]
    --steps S  Run S environments steps in total. [default: 1000]
'''

docoptsettings = docopt.docopt(docoptstring)
debug = docoptsettings["--debug"]
total_steps = int(docoptsettings["--steps"])
n_workers = int(docoptsettings["--m"]) if not debug else 1
n_envs_per_thread = int(docoptsettings["--n"])
n_envs = n_workers * n_envs_per_thread
render = not docoptsettings["--no-rendering"]

settings = {
            #Project
            "run-id" : "THIRDeye_2_oldarch_rewardshapersignflip_0.8-0_comboreward",

            #Train parameters
            "n_samples_each_update"     : 2048,
            "n_train_epochs_per_update" : 10,
            "minibatch_size"            : 64,
            "time_to_reference_update"  : 4, #How after how many do_training calls do we update the reference-model?
            "value_lr"                  : linear_parameter(1e-5, final_val=1e-6, time_horizon=total_steps),
            "prioritized_replay_alpha"  : constant_parameter(0.7),
            "prioritized_replay_beta"   : linear_parameter(0.5, final_val=1.0, time_horizon=total_steps),
            "experience_replay_size"    : 5*10**5,
            "alternating_models"        : False,
            "time_to_training"          : 10**3,
            "single_policy"             : True,

            #Dithering
            # "dithering_scheme"    : "epsilon",
            # "epsilon"  : exp_parameter(1, decay=5*total_steps),
            "dithering_scheme"    : "adaptive_epsilon",
            "epsilon"  : linear_parameter(2.5, final_val=0.5, time_horizon=total_steps),

            #Reward shaping
            "extra_rewards" : True,
            "extra_reward_ammount" : (0.1,),
            "reward_shaper" :  linear_reshaping,
            "reward_shaper_param" : linear_parameter(0.8, final_val=0, time_horizon=0.8*total_steps),

            #Game settings
            "pieces" : [0,6],
            "game_size" : [22,10],
            "time_elapsed_each_action" : 400,
            #Types
            "env_vector_type"   : tetris_environment_vector,
            "env_type"          : tetris_environment,
            "agent_type"        : vector_agent.vector_agent,
            "trainer_type"      : vector_agent_trainer.vector_agent_trainer,

            #Threading
            "run_standalone"       : docoptsettings["--debug"],
            "n_workers"            : n_workers,
            "n_envs_per_thread"    : n_envs_per_thread,
            "worker_steps"         : total_steps // n_envs,
            "process_patience"     : [0.1,0.1, 10.0], #runner/trainer/process_manager
            "worker_net_on_cpu"    : True,
            "trainer_net_on_cpu"   : False,
            #Communication
            "trainer_thread_save_freq"  : 1000,
            "trainer_thread_backup_freq"  : 50,
            "worker_data_send_fequency" : 150,
            "weight_transfer_frequency" : 1,
            "workers_do_processing"     : True,

            #NN
            "pad_visuals"      : True,
            "peephole_convs"   : False,
            #Preprocessing
            "relative_state"   : True, #This means that both players sees themselves as the player to the left, and the other on the right
            "field_as_image"   : True, #This preserves the 2D structure of the playing field, and keeps them separate from the vector part of the state
            "players_separate" : True, #This keeps each players part of the state separate when passed to the neural net

            #Misc.
            "render"            : render,
            "bar_null_moves"    : True,
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
    process_manager.threads["workers"][0].run()
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


###
# Saving this code for now, but it will be removed later.

# ###########################
#
# #
# ##Init SERIAL
# with tf.Session() as session:
#     env = settings["env_vector_type"](
#                                       1,
#                                       settings["env_type"],
#                                       settings=settings
#                                      )
#     agent = settings["agent_type"](
#                                    1,
#                                    id=0,
#                                    sandbox=settings["env_type"](settings=settings),
#                                    session=session,
#                                    settings=settings,
#                                   )
#     print("Starting serial")
#     T_serial_start = time.time()
#     current_player = 1
#     s = env.get_state()
#     print("Go!")
#     for t in range(total_steps):
#         current_player = 1 - current_player
#         _,a = agent.get_action(s, player=current_player)
#         # a = env.get_random_action()
#         r, ds = env.perform_action(a, player=current_player)
#         s = env.get_state()
#         for i,d in enumerate(ds):
#             if d: env.reset(env=i)
#         if render:
#             env.render(env=0)
#         print("Step {}".format((t+1)))
#     T_serial_stop = time.time()
#     print("Serial done")
#     #Done!
#
# #Show what we collected (parallel)!
# print("[Parallel] Collected {} trajectories:".format(None))
# print("Total: {} collected in {} seconds ({} steps/second)".format(total_steps, T_parallel, total_steps/T_parallel))
# print("[Serial] Collected {} trajectories:".format(None))
# print("Total: {} collected in {} seconds ({} steps/second)".format(total_steps, T_serial_stop - T_serial_start, total_steps/(T_serial_stop - T_serial_start) ))
