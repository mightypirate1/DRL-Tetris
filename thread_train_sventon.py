from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.vector_agent import vector_agent, vector_agent_trainer
from agents.vector_q_agent import vector_q_agent, vector_q_agent_trainer
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
  thread_train.py [options]
  thread_train.py restart <file> <clock> [options]

Options:
    --n N           N envs per thread. [default: 128]
    --m M           M workers. [default: 4]
    --steps S       Run S environments steps in total. [default: 1000000]
    --no-rendering  Disables rendering.
    --debug         Runs a single thread doing both data-gathering and training.
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
            "run-id" : "SVENton-Q01-k37-pieceAdv",
            "state_processor_separate_piece" : True,
            "old_state_dict" : False,
            "keyboard_conv" : True,
            "keyboard_range" : 0.7,
            "keyboard_separate_piecevalues" : True,

            "render_screen_dims" : (3840,2160), #My screen is huge
            # "render_simulation" : True

            #Train parameters
            "gae_lambda"                : 0.95, #0.95 default
            "n_step_value_estimates"    : 37,
            "n_samples_each_update"     : 16384,
            # "n_samples_each_update"     : 8192,
            "minibatch_size"            : 128, #128
            "n_train_epochs_per_update" : 1,  #5
            "time_to_reference_update"  : 5, #How after how many do_training calls do we update the reference-model?
            "value_lr"                  : exp_parameter(1e-3, base=10.0, decay=2/total_steps),
            # "n_samples_to_start_training" : 40000, #0

            #Exp-replay parameters
            "prioritized_replay_alpha"      : constant_parameter(0.7),
            "prioritized_replay_beta"       : linear_parameter(0.5, final_val=1.0, time_horizon=total_steps),
            "experience_replay_size"        : 2*10**6,
            "experience_replay_sample_mode" : 'rank',
            # "experience_replay_sample_mode" : 'proportional',

            "alternating_models"        : False,
            "time_to_training"          : 10**3,
            "single_policy"             : True,

            #Dithering
            "dithering_scheme"    : "pareto_distribution",
            "action_temperature"  : linear_parameter(1, final_val=4.0, time_horizon=total_steps),
            # "dithering_scheme"    : "adaptive_epsilon",
            # "epsilon"  : linear_parameter(8, final_val=0.0, time_horizon=total_steps),
            "optimistic_prios" : 0.0,

            #Rewards
            "gamma"             :  0.98,
            "extra_rewards" : False,
            # "reward_ammount" : (1.0, 0.5,),
            # "reward_shaper" :  linear_reshaping,
            # "reward_shaper_param" : linear_parameter(0.0, final_val=0.0, time_horizon=0.3*total_steps),

            #Game settings
            # "pieces" : [6],
            # "game_size" : [10,5],
            "pieces" : [0,6],
            "game_size" : [22,10],
            "time_elapsed_each_action" : 400,
            #Types
            "env_vector_type"   : tetris_environment_vector,
            "env_type"          : tetris_environment,
            "agent_type"        : vector_q_agent.vector_q_agent,
            "trainer_type"      : vector_q_agent_trainer.vector_q_agent_trainer,

            #Threading
            "run_standalone"       : docoptsettings["--debug"],
            "n_workers"            : n_workers,
            "n_envs_per_thread"    : n_envs_per_thread,
            "worker_steps"         : total_steps // n_envs,
            "worker_net_on_cpu"    : not docoptsettings["--debug"],
            "trainer_net_on_cpu"   : False,

            #Communication
            "trainer_thread_save_freq"  : 100,
            "trainer_thread_backup_freq"  : 10,
            "worker_data_send_fequency" : 5,
            "weight_transfer_frequency" : 1,
            "workers_do_processing"     : True,

            #Value net:
            "vectorencoder_n_hidden" : 1,
            "vectorencoder_hidden_size" : 256,
            "vectorencoder_output_size" : 32,
            ###
            "pad_visuals"      : True,
            "visualencoder_n_convs" : 4,
            "visualencoder_n_filters" : (64,32,32,32),
            "visualencoder_filter_sizes" : ((3,3),(5,5),(3,3), (3,3),),
            "peephole_convs"   : False,
            # "visualencoder_poolings" : [], #Pooling after layer numbers in this list
            # "visualencoder_peepholes" : [0,1,2],
            ###
            "valuenet_n_hidden" : 1,
            "valuenet_hidden_size" : 512,
            "nn_regularizer" : 0.001,
            "nn_output_activation" : tf.nn.tanh,
            # "optimizer" : tf.train.GradientDescentOptimizer,
            "optimizer" : tf.train.AdamOptimizer,

            #Misc.
            "render"            : render,
            "bar_null_moves"    : True,
           }

print("Training script executed with settings:")
for x in settings:
    print("\t{} : {}".format(x,settings[x]))
if docoptsettings['restart']:
    restart_file  =     docoptsettings["<file>" ]
    restart_clock = int(docoptsettings["<clock>"])
    print("Restarting training...\nFile: {}\nClock: {}".format(restart_file,restart_clock))
else:
    restart_file, restart_clock = None, 0

process_manager = threads.threaded_runner.threaded_runner(settings=settings, restart=(restart_file, restart_clock))

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
print("Creating processes...")
process_manager.run(total_steps // n_envs )
process_manager.join()
T_parallel = process_manager.get_avg_runtime()
sys.stdout.flush()
print("All processes finished.")
