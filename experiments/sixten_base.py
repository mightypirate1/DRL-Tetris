from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.vector_agent import vector_agent, vector_agent_trainer
from agents.agent_utils.reward_shapers import *
from aux.parameter import *
import threads.threaded_runner
import tensorflow as tf
settings = {
            #Project
            "run-id" : "SIXten-v1.0",
            "description" : "SIXten is a V-fcn learning agent, operating on a prioritized distributed experience replay, doing k-step value estimates, utilizing the world model provided by the tetris_env!",
            # "render_simulation" : True

            #Train parameters
            "n_step_value_estimates"    : 5,
            "n_samples_each_update"     : 16384,
            "minibatch_size"            : 128,
            "n_train_epochs_per_update" : 1,  #5
            "time_to_reference_update"  : 20, #How after how many do_training calls do we update the reference-model?
            "value_lr"                  : exp_parameter(1e-3, base=10.0, decay=3/total_steps),
            "n_samples_to_start_training" : 40000, #0

            #Exp-replay parameters
            "prioritized_replay_alpha"      : constant_parameter(0.7),
            "prioritized_replay_beta"       : linear_parameter(0.5, final_val=1.0, time_horizon=total_steps),
            "experience_replay_size"        : 2*10**6,
            "experience_replay_sample_mode" : 'rank',

            "alternating_models"        : False,
            "time_to_training"          : 10**3,
            "single_policy"             : True,

            #Dithering
            "train_distriburion"    : "distribution_pareto",
            "action_temperature"  : linear_parameter(1, final_val=4.0, time_horizon=total_steps),
            # "train_distriburion"    : "adaptive_epsilon",
            # "epsilon"  : linear_parameter(8, final_val=0.0, time_horizon=total_steps),
            "optimistic_prios" : 0.0,

            #Rewards
            "gamma"             :  0.98,
            "extra_rewards" : False,
            # "reward_ammount" : (1.0, 0.0,),
            # "reward_shaper" :  linear_reshaping,
            # "reward_shaper_param" : linear_parameter(0.0, final_val=0.0, time_horizon=0.3*total_steps),

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
            "run_standalone"       : False,
            "n_workers"            : 3,
            "n_envs_per_thread"    : 16,
            "worker_steps"         : total_steps // 48,
            "worker_net_on_cpu"    : True,
            "trainer_net_on_cpu"   : False,

            #Communication
            "trainer_thread_save_freq"  : 1000,
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
            "visualencoder_n_filters" : (16,32,32,4),
            "visualencoder_filter_sizes" : ((7,7),(3,3), (3,3), (3,3)),
            "peephole_convs"   : True,
            "visualencoder_poolings" : [2,], #Pooling after layer numbers in this list
            "visualencoder_peepholes" : [0,1,2],
            ###
            "valuenet_n_hidden" : 1,
            "valuenet_hidden_size" : 256,
            "nn_regularizer" : 0.001,
            "nn_output_activation" : tf.nn.tanh,
            # "optimizer" : tf.train.GradientDescentOptimizer,
            "optimizer" : tf.train.AdamOptimizer,

            #Misc.
            "render"            : True,
            "bar_null_moves"    : True,
           }
