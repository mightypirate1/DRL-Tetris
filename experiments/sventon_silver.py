import tensorflow as tf

from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.vector_q_agent import vector_q_agent, vector_q_agent_trainer
from agents.agent_utils.reward_shapers import *
from aux.parameter import *

settings = {
            #Project
            "run-id" : "SVENton-silver_alpha_0",
            # "run-id" : "arch01-silver_slim0",
            "description" : "A res-block based architecture. In theory it's good when you scale up :)",
            "state_processor_separate_piece" : True,
            "q_target_locked_for_other_actions" : False,
            "advantage_type" : "mean",#,"none",#"mean", #"max",
            "q_net_type" : "silver",
            "residual_block_settings" : {
                                            "default" : {
                                                            "n_layers" : 3,
                                                            "n_filters" : 64,
                                                        },
                                        },

            "separate_piece_values" : True,
            "advantage_range" : 0.7,
            "piece_advantage_range" : 0.5,

            # "sparse_value_estimate_filter" : [2,3], #Empty list is no filter
            "render_screen_dims" : (3840,2160), #My screen is huge
            # "render_simulation" : True

            #Train parameters
            "gae_lambda"                : 0.92, #0.95 default
            "n_step_value_estimates"    : 17,
            "n_samples_each_update"     : 16384,
            "minibatch_size"            : 32, #256, #128
            "n_train_epochs_per_update" : 1,
            "time_to_reference_update"  : 5, #How after how many do_training calls do we update the reference-model?
            "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=2/total_steps),

            #Exp-replay parameters
            "prioritized_replay_alpha"      : constant_parameter(0.7),
            "prioritized_replay_beta"       : linear_parameter(0.5, final_val=1.0, time_horizon=total_steps),
            "experience_replay_size"        : 2*10**6,
            "experience_replay_sample_mode" : 'rank',
            # "experience_replay_sample_mode" : 'proportional',

            "alternating_models"        : False,
            "time_to_training"          : 0,#10**3,
            "single_policy"             : True,

            #Dithering
            # "dithering_scheme"    : "pareto_distribution",
            # "action_temperature"  : linear_parameter(1, final_val=3.0, time_horizon=total_steps),
            # "dithering_scheme"    : "adaptive_epsilon",
            # "epsilon"  : linear_parameter(10, final_val=0.0, time_horizon=total_steps),
            "dithering_scheme"    : "epsilon",
            "epsilon"  :   exp_parameter(1.0, base=10.0, decay=2/total_steps),
            "optimistic_prios" : 0.0,

            #Rewards
            "gamma"             :  0.98,
            "extra_rewards" : False,
            # "reward_ammount" : (1.0, 0.5,),
            # "reward_shaper" :  linear_reshaping,
            # "reward_shaper_param" : linear_parameter(0.0, final_val=0.0, time_horizon=0.3*total_steps),

            #Game settings
            "pieces" : [0,],
            "game_size" : [10,6],
            # "game_size" : [10,10],
            # "pieces" : [0,6],
            # "game_size" : [22,10],
            "time_elapsed_each_action" : 400,
            "old_state_dict" : False, #SVENton uses the new one

            #Types
            "env_vector_type"   : tetris_environment_vector,
            "env_type"          : tetris_environment,
            "agent_type"        : vector_q_agent.vector_q_agent,
            "trainer_type"      : vector_q_agent_trainer.vector_q_agent_trainer,

            #Threading
            "run_standalone"       : False,
            "n_workers"            : 3,
            "n_envs_per_thread"    : 80,
            "worker_steps"         : total_steps // 240,
            "worker_net_on_cpu"    : True,
            "trainer_net_on_cpu"   : False,

            #Communication
            "trainer_thread_save_freq"  : 100,
            "trainer_thread_backup_freq"  : 1,
            "worker_data_send_fequency" : 1,
            "weight_transfer_frequency" : 1,
            "workers_do_processing"     : True,

            #Misc.
            "render"            : True,
            "bar_null_moves"    : True,
            "resblock_dropout" : 0.15,
            "nn_regularizer" : 0.0001,
            "nn_output_activation" : tf.nn.tanh,
            # "optimizer" : tf.train.GradientDescentOptimizer,
            "optimizer" : tf.train.AdamOptimizer,
           }
###
###
###

# patches = \
# [
#     {
#         "run-id" : "arch02-silver",
#         "residual_block_settings" : {
#                                         "default" : {
#                                                         "n_layers" : 2,
#                                                         "n_filters" : 64,
#                                                     },
#                                     },
#     },
#     {
#         "run-id" : "arch03-silver",
#         "residual_block_settings" : {
#                                         "default" : {
#                                                         "n_layers" : 3,
#                                                         "n_filters" : 64,
#                                                     },
#                                     },
#     },
#     {
#         "run-id" : "arch04-silver",
#         "residual_block_settings" : {
#                                         "default" : {
#                                                         "n_layers" : 1,
#                                                         "n_filters" : 64,
#                                                     },
#                                     },
#     },
#
#         ##################
#         {
#             "run-id" : "arch01-slim-silver",
#             "residual_block_settings" : {
#                                             "default" : {
#                                                             "n_layers" : 4,
#                                                             "n_filters" : 64,
#                                                         },
#                                             "val_stream" : {
#                                                             "n_filters" : 256,
#                                                             }
#                                         },
#         },
#         {
#             "run-id" : "arch02-slim-silver",
#             "residual_block_settings" : {
#                                             "default" : {
#                                                             "n_layers" : 2,
#                                                             "n_filters" : 64,
#                                                         },
#                                         },
#         },
#         {
#             "run-id" : "arch03-slim-silver",
#             "residual_block_settings" : {
#                                             "default" : {
#                                                             "n_layers" : 3,
#                                                             "n_filters" : 64,
#                                                         },
#                                         },
#         },
#         {
#             "run-id" : "arch04-slim-silver",
#             "residual_block_settings" : {
#                                             "default" : {
#                                                             "n_layers" : 1,
#                                                             "n_filters" : 64,
#                                                         },
#                                         },
#
#         },
    # {
    #     'run-id' : "arch01-silver_06",
    #     'pieces' : [0,6],
    # },
    #
    #
    # {
    #    'run-id' : "arch01-silver_big06",
    #    'game_size' : [22,10],
    # },
    #
    # {
    #     'run-id' : "arch01-silver_big0to7",
    #     'pieces' : [0,1,2,3,4,5,6],
    # },
# ]
