import tensorflow as tf

from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.vector_ppo_agent import vector_ppo_agent, vector_ppo_agent_trainer
from agents.agent_utils.reward_shapers import *
from aux.parameter import *

settings = {
            #Project
            # "run-id" : "SVENton-silver_alpha_0",
            "visual_stack" : None,
            "run-id" : "ppoparam3-0-separate",
            "description" : "A res-block based architecture, trained by ppo.",
            "state_processor_separate_piece" : True,
            "q_target_locked_for_other_actions" : False,
            "advantage_type" : "mean",#,"none",#"mean", #"max",
            "q_net_type" : "silver", #it says q, but it's usable for ppo too
            "residual_block_settings" : {
                                            "default" : {
                                                            "n_layers" : 3,
                                                            "n_filters" : 64,
                                                        },
                                            "val_stream" : {
                                                            "n_layers" : 5,
                                                            "n_filters" : 64,
                                                            }
                                        },

            "eval_distribution" : "pi",
            "ppo_parameters" : {
                                'clipping_parameter' : 0.1,
                                'value_loss' : 1.0,
                                'policy_loss' : 0.3,
                                'entropy_loss' : 0.02,
                                },

            "separate_piece_values" : True,
            "advantage_range" : 0.5,
            "piece_advantage_range" : 0.5,

            "render_screen_dims" : (3840,2160), #My screen is huge
            # "render_simulation" : True

            #Train parameters
            "gae_lambda"                : 0.92, #0.95 default
            "n_step_value_estimates"    : 13,
            # "sparse_value_estimate_filter" : [2,3,5], #Empty list is no filter
            # "n_samples_each_update"     : 16384,
            "n_samples_each_update"     : 4096,
            "minibatch_size"            : 256, #256, #128
            "n_train_epochs_per_update" : 2,
            "time_to_reference_update"  : 1, #How after how many do_training calls do we update the reference-model?
            "value_lr"                  : exp_parameter(5e-4, base=10.0, decay=1/total_steps),

            "single_policy"             : True,

            #Rewards
            "gamma"             :  0.98,
            "extra_rewards" : False,
            # "reward_ammount" : (1.0, 0.5,),
            # "reward_shaper" :  linear_reshaping,
            # "reward_shaper_param" : linear_parameter(0.0, final_val=0.0, time_horizon=0.3*total_steps),

            #Game settings
            # "pieces" : [0,],
            "pieces" : [5,],
            "game_size" : [10,6],
            # "game_size" : [10,10],
            # "pieces" : [0,6],
            # "game_size" : [22,10],
            "time_elapsed_each_action" : 400,
            "old_state_dict" : False, #SVENton uses the new one

            #Types
            "env_vector_type"   : tetris_environment_vector,
            "env_type"          : tetris_environment,
            "agent_type"        : vector_ppo_agent.vector_ppo_agent,
            "trainer_type"      : vector_ppo_agent_trainer.vector_ppo_agent_trainer,

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

patches = \
[
    {
    "run-id" : "ppoparam3-0",
    "separate_piece_values" : False,
    },
#     {
#     "run-id" : "ppoparam2-1",
#     "separate_piece_values" : False,
#     # "sparse_value_estimate_filter" : [2,3,5], #Empty list is no filter
#     "value_lr"                  : exp_parameter(1e-5, base=10.0, decay=1/total_steps),
#     },
#     {
#     "run-id" : "ppoparam2-2",
#     "separate_piece_values" : False,
#     # "sparse_value_estimate_filter" : [2,3,5], #Empty list is no filter
#     "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=2/total_steps),
#     },
#     {
#     "run-id" : "ppoparam2-3",
#     "separate_piece_values" : False,
#     # "sparse_value_estimate_filter" : [2,3,5], #Empty list is no filter
#     "value_lr"                  : exp_parameter(1e-3, base=10.0, decay=2/total_steps),
#     },
#     {
#     "run-id" : "ppoparam2-4",
#     "separate_piece_values" : False,
#     # "sparse_value_estimate_filter" : [2,3,5], #Empty list is no filter
#     "value_lr"                  : exp_parameter(5e-4, base=10.0, decay=1/total_steps),
#     },
#     {
#     "run-id" : "ppoparam2-5",
#     "separate_piece_values" : False,
#     # "sparse_value_estimate_filter" : [2,3,5], #Empty list is no filter
#     "value_lr"                  : exp_parameter(5e-4, base=10.0, decay=2/total_steps),
#     },
#     {
#     "run-id" : "ppoparam2-6-4rerunSep",
#     "separate_piece_values" : True,
#     # "sparse_value_estimate_filter" : [2,3,5], #Empty list is no filter
#     "value_lr"                  : exp_parameter(5e-4, base=10.0, decay=1/total_steps),
#     },
#     {
#     "run-id" : "ppoparam2-6-4rerunNonSep",
#     "separate_piece_values" : False,
#     # "sparse_value_estimate_filter" : [2,3,5], #Empty list is no filter
#     "value_lr"                  : exp_parameter(5e-4, base=10.0, decay=1/total_steps),
#     },
#     {
#     "run-id" : "ppoparam2-7-4rerunNonSep",
#     "separate_piece_values" : False,
#     # "sparse_value_estimate_filter" : [2,3,5], #Empty list is no filter
#     "value_lr"                  : exp_parameter(5e-4, base=10.0, decay=1/total_steps),
#     },
]
