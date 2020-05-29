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
            "run-id" : "ELK-rtp-0",
            "description" : "A res-block based architecture, trained by ppo.",
            #
            "state_processor_separate_piece" : True,
            "state_processor_piece_in_statevec" : False,
            #
            "q_target_locked_for_other_actions" : False,
            "advantage_type" : "mean",#,"none",#"mean", #"max",
            "architecture" : "silver",
            "residual_block_settings" : {
                                            "default" : {
                                                            "n_layers" : 3,
                                                            "n_filters" : 64,
                                                            "normalization" : "layer",
                                                        },
                                            "val_stream" : {
                                                            "n_layers" : 4,
                                                            "n_filters" : 64,
                                                            }
                                        },

            "eval_distribution" : "pi",
            "ppo_parameters" : {
                                'clipping_parameter' : 0.05,
                                'value_loss' : 1.0,
                                'policy_loss' : 1.0,
                                'entropy_loss' : 0.01,
                                'negative_dampener' : 1.0,
                                },

            "separate_piece_values" : True,
            "advantage_range" : 0.5,
            "piece_advantage_range" : 0.5,

            "render_screen_dims" : (3840,2160), #My screen is huge
            # "render_simulation" : True

            #Train parameters
            "gae_lambda"                : 0.96, #0.95 default
            "n_step_value_estimates"    : 37,
            "sparse_value_estimate_filter" : [2,3], #Empty list is no filter
            # "n_samples_each_update"     : 16384,
            "n_samples_each_update"     : 4096,
            "minibatch_size"            : 256, #256, #128
            "n_train_epochs_per_update" : 3,
            "time_to_reference_update"  : 1, #How after how many do_training calls do we update the reference-model?
            "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
            "single_policy"             : True,

            #Rewards
            "gamma"             :  0.98,
            "extra_rewards" : False,
            # "reward_ammount" : (1.0, 0.5,),
            # "reward_shaper" :  linear_reshaping,
            # "reward_shaper_param" : linear_parameter(0.0, final_val=0.0, time_horizon=0.3*total_steps),

            #Game settings
            # "pieces" : [0,],
            # "pieces" : [5,],
            # "game_size" : [10,6],
            # "game_size" : [10,10],
            "pieces" : [2,3,4,5],
            "game_size" : [22,10],
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
            "resblock_dropout" : 0.0,#0.15,
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
    "run-id" : "ELK-rtp-1",
    "ppo_parameters" : {
                        'clipping_parameter' : 0.1,
                        'value_loss' : 1.0,
                        'policy_loss' : 1.0,
                        'entropy_loss' : 0.01,
                        'negative_dampener' : 1.0,
                        },
    },
    {
    "run-id" : "ELK-rtp-2",
    "ppo_parameters" : {
                        'clipping_parameter' : 0.05,
                        'value_loss' : 1.0,
                        'policy_loss' : 1.0,
                        'entropy_loss' : 0.01,
                        'negative_dampener' : 0.5,
                        },
    },
    {
    "run-id" : "ELK-rtp-3",
    "ppo_parameters" : {
                        'clipping_parameter' : 0.05,
                        'value_loss' : 1.0,
                        'policy_loss' : 0.5,
                        'entropy_loss' : 0.01,
                        'negative_dampener' : 1.0,
                        },
    },
    {
    "run-id" : "ELK-rtp-4",
    "ppo_parameters" : {
                        'clipping_parameter' : 0.05,
                        'value_loss' : 0.5,
                        'policy_loss' : 1.0,
                        'entropy_loss' : 0.01,
                        'negative_dampener' : 1.0,
                        },
    },
    {
    "run-id" : "ELK-rtp-5",
    "ppo_parameters" : {
                        'clipping_parameter' : 0.1,
                        'value_loss' : 1.0,
                        'policy_loss' : 1.0,
                        'entropy_loss' : 0.01,
                        'negative_dampener' : 0.5,
                        },
    },
    # {
    # "run-id" : "ELK-rtp-0-1piece_net",
    # "state_processor_piece_in_statevec" : True,
    # },

]
