import tensorflow as tf

from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.sventon_agent import sventon_agent, sventon_agent_ppo_trainer, sventon_agent_dqn_trainer
from agents.agent_utils.reward_shapers import *
from aux.parameter import *

settings = {
            #Project
            "run-id" : "ELK-Z03",
            "presets" : ["default", "sventon", "sventon_ppo", "resblock"],

            #RL-algo-settings
            "ppo_parameters" : {
                                'clipping_parameter' : 0.2,
                                'value_loss' : 1.0,
                                'policy_loss' : 1.0,
                                'entropy_loss' : 0.01,
                                'negative_dampener' : 1.0,
                                },
            #Architecture
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

            #Train parameters
            "gae_lambda"                : 0.96, #0.95 default
            "n_step_value_estimates"    : 37,
            "sparse_value_estimate_filter" : [2,3], #Empty list is no filter
            "n_samples_each_update"     : 4096,
            "minibatch_size"            : 256, #256, #128
            "n_train_epochs_per_update" : 3,
            "time_to_reference_update"  : 1, #How after how many do_training calls do we update the reference-model?
            "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),

            #Game settings
            # "game_size" : [10,6],
            "pieces" : [0,1,2,3,4,5,6],
            "game_size" : [22,10],

            #Threading
            "n_workers"            : 3,
            "n_envs_per_thread"    : 80,
            "worker_steps"         : total_steps // 240,
            #Misc
            "render_screen_dims" : (3840,2160),
           }
###
###
###

patches = \
[
    {
    "run-id" : "ELK-Z04",
    "ppo_parameters" : {
                        'clipping_parameter' : 0.2,
                        'value_loss' : 1.0,
                        'policy_loss' : 0.5,
                        'entropy_loss' : 0.01,
                        'negative_dampener' : 1.0,
                        },
    },
    {
    "run-id" : "ELK-Z05",
    "ppo_parameters" : {
                        'clipping_parameter' : 0.2,
                        'value_loss' : 0.5,
                        'policy_loss' : 1.0,
                        'entropy_loss' : 0.01,
                        'negative_dampener' : 1.0,
                        },
    },
    {
    "run-id" : "ELK-Z06",
    "ppo_parameters" : {
                        'clipping_parameter' : 0.2,
                        'value_loss' : 0.5,
                        'policy_loss' : 1.0,
                        'entropy_loss' : 0.02,
                        'negative_dampener' : 1.0,
                        },
    },
    {
    "run-id" : "ELK-Z07",
    "ppo_parameters" : {
                        'clipping_parameter' : 0.2,
                        'value_loss' : 0.5,
                        'policy_loss' : 1.0,
                        'entropy_loss' : 0.01,
                        'negative_dampener' : 0.5,
                        },
    },
    {
    "run-id" : "ELK-Z08",
    "ppo_parameters" : {
                        'clipping_parameter' : 0.2,
                        'value_loss' : 0.5,
                        'policy_loss' : 1.0,
                        'entropy_loss' : 0.01,
                        'negative_dampener' : 1.0,
                        },
    "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=2/total_steps),
    },
    # {
    # "run-id" : "ELK-X03-fullset0",
    # "pieces" : [0,1,2,3,4,5,6],
    # "ppo_parameters" : {
    #                     'clipping_parameter' : 0.2,
    #                     'value_loss' : 1.0,
    #                     'policy_loss' : 1.0,
    #                     'entropy_loss' : 0.01,
    #                     'negative_dampener' : 1.0,
    #                    },
    # "value_lr"                  : exp_parameter(5e-4, base=10.0, decay=2/total_steps),
    # },
]
