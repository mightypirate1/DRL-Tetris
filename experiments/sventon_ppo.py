import tensorflow as tf

from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.sventon_agent import sventon_agent, sventon_agent_ppo_trainer, sventon_agent_dqn_trainer
from agents.agent_utils.reward_shapers import *
from aux.parameter import *

settings = {
            #Project
            "run-id" : "Gr8-Q01",
            "presets" : ["default", "sventon", "sventon_ppo", "resblock"],
            "worker_net_on_cpu" : False,
            # "separate_piece_values" : False,

            #RL-algo-settings
            "ppo_parameters" : {
                                'clipping_parameter' : 0.1,
                                'value_loss' : 1.0,
                                'policy_loss' : 0.2,
                                'entropy_loss' : 0.002,

                                'negative_dampener' : 1.0,
                                'entropy_floor_loss' : 0.0,
                                },
            "value_estimator_params" : {
                                        "truncate_aggregation" : False,
                                        },
            "ppo_epsilon" : constant_parameter(0.05),
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
            "n_step_value_estimates"    : 23,
            "sparse_value_estimate_filter" : [2,3,11,19], #Empty list is no filter
            "n_samples_each_update"     : 4096,
            "minibatch_size"            : 64, #256, #128
            "n_train_epochs_per_update" : 1,
            "time_to_reference_update"  : 1, #How after how many do_training calls do we update the reference-model?
            "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
            #Game settings
            # "game_size" : [10,6],
            # "pieces" : [2,3,4,5],

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
# {
# "run-id" : "Gr8-Q02k1",
# #RL-algo-settings
# "ppo_parameters" : {
# 'clipping_parameter' : 0.1,
# 'value_loss' : 1.0,
# 'policy_loss' : 0.2,
# 'entropy_loss' : 0.002,
#
# 'negative_dampener' : 1.0,
# 'entropy_floor_loss' : 0.0,
# },
# "value_estimator_params" : {
# "truncate_aggregation" : False,
# },
# "n_step_value_estimates" : 1,
# "n_train_epochs_per_update" : 3,
# "minibatch_size"            : 64,
# "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
# },
#     {
#         "run-id" : "Gr8-Q02k5",
#         #RL-algo-settings
#         "ppo_parameters" : {
#                             'clipping_parameter' : 0.1,
#                             'value_loss' : 1.0,
#                             'policy_loss' : 0.2,
#                             'entropy_loss' : 0.002,
#
#                             'negative_dampener' : 1.0,
#                             'entropy_floor_loss' : 0.0,
#                             },
#         "value_estimator_params" : {
#                                     "truncate_aggregation" : False,
#                                     },
#         "n_step_value_estimates" : 5,
#         "n_train_epochs_per_update" : 2,
#         "minibatch_size"            : 64,
#         "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
#     },
    # {
    #     "run-id" : "Gr8-Q03",
    #     #RL-algo-settings
    #     "ppo_parameters" : {
    #                         'clipping_parameter' : 0.1,
    #                         'value_loss' : 1.0,
    #                         'policy_loss' : 0.2,
    #                         'entropy_loss' : 0.002,
    #
    #                         'negative_dampener' : 1.0,
    #                         'entropy_floor_loss' : 0.0,
    #                         },
    #     "n_step_value_estimates" : 23,
    #     "value_estimator_params" : {
    #                                 "truncate_aggregation" : False,
    #                                 },
    #     "minibatch_size"            : 32,
    #     "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
    # },
    # {
    #     "run-id" : "Gr8-Q04",
    #     #RL-algo-settings
    #     "ppo_parameters" : {
    #                         'clipping_parameter' : 0.1,
    #                         'value_loss' : 1.0,
    #                         'policy_loss' : 0.2,
    #                         'entropy_loss' : 0.002,
    #
    #                         'negative_dampener' : 1.0,
    #                         'entropy_floor_loss' : 0.0,
    #                         },
    #     "value_estimator_params" : {
    #                                 "truncate_aggregation" : False,
    #                                 },
    #     "minibatch_size"            : 32,
    #     "value_lr"                  : exp_parameter(3e-5, base=10.0, decay=1/total_steps),
    # },
    # {
    #     "run-id" : "Gr8-Q05",
    #     #RL-algo-settings
    #     "ppo_parameters" : {
    #                         'clipping_parameter' : 0.1,
    #                         'value_loss' : 1.0,
    #                         'policy_loss' : 0.2,
    #                         'entropy_loss' : 0.002,
    #
    #                         'negative_dampener' : 1.0,
    #                         'entropy_floor_loss' : 0.0,
    #                         },
    #     "value_estimator_params" : {
    #                                 "truncate_aggregation" : False,
    #                                 },
    #     "minibatch_size"            : 128,
    #     "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
    # },

    # {
    #     "run-id" : "Gr8-Q06",
    #     #RL-algo-settings
    #     "ppo_parameters" : {
    #                         'clipping_parameter' : 0.1,
    #                         'value_loss' : 1.0,
    #                         'policy_loss' : 0.15,
    #                         'entropy_loss' : 0.002,
    #
    #                         'negative_dampener' : 1.0,
    #                         'entropy_floor_loss' : 0.0,
    #                         },
    #     "value_estimator_params" : {
    #                                 "truncate_aggregation" : False,
    #                                 },
    #     "minibatch_size"            : 64,
    #     "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
    # },

    {
        "run-id" : "Gr8-Q07",
        #RL-algo-settings
        "ppo_parameters" : {
                            'clipping_parameter' : 0.1,
                            'value_loss' : 1.0,
                            'policy_loss' : 0.2,
                            'entropy_loss' : 0.002,

                            'negative_dampener' : 1.0,
                            'entropy_floor_loss' : 2.0,
                            },
        "value_estimator_params" : {
                                    "truncate_aggregation" : False,
                                    },
        "ppo_epsilon" : constant_parameter(0.07),
    },
    {
        "run-id" : "Gr8-Q08dampfl2",
        #RL-algo-settings
        "ppo_parameters" : {
                            'clipping_parameter' : 0.1,
                            'value_loss' : 1.0,
                            'policy_loss' : 0.2,
                            'entropy_loss' : 0.002,

                            'negative_dampener' : 0.5,
                            'entropy_floor_loss' : 0.0,
                            },
        "value_estimator_params" : {
                                    "truncate_aggregation" : False,
                                    },
        "ppo_epsilon" : constant_parameter(0.05),
    },
    {
        "run-id" : "Gr8-Q09trnk",
        #RL-algo-settings
        "ppo_parameters" : {
                            'clipping_parameter' : 0.1,
                            'value_loss' : 1.0,
                            'policy_loss' : 0.2,
                            'entropy_loss' : 0.002,

                            'negative_dampener' : 1.0,
                            'entropy_floor_loss' : 0.0,
                            },
        "value_estimator_params" : {
                                    "truncate_aggregation" : True,
                                    },
        "ppo_epsilon" : constant_parameter(0.05),
    },
]
