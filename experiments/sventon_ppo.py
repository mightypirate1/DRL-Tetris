import tensorflow as tf
from tools.parameter import *

settings = {
            #Project
            "augment_data" : False,
            "normalize_advantages" : True,
            "run-id" : "I-Z07",
            "trainer_thread_save_freq" : 1000,
            "presets" : ["default", "sventon", "sventon_ppo", "resblock"],
            "n_step_value_estimates"    : 1,


            #RL-algo-settings
            "ppo_parameters" : {
                    'clipping_parameter' : 0.10,
                    'value_loss' : 0.4,
                    'policy_loss' : 0.9,

                    'entropy_loss' : linear_parameter(0.05, final_val=0.0, time_horizon=2e8, max=0.027),
                    'entropy_floor_loss' : 0.0,
                    },
            "nn_regularizer" : 5e-6,
            # "resblock_dropout" : 0.08,


            #Train parameters
            "value_lr"                  : exp_parameter(exp_parameter(3e-5, base=10.0, decay=1/17e6, min=2e-6), base=10.0, decay=1/2e8),
            "gae_lambda"                : 0.87, #0.95 default
            "n_samples_each_update"     : 4096,
            "minibatch_size"            : 256,
            "n_train_epochs_per_update" : 1,

            #Architecture
            "residual_block_settings" : {
                                            "default" : {
                                                            "n_layers" : 5,
                                                            "n_filters" : 64,
                                                            "normalization" : "layer",
                                                            # "filter_size" : (3,3),
                                                        },
                                            "val_stream" : {
                                                            "n_layers" : 6,
                                                            "n_filters" : 128,
                                                            "filter_size" : (5,5),
                                                            }
                                        },


            #Game settings
            # "game_size" : [10,6],
            # "pieces" : [2,3,4,5],

            #Threading
            "n_workers"            : 3,
            "n_envs_per_thread"    : 80,
            "worker_steps"         : total_steps // 3*80,
            #Misc
            "render_screen_dims" : (3840,2160),
           }
###
###
###

patches = \
[

# {
#     "run-id" : "I-Z06g-lambda87",
#     #RL-algo-settings
#     "ppo_parameters" : {
#             'clipping_parameter' : 0.10,
#             'value_loss' : 0.4,
#             'policy_loss' : 0.9,
#
#             'entropy_loss' : linear_parameter(0.1, final_val=0.0, time_horizon=2e8, max=0.06),
#             'entropy_floor_loss' : 0.0,
#             },
# },

# {
#     "run-id" : "I-Z04b",
#     "nn_regularizer" : 5e-6,
#     "gae_lambda" : 0.87,
#     "n_samples_each_update"     : 4096,
#     "value_lr"                  : exp_parameter(exp_parameter(3e-5, base=10.0, decay=1/17e6, min=1e-5), base=10.0, decay=1/2e8),
#     #RL-algo-settings
#     "ppo_parameters" : {
#             'clipping_parameter' : 0.10,
#             'value_loss' : 0.4,
#             'policy_loss' : 0.9,
#
#             'entropy_loss' : linear_parameter(0.03, final_val=0.0, time_horizon=2e8, max=0.0025),
#             'entropy_floor_loss' : 0.0,
#             },
# },
# {
#     "run-id" : "I-Z04c",
#     "nn_regularizer" : 5e-6,
#     "gae_lambda" : 0.87,
#     "n_samples_each_update"     : 4096,
#     "value_lr"                  : exp_parameter(exp_parameter(3e-5, base=10.0, decay=1/17e6, min=1e-5), base=10.0, decay=1/2e8),
#     #RL-algo-settings
#     "ppo_parameters" : {
#             'clipping_parameter' : 0.10,
#             'value_loss' : 0.4,
#             'policy_loss' : 0.9,
#
#             'entropy_loss' : linear_parameter(0.04, final_val=0.0, time_horizon=2e8, max=0.0035),
#             'entropy_floor_loss' : 0.0,
#             },
# },
# {
#     "run-id" : "I-Z04d",
#     "nn_regularizer" : 5e-6,
#     "gae_lambda" : 0.87,
#     "n_samples_each_update"     : 4096,
#     "value_lr"                  : exp_parameter(exp_parameter(1e-5, base=10.0, decay=1/17e6, min=1e-5), base=10.0, decay=1/2e8),
#     #RL-algo-settings
#     "ppo_parameters" : {
#             'clipping_parameter' : 0.10,
#             'value_loss' : 0.4,
#             'policy_loss' : 0.9,
#
#             'entropy_loss' : linear_parameter(0.02, final_val=0.0, time_horizon=2e8, max=0.0015),
#             'entropy_floor_loss' : 0.0,
#             },
# },
# {
#     "run-id" : "I-Z04e",
#     "nn_regularizer" : 5e-6,
#     "gae_lambda" : 0.87,
#     "n_samples_each_update"     : 4096,
#     "value_lr"                  : exp_parameter(exp_parameter(3e-5, base=10.0, decay=1/17e6, min=1e-5), base=10.0, decay=1/2e8),
#     #RL-algo-settings
#     "ppo_parameters" : {
#             'clipping_parameter' : 0.10,
#             'value_loss' : 0.4,
#             'policy_loss' : 0.9,
#
#             'entropy_loss' : linear_parameter(0.02, final_val=0.0, time_horizon=2e8, max=0.0015),
#             'entropy_floor_loss' : 10.0,
#             },
# },
# {
#     "run-id" : "I-Z04f",
#     "nn_regularizer" : 5e-6,
#     "gae_lambda" : 0.87,
#     "n_samples_each_update"     : 4096,
#     "value_lr"                  : exp_parameter(exp_parameter(3e-5, base=10.0, decay=1/17e6, min=1e-5), base=10.0, decay=1/2e8),
#     #RL-algo-settings
#     "ppo_parameters" : {
#             'clipping_parameter' : 0.10,
#             'value_loss' : 0.4,
#             'policy_loss' : 0.9,
#
#             'entropy_loss' : linear_parameter(0.05, final_val=0.0, time_horizon=2e8, max=0.0045),
#             'entropy_floor_loss' : 0.0,
#             },
# },
# {
#     "run-id" : "I-Z04g",
#     "nn_regularizer" : 5e-6,
#     "gae_lambda" : 0.87,
#     "n_samples_each_update"     : 4096,
#     "value_lr"                  : exp_parameter(exp_parameter(3e-5, base=10.0, decay=1/17e6, min=1e-5), base=10.0, decay=1/2e8),
#     #RL-algo-settings
#     "ppo_parameters" : {
#             'clipping_parameter' : 0.10,
#             'value_loss' : 0.4,
#             'policy_loss' : 0.9,
#
#             'entropy_loss' : linear_parameter(0.07, final_val=0.0, time_horizon=2e8, max=0.0065),
#             'entropy_floor_loss' : 0.0,
#             },
# },
# {
#     "run-id" : "I-Z04h",
#     "nn_regularizer" : 5e-6,
#     "gae_lambda" : 0.87,
#     "n_samples_each_update"     : 4096,
#     "value_lr"                  : exp_parameter(exp_parameter(3e-5, base=10.0, decay=1/17e6, min=1e-5), base=10.0, decay=1/2e8),
#     #RL-algo-settings
#     "ppo_parameters" : {
#             'clipping_parameter' : 0.10,
#             'value_loss' : 0.4,
#             'policy_loss' : 0.9,
#
#             'entropy_loss' : linear_parameter(0.09, final_val=0.0, time_horizon=2e8, max=0.0085),
#             'entropy_floor_loss' : 0.0,
#             },
# },
# {
#     "run-id" : "I-Z04i",
#     "nn_regularizer" : 5e-6,
#     "gae_lambda" : 0.87,
#     "n_samples_each_update"     : 4096,
#     "value_lr"                  : exp_parameter(exp_parameter(3e-5, base=10.0, decay=1/17e6, min=1e-5), base=10.0, decay=1/2e8),
#     #RL-algo-settings
#     "ppo_parameters" : {
#             'clipping_parameter' : 0.10,
#             'value_loss' : 0.4,
#             'policy_loss' : 0.9,
#
#             'entropy_loss' : linear_parameter(0.13, final_val=0.0, time_horizon=2e8, max=0.00125),
#             'entropy_floor_loss' : 0.0,
#             },
# },

# {
#         "run-id" : "I-Q06-lambda95",
#         "value_lr"                  : exp_parameter(exp_parameter(3e-5, base=10.0, decay=1/17e6, min=1e-5), base=10.0, decay=1/2e8),
#         "gae_lambda" : 0.95,
# },
# {
#         "run-id" : "I-Q06-lambda85",
#         "value_lr"                  : exp_parameter(exp_parameter(3e-5, base=10.0, decay=1/17e6, min=1e-5), base=10.0, decay=1/2e8),
#         "gae_lambda" : 0.85,
# },
# {
#         "run-id" : "I-Q06-lambda75-entropy026",
#         "value_lr"                  : exp_parameter(exp_parameter(3e-5, base=10.0, decay=1/17e6, min=1e-5), base=10.0, decay=1/2e8),
#         "gae_lambda" : 0.75,
#         "ppo_parameters" : {
#                 'clipping_parameter' : 0.10,
#                 'value_loss' : 0.4,
#                 'policy_loss' : 0.9,
#                 'entropy_loss' : linear_parameter(0.024, final_val=0.0, time_horizon=2e8, max=0.026),
#                 'entropy_floor_loss' : 0.0,
#                 },
# },
# {
#         "run-id" : "I-Q06-lambda65",
#         "value_lr"                  : exp_parameter(exp_parameter(3e-5, base=10.0, decay=1/17e6, min=1e-5), base=10.0, decay=1/2e8),
#         "gae_lambda" : 0.65,
# },
# {
#         "run-id" : "I-Q06-lambda55",
#         "value_lr"                  : exp_parameter(exp_parameter(3e-5, base=10.0, decay=1/17e6, min=1e-5), base=10.0, decay=1/2e8),
#         "gae_lambda" : 0.55,
# },
]
