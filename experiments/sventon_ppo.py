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

            #Noise
            "parameter_noise" : {
                                    "type" : "multiplicative_gaussian",
                                    "std_dev" : 0.1,
                                },

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
