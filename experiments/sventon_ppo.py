import tensorflow as tf

from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.sventon_agent import sventon_agent, sventon_agent_ppo_trainer, sventon_agent_dqn_trainer
from agents.agent_utils.reward_shapers import *
from aux.parameter import *

settings = {
            #Project
            "run-id" : "DBG09",
            "presets" : ["default", "sventon", "sventon_ppo", "resblock"],
            "n_step_value_estimates"    : 1,

            #RL-algo-settings
            "ppo_parameters" : {
                    'clipping_parameter' : 0.15,
                    'value_loss' : 0.2,
                    'policy_loss' : 1.0,
                    'entropy_loss' : 0.0002,
                    'negative_dampener' : 1.0,
                    'entropy_floor_loss' : 2.0,
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
            "gae_lambda"                : 0.97, #0.95 default
            "n_samples_each_update"     : 4096,
            # "n_samples_each_update"     : 128,
            # "time_to_training" : 0,
            # "n_samples_each_update"     : 256,
            "minibatch_size"            : 64, #256, #128
            "n_train_epochs_per_update" : 3,
            "dynamic_n_epochs"          : True,
            "time_to_reference_update"  : 1, #How after how many do_training calls do we update the reference-model?
            "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
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
# "run-id" : "DBG05",
# "ppo_parameters" : {
#         'clipping_parameter' : 0.1,
#         'value_loss' : 0.3,
#         'policy_loss' : 1.0,
#         'entropy_loss' : 0.0005,
#         'negative_dampener' : 1.0,
#         'entropy_floor_loss' : 0.0,
#         },
# "minibatch_size"            : 256, #256, #128
# "n_train_epochs_per_update" : 3,
# "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
# },
# {
# "run-id" : "DBG06",
# "ppo_parameters" : {
#         'clipping_parameter' : 0.2,
#         'value_loss' : 0.3,
#         'policy_loss' : 1.0,
#         'entropy_loss' : 0.0003,
#         'negative_dampener' : 1.0,
#         'entropy_floor_loss' : 0.0,
#         },
# "minibatch_size"            : 64, #256, #128
# "n_train_epochs_per_update" : 2,
# "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
# },
# {
# "run-id" : "DBG07",###KOTH
# "ppo_parameters" : {
#         'clipping_parameter' : 0.15,
#         'value_loss' : 0.2,
#         'policy_loss' : 1.0,
#         'entropy_loss' : 0.0001,
#         'negative_dampener' : 1.0,
#         'entropy_floor_loss' : 2.0,
#         },
# "minibatch_size"            : 64, #256, #128
# "n_train_epochs_per_update" : 3,
# "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
# },
# {
# "run-id" : "DBG08",
# "ppo_parameters" : {
#         'clipping_parameter' : 0.15,
#         'value_loss' : 0.2,
#         'policy_loss' : 1.0,
#         'entropy_loss' : 0.0004,
#         'negative_dampener' : 1.0,
#         'entropy_floor_loss' : 2.0,
#         },
# "minibatch_size"            : 64, #256, #128
# "n_train_epochs_per_update" : 3,
# "value_lr"                  : exp_parameter(3e-5, base=10.0, decay=1/total_steps),
# },
{
"run-id" : "DBG11",
"ppo_parameters" : {
'clipping_parameter' : 0.15,
'value_loss' : 1.0,
'policy_loss' : 1.0,
'entropy_loss' : 0.0002,
'negative_dampener' : 1.0,
'entropy_floor_loss' : 2.0,
},
"minibatch_size"            : 64, #256, #128
"n_train_epochs_per_update" : 3,
"value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
},
{
"run-id" : "DBG12",
"ppo_parameters" : {
        'clipping_parameter' : 0.15,
        'value_loss' : 0.4,
        'policy_loss' : 1.0,
        'entropy_loss' : 0.0002,
        'negative_dampener' : 1.0,
        'entropy_floor_loss' : 2.0,
        },
"minibatch_size"            : 64, #256, #128
"n_train_epochs_per_update" : 1,
"value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
},
{
"run-id" : "DBG13",
"ppo_parameters" : {
        'clipping_parameter' : 0.15,
        'value_loss' : 0.2,
        'policy_loss' : 1.0,
        'entropy_loss' : 0.002,
        'negative_dampener' : 1.0,
        'entropy_floor_loss' : 2.0,
        },
"minibatch_size"            : 256, #256, #128
"n_train_epochs_per_update" : 2,
"value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),
},
]
