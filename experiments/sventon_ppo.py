import tensorflow as tf

from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.sventon_agent import sventon_agent, sventon_agent_ppo_trainer, sventon_agent_dqn_trainer
from agents.agent_utils.reward_shapers import *
from tools.parameter import *

settings = {
            #Project
            "augment_data" : False,
            "run-id" : "SVENton-v1.0",
            "presets" : ["default", "sventon", "sventon_ppo", "resblock"],
            "n_step_value_estimates"    : 1,

            #RL-algo-settings
            "ppo_parameters" : {
                    'clipping_parameter' : 0.15,
                    'value_loss' : 0.33,
                    'policy_loss' : 1.0,

                    'entropy_loss' : 0.00007,
                    'entropy_floor_loss' : 10.0,
                    },
            "ppo_epsilon" : constant_parameter(0.1),
            #Train parameters
            "value_lr"                  : constant_parameter(5e-5),
            "gae_lambda"                : 0.93, #0.95 default
            "n_samples_each_update"     : 4096,
            "minibatch_size"            : 128, #256, #128
            "n_train_epochs_per_update" : 4,

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
# "run-id" : "FAB-X02-128",
# "augment_data" : False,
# "minibatch_size"            : 128,
# },
#
# {
# "run-id" : "FAB-X02-256",
# "minibatch_size"            : 256,
# },
#
# {
# "run-id" : "FAB-X02-64",
# "minibatch_size"            : 64,
# "augment_data" : False,
# },
# {
# "run-id" : "FAB-X02-128-aug",
# "minibatch_size"            : 128,
# "augment_data" : True,
# },
# {
# "run-id" : "FAB-X02-128-aug-lowerLr",
# "value_lr"                  : constant_parameter(3e-5),
# "minibatch_size"            : 128,
# "augment_data" : True,
# },

]
