import tensorflow.compat.v1 as tf
from tools.parameter import *

settings = {
    "run-id" : "docker016",
    "presets" : ["default", "sventon", "sventon_ppo", "resblock"],

    #Project
    "augment_data" : False,
    # docker008 is just cautious -
    "compress_advantages" : {'lr':0.005, 'safety':3.0, 'clip_val':8.0, 'cautious':False,},
    "compress_value_loss" : {'lr':0.005, 'safety':3.0, 'clip_val':8.0, 'cautious':False,},
    "n_step_value_estimates"    : 1,

    #RL-algo-settings
    "ppo_parameters" : {
        'clipping_parameter' : 0.15,
        'value_loss' : 0.01, # 0.005,
        'policy_loss' : 0.9, # 1.7,
        'entropy_loss' : 0.0,         # 0.001,
        'entropy_floor_loss' : 0.000, # 0.013,
        'rescaled_entropy' : 0.0,
        'ppo_epsilon': 0.00,          # 0.05,

        # 'entropy_floor_loss' : 0.0,
        # 'rescaled_entropy' : 0.0,
    },

    #Train parameters
    "value_lr"                  : 1e-7, # 5e-6, # exp_parameter(3e-5, base=10.0, decay=2/2e8),
    "n_samples_each_update"     : 2048,
    "minibatch_size"            : 64,
    "n_train_epochs_per_update" : 4,

    "gae_lambda"                : 0.7,
    "gamma"                     : 0.98,

    "record-parameters-to-tb" : False,

    "nn_regularizer" : 1e-5,
    "resblock_dropout" : 0.25,

    "experience_replay_size"    : 2*10**4,

    #Architecture
    "residual_block_settings" : {
        "default" : {
            "n_layers" : 5,
            "n_filters" : 64,
            "normalization" : "layer",
            "filter_size" : (3,3),
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
    "n_envs_per_thread"    : 30,
    "worker_steps"         : total_steps // 3*40,
    #Misc
    "render_screen_dims" : (960,1080),
    # "render_screen_dims" : (1920,1080),
}

patches = [

]
