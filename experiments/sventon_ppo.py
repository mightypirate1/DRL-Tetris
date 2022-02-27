import tensorflow.compat.v1 as tf
from tools.parameter import *

settings = {
    "run-id" : "docker004",
    "presets" : ["default", "sventon", "sventon_ppo", "resblock"],

    #Project
    "augment_data" : False,
    "normalize_advantages" : True,
    "n_step_value_estimates"    : 1,

    #RL-algo-settings
    "ppo_parameters" : {
        'clipping_parameter' : 0.10,
        'value_loss' : 0.4,
        'policy_loss' : 1.4,
        'entropy_loss' : 0.02,
        'entropy_floor_loss' : 0.0,
        'rescaled_entropy' : 1.0,
        # 'clipping_parameter' : 0.10,
        # 'value_loss' : 0.4,
        # 'policy_loss' : 0.9,
        # 'entropy_loss' : 0.025,
        # 'entropy_floor_loss' : 0.0,
        # 'rescaled_entropy' : 0.0,
    },

    "nn_regularizer" : 1e-4,
    "resblock_dropout" : 0.15,
    #Train parameters
    "value_lr"                  : 1e-5, # exp_parameter(3e-5, base=10.0, decay=2/2e8),

    # "gae_lambda"                : 0.85,
    "gae_lambda"                : 0.98, #linear_parameter(2.0, final_val=0.0, time_horizon=2e8, max=0.85),
    "n_samples_each_update"     : 2048,
    "minibatch_size"            : 64,
    "n_train_epochs_per_update" : 4,

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
    "render_screen_dims" : (1920,1080),
}

patches = [

]
