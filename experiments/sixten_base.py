from tools.parameter import *
import tensorflow.compat.v1 as tf
settings = {
            #Project
            "run-id" : "SIXten-v1.0",
            "description" : "SIXten is a V-fcn learning agent, operating on a prioritized distributed experience replay, doing k-step value estimates, utilizing the world model provided by the tetris_env!",
            "presets": ["default", "keyboardconv", "sixten"],
            "architecture" : "vanilla",

            #Train parameters
            "action_temperature"    : linear_parameter(1, final_val=4.0, time_horizon=total_steps),
            "n_step_value_estimates"    : 5,
            "n_samples_each_update"     : 16384,
            "minibatch_size"            : 128,
            "n_train_epochs_per_update" : 1,  #5
            "time_to_reference_update"  : 20, #How after how many do_training calls do we update the reference-model?
            "value_lr"                  : exp_parameter(1e-3, base=10.0, decay=3/total_steps),

            #Exp-replay parameters
            "prioritized_replay_alpha"      : constant_parameter(0.7),
            "prioritized_replay_beta"       : linear_parameter(0.5, final_val=1.0, time_horizon=total_steps),

            #Game settings
            "pieces" : [0,6],
            "game_size" : [22,10],

            #Threading
            "n_workers"            : 3,
            "n_envs_per_thread"    : 16,
            "worker_steps"         : total_steps // 48,
            "worker_net_on_cpu"    : True,
            "trainer_net_on_cpu"   : False,
           }
