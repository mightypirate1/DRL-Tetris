import tensorflow as tf

from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.sventon_agent import sventon_agent, sventon_agent_ppo_trainer, sventon_agent_dqn_trainer
from agents.agent_utils.reward_shapers import *
from aux.parameter import *

settings = {
            #Project
            "run-id" : "MOOSE-1",
            "presets" : ["default", "sventon", "sventon_dqn", "resblock"],

            #RL-algo-settings
            "train_distriburion" : "pareto_distribution",
            "action_temperature" : linear_parameter(1, final_val=3.0, time_horizon=total_steps),
            "prioritized_replay_alpha"      : constant_parameter(0.7),
            "prioritized_replay_beta"       : linear_parameter(0.5, final_val=1.0, time_horizon=total_steps),

            #Train parameters
            "gae_lambda"                : 0.96, #0.95 default
            "n_step_value_estimates"    : 37,
            "sparse_value_estimate_filter" : [3,4,5,13], #Empty list is no filter
            "n_samples_each_update"     : 256,
            # "n_samples_each_update"     : 4096,
            "minibatch_size"            : 256, #256, #128
            "n_train_epochs_per_update" : 10,
            "time_to_reference_update"  : 1, #How after how many do_training calls do we update the reference-model?
            "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=1/total_steps),

            #Game settings
            # "game_size" : [10,6],
            "pieces" : [2,3,4,5],
            "game_size" : [22,10],

            #Threading
            "n_workers"            : 3,
            "n_envs_per_thread"    : 80,
            "worker_steps"         : total_steps // 240,
           }
###
###
###

patches = \
[
    # {
    # "run-id" : "Fast-1",
    # "value_lr"                  : exp_parameter(1e-3, base=10.0, decay=1/total_steps),
    # },
    # {
    # "run-id" : "Fast-1",
    # "value_lr"                  : exp_parameter(1e-5, base=10.0, decay=1/total_steps),
    # },
]
