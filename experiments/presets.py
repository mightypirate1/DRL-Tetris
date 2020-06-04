import tensorflow as tf
from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.sventon_agent import sventon_agent, sventon_agent_ppo_trainer, sventon_agent_dqn_trainer
from aux.parameter import *

presets = {
            "sventon" : {
                        "worker_net_on_cpu" : False,
                        #forced
                        "old_state_dict" : False, #SVENton uses the new one
                        "state_processor_separate_piece" : True,
                        #general
                        "gamma"             :  0.98,
                        "state_processor_piece_in_statevec" : False,
                        "experience_replay_size" : 2*10**6,
                        #q/v calculations
                        "q_target_locked_for_other_actions" : False,
                        "advantage_type" : "mean",#,"none",#"mean", #"max",
                        "separate_piece_values" : True,
                        "agent_type" : sventon_agent.sventon_agent,
                        #training params
                        "n_samples_each_update" : 8192,
                        "minibatch_size"            : 32, #256, #128
                        "n_train_epochs_per_update" : 3,
                        "time_to_reference_update"  : 1,
                        "n_samples_to_start_training" : 0,
                        #value estimator
                        "workers_do_processing" : True,
                        "value_estimator_params" : {
                                                    "truncate_aggregation" : True,
                                                    },
                        #nn stuff
                        "value_lr" : constant_parameter(1e-4),
                        "nn_regularizer" : 0.0001,
                        "nn_output_activation" : tf.nn.tanh,
                        # "optimizer" : tf.train.GradientDescentOptimizer,
                        "optimizer" : tf.train.AdamOptimizer,
                        #
                        "trainer_thread_save_freq"  : 100,
                        "workers_do_processing"     : False,
                        ###
                        ### Maybe remove from project...
                        ###
                        "time_to_training" : 1024,
                        "advantage_range" : 1.0,
                        "piece_advantage_range" : 1.0,
                        "tau_learning_rate" : 0.01,
                        "winrate_learningrate" : 0.01,
                        },
            "sventon_ppo" : {
                                "sventon_flavour" : "ppo",
                                "trainer_type" : sventon_agent_ppo_trainer.sventon_agent_ppo_trainer,
                                "eval_distribution" : "pi",
                                "train_distriburion" : "pi",
                                "workers_computes_advantages" : True,
                                "ppo_parameters" : {
                                                    'clipping_parameter' : 0.05,
                                                    'value_loss' : 1.0,
                                                    'policy_loss' : 1.0,
                                                    'entropy_loss' : 0.01,
                                                    'negative_dampener' : 1.0,
                                                    },
                            },
            "sventon_dqn" : {
                                "sventon_flavour" : "dqn",
                                "eval_distribution" : "argmax",
                                "train_distriburion" : "epsilon",
                                "epsilon" : constant_parameter(0.05),
                                "trainer_type" : sventon_agent_dqn_trainer.sventon_agent_dqn_trainer,
                                "experience_replay_sample_mode" : "rank",
                                "prioritized_replay_alpha"      : constant_parameter(0.7),
                                "prioritized_replay_beta"       : constant_parameter(0.7),
                                "optimistic_prios" : 0.0,
                                "workers_computes_advantages" : False,
                            },
            "resblock" : {
                            "resblock_dropout" : 0.0,#0.15,
                            "architecture" : "silver", #it says q, but it's usable for ppo too
                            "residual_block_settings" : {
                                                            "default" : {
                                                                            "n_layers" : 3,
                                                                            "n_filters" : 64,
                                                                            "normalization" : "layer",
                                                                            "dropout" : 0.0,
                                                                        },
                                                            "val_stream" : {
                                                                            "n_layers" : 4,
                                                                            "n_filters" : 64,
                                                                            }
                                                        },
                         },
            "default" : { #everyone does this
                         "env_vector_type"   : tetris_environment_vector,
                         "env_type"          : tetris_environment,
                         "run_standalone"       : False,
                         "single_policy"             : True,
                         "extra_rewards" : False,
                         "worker_net_on_cpu"    : True,
                         "trainer_net_on_cpu"   : False,
                         "bar_null_moves"    : True,
                         "time_elapsed_each_action" : 400,
                         "state_processor" : "state_dict",
                         "action_type" : "place_block",
                         "game_size" : [22, 10],
                         "pieces" : [0,1,2,3,4,5,6],

                         "trainer_thread_backup_freq"  : 1,
                         "worker_data_send_fequency" : 1,
                         "weight_transfer_frequency" : 1,
                         "worker_niceness"   : 5,
                         "n_players" : 2,
                         "worker_summaries" : False,
                         #Misc.
                         "render"            : True,
                         "bar_null_moves"    : True,
                         "render_color_theme" : [
                                                "171717",
                                                "d900ff",
                                                "ff9400",
                                                "9b00ff",
                                                "ff00a4",
                                                "ff00ed",
                                                "ff5c00",
                                                "7900ff",
                                                # "ffaf00",
                                                "400080"
                                                ],
                         "render_screen_dims" : (1970,1080),
                        }
}
