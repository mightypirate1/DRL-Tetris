import tensorflow as tf
from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.sventon_agent import sventon_agent, sventon_agent_ppo_trainer, sventon_agent_dqn_trainer
from agents.sherlock_agent import sherlock_agent, sherlock_agent_ppo_trainer
from agents.vector_agent import vector_agent, vector_agent_trainer
from tools.parameter import *

presets = {
            "sherlock" : {
                            "agent_type" : sherlock_agent.sherlock_agent,
                            "trainer_type" : sherlock_agent_ppo_trainer.sherlock_agent_ppo_trainer,
                        },

            "sventon" : {
                        "augment_data" : False, #make this option good, or remove from release
                        "dynamic_n_epochs" : False,
                        "worker_net_on_cpu" : False,
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
                        "value_estimator_params" : {
                                                    "truncate_aggregation" : True,
                                                    },
                        #nn stuff
                        "value_lr" : constant_parameter(1e-4),
                        "nn_output_activation" : tf.nn.tanh,
                        #
                        "trainer_thread_save_freq"  : 100,
                        ###
                        ### Maybe remove from project...
                        ###
                        "advantage_range" : 1.0,
                        "piece_advantage_range" : 1.0,
                        "tau_learning_rate" : 0.01,
                        "winrate_learningrate" : 0.01,
                        },

            "sventon_ppo" : {
                                "sventon_flavour" : "ppo",
                                "trainer_type" : sventon_agent_ppo_trainer.sventon_agent_ppo_trainer,
                                "eval_distribution" : "pi",
                                "train_distribution" : "pi",
                                "workers_computes_advantages" : True,
                                "ppo_parameters" : {
                                                    'clipping_parameter' : 0.05,
                                                    'value_loss' : 1.0,
                                                    'policy_loss' : 1.0,
                                                    'entropy_loss' : 0.01,
                                                    'negative_dampener' : 1.0,
                                                    },
                                "dynamic_n_epochs" : True,
                                "experience_replay_size" : 5*10**4,
                            },

            "sventon_dqn" : {
                                "sventon_flavour" : "dqn",
                                "eval_distribution" : "argmax",
                                "train_distribution" : "epsilon",
                                "epsilon" : constant_parameter(0.05),
                                "trainer_type" : sventon_agent_dqn_trainer.sventon_agent_dqn_trainer,
                                "prioritized_replay_alpha"      : constant_parameter(0.7),
                                "prioritized_replay_beta"       : constant_parameter(0.7),
                                "optimistic_prios" : 0.0,
                                "workers_computes_advantages" : False,
                            },

            "sixten" : {
                        "agent_type"        : vector_agent.vector_agent,
                        "trainer_type"      : vector_agent_trainer.vector_agent_trainer,
                        "train_distribution"    : "distribution_pareto",
                        "action_temperature"    : constant_parameter(1),
                        "optimistic_prios"      : 0.0,
                        "trainer_thread_save_freq"  : 100,
                        "n_samples_each_update"     : 16384,
                        "minibatch_size"            : 128,
                        "n_train_epochs_per_update" : 1,  #5
                        "time_to_reference_update"  : 20, #How after how many do_training calls do we update the reference-model?
                        "value_lr"                  : constant_parameter(1e-4),
                        "n_samples_to_start_training" : 40000,
                        "old_state_dict" : True,
                        "state_processor_separate_piece" : False,
                        "state_processor_piece_in_statevec" : True,
                        "field_as_image" : True,
                        "players_separate" : True,
                        "relative_state" : True,
                        "n_value_functions" : 1,
                        "gae_lambda" : 0.97,
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

            "keyboardconv" : {
                                "architecture" : "keyboard",
                                "pad_visuals"      : True,
                                "vectorencoder_n_hidden" : 1,
                                "vectorencoder_hidden_size" : 256,
                                "vectorencoder_output_size" : 32,
                                "visualencoder_n_convs" : 4,
                                "visualencoder_n_filters" : (16,32,32,4),
                                "visualencoder_filter_sizes" : ((7,7),(3,3), (3,3), (3,3)),
                                "peephole_convs"   : True,
                                "visualencoder_poolings" : [2,], #Pooling after layer numbers in this list
                                "visualencoder_peepholes" : [0,1,2],
                                "valuenet_n_hidden" : 1,
                                "valuenet_hidden_size" : 256,
                                "nn_output_activation" : tf.nn.tanh,
                            },

            "default" : { #everyone does this
                         "env_vector_type"   : tetris_environment_vector,
                         "env_type"          : tetris_environment,

                         "eval_distribution" : "argmax",
                         "game_size" : [22, 10],
                         "pieces" : [0,1,2,3,4,5,6],
                         "n_players" : 2,
                         "bar_null_moves"    : True,
                         "time_elapsed_each_action" : 400,
                         "old_state_dict" : False,
                         "state_processor_separate_piece" : True,
                         "state_processor_piece_in_statevec" : False,

                         "gamma"             :  0.98,
                         "n_step_value_estimates"    : 5,
                         "extra_rewards" : False,
                         "state_processor" : "state_dict",
                         "action_type" : "place_block",
                         "experience_replay_size" : 2*10**6,
                         "experience_replay_sample_mode" : "rank",
                         "alternating_models"        : False,
                         "time_to_training"          : 1024,

                         "workers_do_processing"     : True,
                         "trainer_thread_backup_freq"  : 1,
                         "worker_data_send_fequency" : 1,
                         "weight_transfer_frequency" : 1,
                         "worker_niceness"   : 5,
                         "worker_summaries" : False,
                         "run_standalone"       : False,
                         "single_policy"             : True,
                         "worker_net_on_cpu"    : True,
                         "trainer_net_on_cpu"   : False,
                         #Misc.
                         "render"            : True,
                         "render_simulation" : False,
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
                         "nn_regularizer" : 0.0001,
                         # "optimizer" : tf.train.GradientDescentOptimizer,
                         "optimizer" : tf.train.AdamOptimizer,
                        }
}
