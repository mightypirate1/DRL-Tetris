import tensorflow as tf

from environment.tetris_environment_vector import tetris_environment_vector
from environment.tetris_environment import tetris_environment
from agents.vector_q_agent import vector_q_agent, vector_q_agent_trainer
from agents.agent_utils.reward_shapers import *
from aux.parameter import *

settings = {
            #Project
            "run-id" : "bogus0",
            # "run-id" : "SVENton-alpha0",
            "state_processor_separate_piece" : True,
            "q_target_locked_for_other_actions" : False,
            "advantage_type" : "max",#,"none",#"mean", #"max",
            "old_state_dict" : False,
            "keyboard_conv" : True, #True,

            "separate_piece_values" : False,
            "keyboard_range" : 0.7,
            "piece_advantage_range" : 0.5,

            "EXPERIMENTAL_estimate" : True,
            "render_screen_dims" : (3840,2160), #My screen is huge
            # "render_simulation" : True

            #Train parameters
            "gae_lambda"                : 0.92, #0.95 default
            "n_step_value_estimates"    : 17,
            # "n_samples_each_update"     : 16384,
            "n_samples_each_update"     : 8192,
            "minibatch_size"            : 256,#256, #128
            "n_train_epochs_per_update" : 1,
            "time_to_reference_update"  : 3, #How after how many do_training calls do we update the reference-model?
            "value_lr"                  : exp_parameter(1e-4, base=10.0, decay=2/total_steps),
            # "n_samples_to_start_training" : 40000, #0

            #Exp-replay parameters
            "prioritized_replay_alpha"      : constant_parameter(0.7),
            "prioritized_replay_beta"       : linear_parameter(0.5, final_val=1.0, time_horizon=total_steps),
            "experience_replay_size"        : 2*10**6,
            "experience_replay_sample_mode" : 'rank',
            # "experience_replay_sample_mode" : 'proportional',

            "alternating_models"        : False,
            "time_to_training"          : 0,#10**3,
            "single_policy"             : True,

            #Dithering
            # "dithering_scheme"    : "pareto_distribution",
            # "action_temperature"  : linear_parameter(1, final_val=3.0, time_horizon=total_steps),
            # "dithering_scheme"    : "adaptive_epsilon",
            # "epsilon"  : linear_parameter(10, final_val=0.0, time_horizon=total_steps),
            "dithering_scheme"    : "epsilon",
            "epsilon"  : exp_parameter(1.0, base=10.0, decay=2/total_steps),
            "optimistic_prios" : 0.0,

            #Rewards
            "gamma"             :  0.98,
            "extra_rewards" : False,
            # "reward_ammount" : (1.0, 0.5,),
            # "reward_shaper" :  linear_reshaping,
            # "reward_shaper_param" : linear_parameter(0.0, final_val=0.0, time_horizon=0.3*total_steps),

            #Game settings
            "pieces" : [0,],
            "game_size" : [10,6],
            # "game_size" : [10,10],
            # "pieces" : [0,6],
            # "game_size" : [22,10],
            "time_elapsed_each_action" : 400,
            #Types
            "env_vector_type"   : tetris_environment_vector,
            "env_type"          : tetris_environment,
            "agent_type"        : vector_q_agent.vector_q_agent,
            "trainer_type"      : vector_q_agent_trainer.vector_q_agent_trainer,

            #Threading
            "run_standalone"       : False,
            "n_workers"            : 3,
            "n_envs_per_thread"    : 80,
            "worker_steps"         : 125000,
            "worker_net_on_cpu"    : True,
            "trainer_net_on_cpu"   : False,

            #Communication
            "trainer_thread_save_freq"  : 100,
            "trainer_thread_backup_freq"  : 1,
            "worker_data_send_fequency" : 1,
            "weight_transfer_frequency" : 1,
            "workers_do_processing"     : True,

            #Value net:
            "vectorencoder_n_hidden" : 1,
            "vectorencoder_hidden_size" : 256,
            "vectorencoder_output_size" : 32,
            ###
            "pad_visuals"      : True,
            "visualencoder_n_convs" : 3,
            "visualencoder_n_filters" : (64,64,128,),
            "visualencoder_filter_sizes" : ((5,5),(5,3),(5,3),),
            "visualencoder_poolings" : [3], #Pooling after layer numbers in this list
            "visualencoder_dropout" : 0.15, #Is this keep-rate or drop-rate...?
            "peephole_convs"   : True,
            "peephole_join_style"   : "add", #"concat"
            "visualencoder_peepholes" : [0,1],
            ##Kbd-vis
            "kbd_vis_n_convs" : 3,
            "kbd_vis_n_filters" : [128,128,128],
            ##Kbd
            "keyboard_n_convs" : 3,
            "keyboard_n_filters" : (64,64,),
            ###
            "valuenet_n_hidden" : 2,
            "valuenet_hidden_size" : 512,
            "nn_regularizer" : 0.0001,
            "nn_output_activation" : tf.nn.tanh,
            # "optimizer" : tf.train.GradientDescentOptimizer,
            "optimizer" : tf.train.AdamOptimizer,

            #Misc.
            "render"            : True,
            "bar_null_moves"    : True,
           }

patches = \
[
    {
        'run-id' : "bogus1"
    },
]
