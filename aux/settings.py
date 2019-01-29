import tensorflow as tf
from aux.parameter import *


#Aliasing for pieces, so they are easier to specify :)
(l,j,s,z,i,t,o) = (0,1,2,3,4,5,6)

default_settings = {
                ## AGENT:
                    "agent_type"   : None,
                    "trainer_type" : None,
                    #Training:
                    "minibatch_size" : 32,
                    "time_to_training" : 10**3,
                    "time_to_reference_update" : 5, #How after how many do_training calls do we update the reference-model?
                    "n_samples_each_update" : 2048,
                    "n_train_epochs_per_update" : 5,
                    "value_lr" : linear_parameter(5*10**-6, final_val=5*10**-8, time_horizon=10**7),
                    "alternating_models" : True, #This means that instead of using a reference-model to produce target values for the model, we instead make the two models switch roles periodically. Empirically this seems to maybe have advantages.
                    #exploration
                    "dithering_scheme" : "adaptive_epsilon",
                    "epsilon" : constant_parameter(1.0),
                    #avg_trajectory_length_learning:
                    "tau_learning_rate" : 0.001,
                    #experience replay:
                    "experience_replay_size" : 10**5,
                    "prioritized_experience_replay" : True,
                    "prioritized_replay_alpha" : constant_parameter(0.7),
                    "prioritized_replay_beta" : linear_parameter(0.5,final_val=1,time_horizon=10**7), #0.5, used in paper, then linear increase to 1...
                    #discount factors
                    "gamma_extrinsic" : 0.998,   #Not in use...
                    "gamma_intrinsic" : 0.90,  #Not in use...

                ##NEURAL NET:
                    #Preprocessing
                    "relative_state"   : True,  #This means that both players sees themselves as the player to the left, and the other on the right
                    "field_as_image"   : True, #This preserves the 2D structure of the playing field, and keeps them separate from the vector part of the state
                    "players_separate" : True, #This keeps each players part of the state separate when passed to the neural net
                    #Value net:
                    "vectorencoder_n_hidden" : 2,
                    "vectorencoder_hidden_size" : 256,
                    "vectorencoder_output_size" : 256,
                    "visualencoder_n_convs" : 3,
                    "visualencoder_n_filters" : (32,32,16),
                    "visualencoder_filter_sizes" : ((3,3), (3,3), (5,5)),
                    "visualencoder_poolings" : [2,], #Pooling after layer numbers in this list
                    "valuenet_n_hidden" : 1,
                    "valuenet_hidden_size" : 1024,
                    "nn_regularizer" : 0.001,

                ##MULTIPROCESSING:
                    "worker_net_on_cpu" : True,
                    "trainer_net_on_cpu" : False,
                    "worker_data_send_fequency" : 100,
                    "weight_transfer_frequency" : 100,
                    #Threading
                    "n_workers"         : 4,
                    "n_envs_per_thread" : 16,
                    "worker_steps"      : 1000,
                    "process_patience"  : [0.1,0.1, 10.0], #runner/trainer/process_manager
                    #Division of labour
                    "wrangler_unpacks" : False,
                    "wrangler_update_mode" : "all", #"none", "budget"
                    "wrangler_trainerfeed_target_length" : 20, #How many samples do we try to get into the que?

                ##CURIOSITY NET:
                    "n_hidden_layers" : 3,
                    "layer_size" : 2500,
                    "output_size" : 10,
                    "RDN_activation" : tf.nn.tanh,
                    "curiosity_norm" : 2,
                    "input_normalization_lr" : 0.05,
                    "output_mu_lr" : 0.05,
                    "curiosity_lr" : constant_parameter(0.000001),

                ##ENV:
                    "environment_logging" : True,
                    "env_type" : None,
                    "env_vector_type" : None,
                    "n_players" : 2,
                    "pieces" : [l,j,s,z,i,t,o],
                    "game_size" : [22,10],
                    "time_elapsed_each_action" : 100,
                    "state_processor" : "state_dict",        #This is a function that is applied to the state before returning it.
                    "action_type" : "place_block",  #This determines if an action represents pressing a key or placing a block. Allowd values is "press_key" and "place_block"
                    "render" : True,                 #Gfx on?,
                    "pause_on_keypress" : True,
                    "render_screen_dims" : (860,400),
                    "render_simulation" : False,    #This renders the outcomes of the first 4 non-empty action sequences when simulating.
                    "bar_null_moves" : False,

                ##GAME CONTROLLER:
                    "max_round_time" : None,
                    "balance_winrate" : True,
                    "winrate_tolerance" : 1.2,
                    "winrate_learningrate" : 0.04,

                    }
