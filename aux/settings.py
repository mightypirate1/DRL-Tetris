import tensorflow as tf
from aux.parameter import *


#Aliasing for pieces, so they are easier to specify :)
(l,j,s,z,i,t,o) = (0,1,2,3,4,5,6)

default_settings = {
                ## AGENT:
                    "agent" : None,
                    #Training:
                    "minibatch_size" : 32,
                    "time_to_training" : 10**3,
                    "time_to_reference_update" : 5, #How after how many do_training calls do we update the reference-model?
                    "n_samples_each_update" : 2**7,
                    "n_train_epochs_per_update" : 5,
                    "value_lr" : linear_parameter(5*10**-6, final_val=5*10**-8, time_horizon=10**7),
                    "alternating_models" : True, #This means that instead of using a reference-model to produce target values for the model, we instead make the two models switch roles periodically. Empirically this seems to maybe have advantages.
                    #exploration
                    "dithering_scheme" : "adaptive_epsilon",
                    "epsilon" : constant_parameter(1.0),
                    #curiosity
                    "use_curiosity" : False,
                    "curiosity_during_testing" : False,
                    "curiosity_reward_multiplier" : 0.3,
                    "curiosity_amount" : linear_parameter(1, final_val=0, time_horizon=0.6*10**7),
                    #avg_trajectory_length_learning:
                    "tau_learning_rate" : 0.01,
                    #experience replay:
                    "experience_replay_size" : 10**5,
                    "prioritized_experience_replay" : True,
                    "prioritized_replay_alpha" : constant_parameter(0.7),
                    "prioritized_replay_beta" : linear_parameter(0.5,final_val=1,time_horizon=10**7), #0.5, used in paper, then linear increase to 1...
                    #discount factors
                    "gamma_extrinsic" : 0.998,
                    "gamma_intrinsic" : 0.90,

                ##VALUE NET:
                    "value_head_n_hidden" : 5,
                    "value_head_hidden_size" : 512,
                    "worker_net_on_cpu" : True,

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
                    "env" : None,
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
