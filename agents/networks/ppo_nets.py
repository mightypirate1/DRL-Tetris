
import tensorflow as tf
import numpy as np

import aux.utils as utils
from agents.networks.builders import q_nets as Q
from agents.networks import network_utils as N

class ppo_nets:
    ###
    ### Contents:
    ###
    ###  (I)    init (internal variables and placeholders, call the nn-construction etc)
    ###  (II)   eval(...) function call (for use from the outside)
    ###  (III)  train(...) function call (for use from the outside)
    ###  (IV)   builder fcn: constructs a double duelling k-step Q-net, depending on input specs
    ###  (V)    training_ops builder: once the data in in the right places, we specify Q-updates etc
    ###  (VI)   utility fcns for ref-updates and weight loading etc
    ###

    # (I) init
    def __init__(self, agent_id, name, state_size, output_shape, sess, k_step=1, settings=None, worker_only=False):
        assert len(output_shape) == 3, "expected 3D-actions"
        assert k_step > 0, "k_step AKA n_step_value_estimates has to be greater than 0!"

        #Basics
        self.settings = utils.parse_settings(settings)
        self.scope_name = "agent{}_{}".format(agent_id,name)
        self.name = name
        self.session = sess
        self.worker_only = worker_only

        #Choose what type of architecture is used for Q-heads!
        if self.settings["q_net_type"] == "silver":
            self.network_type = Q.q_net_silver
        elif self.settings["q_net_type"] == "vanilla":
            self.network_type = Q.q_net_vanilla
        elif self.settings["q_net_type"] == "keyboard":
            self.network_type = Q.q_net_keyboard

        #Shapes and params
        self.output_shape = self.n_rotations, self.n_translations, self.n_pieces = output_shape
        self.state_size_vec, self.state_size_vis = state_size
        self.k_step = k_step

        #DBG
        self.stats_tensors, self.debug_tensors, self.visualization_tensors = [], [], []

        #Define tensors/placeholders
        with tf.variable_scope(self.scope_name):
            if self.worker_only:
                self.rewards_tf             =  None
                self.dones_tf               =  None
                self.vector_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
                self.visual_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
                self.vector_inputs_training =  None
                self.visual_inputs_training =  None
                self.actions_training_tf    =  None
                self.pieces_training_tf     =  None
                self.probabilities_old_tf   =  None
            else:
                self.vector_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
                self.visual_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
                self.vector_inputs_training = [tf.placeholder(tf.float32, (None, k_step+1)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
                self.visual_inputs_training = [tf.placeholder(tf.float32, (None, k_step+1)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
                self.rewards_tf             =  tf.placeholder(tf.float32, (None, k_step+1, 1), name='reward')
                self.dones_tf               =  tf.placeholder(tf.int32, (None, k_step+1, 1), name='done')
                self.actions_training_tf    =  tf.placeholder(tf.uint8, (None, k_step+1, 2), name='action')
                self.pieces_training_tf     =  tf.placeholder(tf.uint8, (None, k_step+1, 1), name='piece')
                self.probabilities_old_tf   =  tf.placeholder(tf.float32, (None, k_step+1, 1), name='probabilities')
                self.learning_rate_tf       =  tf.placeholder(tf.float32, shape=())
            self.training_tf = tf.placeholder(tf.bool, shape=())
            self.main_nets  = self.network_type("nets-main",      output_shape, state_size, self.settings, worker_only=worker_only, training=self.training_tf, kbd_activation=N.action_softmax, raw_outputs=True)
            self.ref_nets   = self.network_type("nets-reference", output_shape, state_size, self.settings, worker_only=worker_only, training=self.training_tf, kbd_activation=N.action_softmax, raw_outputs=True)
            #Create trainer and a network to train!
            self.pi_tf, self.v_tf, self.training_probs_tf, self.old_prob_tf, self.training_values_tf, self.target_values_tf, self.advantages_tf, self.action_entropy_tf, self.main_scope, self.ref_scope\
                                    = self.create_network_with_trainer(
                                                                        self.vector_inputs,
                                                                        self.visual_inputs,
                                                                        self.vector_inputs_training,
                                                                        self.visual_inputs_training,
                                                                        self.actions_training_tf,
                                                                        self.pieces_training_tf,
                                                                        self.rewards_tf,
                                                                        self.dones_tf,
                                                                       )
            self.main_net_vars        = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.main_scope.name)
            self.main_net_assign_list = self.create_weight_setting_ops(self.main_net_vars)
            if not self.worker_only:
                self.reference_net_vars        = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.ref_scope.name)
                self.reference_net_assign_list = self.create_weight_setting_ops(self.reference_net_vars)
                self.training_ops              = self.create_training_ops()
            self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
            self.init_ops = tf.variables_initializer(self.all_vars)
        #Run init-op
        self.session.run(self.init_ops)

    # (II) eval
    def evaluate(self,
                inputs,
                only_policy=False,
                ):
        vector, visual = inputs
        run_list = [self.pi_tf, self.v_tf] if not only_policy else [self.pi_tf]
        feed_dict = {self.training_tf : False}
        for idx, vec in enumerate(vector):
            feed_dict[self.vector_inputs[idx]] = vec
        for idx, vis in enumerate(visual):
            feed_dict[self.visual_inputs[idx]] = vis
        return_values = self.session.run(run_list, feed_dict=feed_dict)
        return return_values

    # (III) train
    def train(  self,
                vector_states,
                visual_states,
                actions,
                pieces,
                probabilities,
                rewards,
                dones,
                lr=None,
                fetch_visualizations=False,
              ):
        vis_tensors = [] #This feature is deactivated for now. Somehow it broke the nn; tf didnt think it had any inputs when I used them :(
        run_list = [
                    self.training_ops,
                    self.stats_tensors,
                    self.debug_tensors,
                    vis_tensors,
                    ]
        feed_dict = {
                        self.pieces_training_tf : pieces,
                        self.probabilities_old_tf : probabilities,
                        self.actions_training_tf : actions,
                        self.rewards_tf : rewards,
                        self.dones_tf : dones,
                        self.learning_rate_tf : lr,
                        self.training_tf : True,
                    }
        #Add also the state information to the appropriate placeholders..
        for idx, vec in enumerate(vector_states):
            feed_dict[self.vector_inputs_training[idx]] = vec
        for idx, vis in enumerate(visual_states):
            feed_dict[self.visual_inputs_training[idx]] = vis
        _, stats, dbg, vis = self.session.run(run_list, feed_dict=feed_dict)
        N.debug_prints(dbg,self.debug_tensors)
        return zip(self.stats_tensors, stats), zip(self.visualization_tensors, vis)

    # (IV)
    def create_network_with_trainer(self, vector_states, visual_states, vector_states_training, visual_states_training, actions_training, pieces_training, rewards, dones):
        with tf.variable_scope("ppo-nets") as vs:
            v_tf, pi_tf, main_scope = self.main_nets(vector_states, visual_states)

            #Workers are easy!
            if self.worker_only:
                return pi_tf, v_tf, None, None, None, None, None, None, main_scope, None

            # # #
            # Prepare values for ppo updates:
            # # #

            #0) Figure out what estimates we need:
            estimator_steps = [i for i in range(1,self.k_step+1)]
            if "sparse_value_estimate_filter" in self.settings: #filter out all k that are divisible by any of the numbers provided
                if len(self.settings["sparse_value_estimate_filter"]) > 0:
                    filter = np.array(self.settings["sparse_value_estimate_filter"]).reshape((1,-1))
                    steps = np.array(estimator_steps).reshape((-1,1))
                    filter = np.prod((steps % filter),axis=1)!=0
                    estimator_steps = steps[np.where(filter)].ravel().tolist()

            #1) Evaluate all the states, and create a bool to tell if the round is over
            dones_tf = tf.minimum(1, tf.cumsum(dones, axis=1)) #Minimum ensures there is no stray done from an adjacent trajectory influencing us...
            done_time_tf = tf.reduce_sum( 1-dones_tf, axis=1)
            pi_t_tf, v_t_tf = [None for _ in range(self.k_step+1)], [None for _ in range(self.k_step+1)]
            for t in [0] + estimator_steps:
                # Appy our Q-nets to each time step to yield all V- & Q-estimates!
                nets = self.main_nets if t==0 else self.ref_nets
                s_t_vec = [vec_state[:,t,:] for vec_state in vector_states_training]
                s_t_vis = [vis_state[:,t,:] for vis_state in visual_states_training]
                v_all,pi,ref_scope = nets(s_t_vec, s_t_vis)
                pi_t_tf[t] = pi
                # Figure out which V to assign to each state:
                if self.settings["separate_piece_values"]:
                    #If we do separate piece-values, we compute the average
                    mean_v = tf.reduce_sum(v_all * nets.used_pieces_mask_tf, axis=3, keepdims=True) / nets.n_used_pieces
                    mean_v = tf.reshape(mean_v, [-1, 1])
                    #Ugly to haveto use nets' mask etc. Think of a better solution!
                else:
                    #Else we just reduce dimentions
                    mean_v = tf.reshape(v_all, [-1, 1])
                    v_all = tf.reshape(v_all, [-1, 1])
                v_t_tf[t] = mean_v if t>0 else v_all #All future states are treated as if we do not know the piece (which we don't), and current state as if we do!

            #2) Do all the V-estimators, k-step style
            gamma = -self.settings["gamma"] if self.settings["single_policy"] else self.settings["gamma"]
            def k_step_estimate(k):
                k = int(k)
                e = 0
                for t in range(k):
                    e += rewards[:,t,:] * tf.cast((done_time_tf >= t),tf.float32) * (gamma**t)
                e += v_t_tf[k] * tf.cast((done_time_tf >= k),tf.float32) * (gamma**k)
                return e
            estimators = [(k,k_step_estimate(k)) for k in estimator_steps if k <= self.k_step]

            #3) GAE-style aggregation
            weight = 0
            estimator_sum_tf = 0
            gae_lambda = self.settings["gae_lambda"]
            for k,e in estimators:
                estimator_sum_tf += e * gae_lambda**k
                weight += gae_lambda**k
            value_estimator_tf = estimator_sum_tf / weight

            #4) Prepare training value, and training probability!
            r_mask = tf.reshape(tf.one_hot(actions_training[:,0,0], self.n_rotations),    (-1, self.n_rotations,    1,  1), name='r_mask')
            t_mask = tf.reshape(tf.one_hot(actions_training[:,0,1], self.n_translations), (-1,  1, self.n_translations, 1), name='t_mask')
            p_mask = tf.reshape(tf.one_hot(pieces_training[:,0,:],  self.n_pieces),       (-1,  1,  1, self.n_pieces     ), name='p_mask')
            rtp_mask = r_mask*t_mask*p_mask
            V_of_s_t = v_t_tf[0] #shape [?,1,1,n] where n is P if separate_piece_values else 1
            if self.settings["separate_piece_values"]:
                V_of_s_t = V_of_s_t * p_mask #Mask out the value we want
                V_of_s_t = tf.reduce_sum(V_of_s_t, axis=[2,3])
            PI_all = pi_t_tf[0]
            _PI_of_a_t = PI_all * rtp_mask
            PI_of_a_t = tf.reduce_sum(_PI_of_a_t, axis=[1,2,3])

            #5) Target and predicted values. Advantages, entropy and action-probability for policy update!
            target_values_tf = tf.stop_gradient(value_estimator_tf)
            training_values_tf = V_of_s_t
            advantages_tf = tf.stop_gradient(target_values_tf - training_values_tf)
            training_prob_tf = tf.expand_dims(PI_of_a_t,1)
            old_prob_tf = self.probabilities_old_tf[:,0,:]
            action_entropy_tf = tf.reduce_sum(N.action_entropy(PI_all) * p_mask, axis=3) #axis 1 is singleton anyway...

            #As always - we like some stats!
            self.output_as_stats(value_estimator_tf, name="target-val")
            self.output_as_stats(done_time_tf, name="done_time")
        return pi_tf, v_tf, training_prob_tf, old_prob_tf, training_values_tf, target_values_tf, advantages_tf, action_entropy_tf, main_scope, ref_scope

    # (V)
    def create_training_ops(self):
        if self.worker_only:
            return None
        params = self.settings["ppo_parameters"]
        clip_param, c1, c2, c3, e = params["clipping_parameter"], params["value_loss"], params["policy_loss"], params["entropy_loss"], 10**-6
        r = tf.maximum(self.training_probs_tf, e) / tf.maximum(self.old_prob_tf, e)
        clipped_r = tf.clip_by_value( r, 1-clip_param, 1+clip_param )
        policy_loss = tf.minimum( r * self.advantages_tf, clipped_r * self.advantages_tf )
        self.value_loss_tf   =  c1 * tf.losses.mean_squared_error(self.target_values_tf, self.training_values_tf)
        self.policy_loss_tf  = -c2 * tf.reduce_mean(policy_loss)
        self.entropy_loss_tf = -c3 * tf.reduce_mean(self.action_entropy_tf)
        self.regularizer_tf = self.settings["nn_regularizer"] * tf.add_n([tf.nn.l2_loss(v) for v in self.main_net_vars])
        self.loss_tf = self.value_loss_tf + self.policy_loss_tf + self.entropy_loss_tf + self.regularizer_tf
        training_ops = self.settings["optimizer"](learning_rate=self.learning_rate_tf).minimize(self.loss_tf)
        #Stats: we like stats.
        self.output_as_stats( self.loss_tf, name='tot_loss', only_mean=True)
        self.output_as_stats( self.value_loss_tf, name='value_loss', only_mean=True)
        self.output_as_stats(-self.policy_loss_tf, name='policy_loss', only_mean=True)
        self.output_as_stats(-self.entropy_loss_tf, name='entropy_loss', only_mean=True)
        self.output_as_stats( self.regularizer_tf, name='reg_loss', only_mean=True)
        return training_ops

    ###
    ###  (VI) Utilities etc...
    ###

    def create_weight_setting_ops(self, collection):
        assign_placeholder_list = []
        for var in collection:
            shape, dtype = var.shape, var.dtype
            assign_val_placeholder_tf = tf.placeholder(shape=shape, dtype=dtype)
            assign_op_tf = var.assign(assign_val_placeholder_tf)
            assign_placeholder_list.append(
                                            {
                                                "assign_op" : assign_op_tf,
                                                "assign_val_placeholder" : assign_val_placeholder_tf,
                                            }
                                          )
        return assign_placeholder_list

    def swap_networks(self):
        # Swaps reference- and main-weights
        main_weights = self.get_weights(self.main_net_vars)
        ref_weights  = self.get_weights(self.reference_net_vars)
        self.set_weights(self.reference_net_assign_list, main_weights)
        self.set_weights(self.main_net_assign_list, ref_weights )

    def reference_update(self):
        main_weights = self.get_weights(self.main_net_vars)
        self.set_weights(self.reference_net_assign_list,main_weights)

    def get_weights(self, collection):
        ret = self.session.run(collection)
        return ret

    def set_weights(self, assign_list, weights):
        run_list = []
        feed_dict = {}
        for w,assign in zip(weights,assign_list):
            run_list.append(assign['assign_op'])
            feed_dict[assign['assign_val_placeholder']] = w
        self.session.run(run_list, feed_dict=feed_dict)

    def output_as_stats(self, tensor, name=None, only_mean=False):
        if name is None:
            name = tensor.name
        #Corner case :)
        if len(tensor.shape) == 0:
            self.stats_tensors.append(tf.identity(tensor, name=name))
            return
        self.stats_tensors.append(tf.reduce_mean(tensor, axis=0, name=name+'_mean'))
        if only_mean:
            return
        self.stats_tensors.append(tf.reduce_max(tensor, axis=0, name=name+'_max'))
        self.stats_tensors.append(tf.reduce_min(tensor, axis=0, name=name+'_min'))

    def create_fixed_targets(self, Q, mask, target):
        _target = tf.reshape(target, [-1,1,1,1]) * mask
        return _target + (1-mask) * Q

    def get_random_conv_layers(self,tensor, n):
        _max = tensor.shape.as_list()[3]
        if _max - n > 3:
            start = np.random.choice(_max-n)
            idxs = slice(start, start+3)
        else:
            idxs = slice(0,n)
        return tensor[0,:,:,idxs]

    @property
    def output(self):
        return self.q_tf
