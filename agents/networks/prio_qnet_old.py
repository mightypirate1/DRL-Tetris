
import tensorflow.compat.v1 as tf
import numpy as np

import tools.utils as utils
from agents.networks.builders import sventon_architectures as arch
from agents.networks import network_utils as N

class prio_qnet:
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
        if self.settings["architecture"] == "silver":
            self.network_type = arch.resblock_net
        elif self.settings["architecture"] == "vanilla":
            self.network_type = arch.convthendense
        elif self.settings["architecture"] == "keyboard":
            self.network_type = arch.convkeyboard

        #Shapes and params
        self.output_shape = self.n_rotations, self.n_translations, self.n_pieces = output_shape
        self.state_size_vec, self.state_size_vis = state_size
        self.k_step = k_step

        #DBG
        self.stats_tensors_targets, self.stats_tensors_training, self.debug_tensors, self.visualization_tensors = [], [], [], []

        #Define tensors/placeholders
        with tf.variable_scope(self.scope_name):
            self.actions_tf             =  tf.placeholder(tf.uint8, (None, 2))
            self.pieces_tf              =  tf.placeholder(tf.uint8, (None, 1))
            self.vector_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
            self.visual_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
            if self.worker_only:
                self.rewards_tf             =  None
                self.dones_tf               =  None
                self.vector_inputs_training =  None
                self.visual_inputs_training =  None
                self.actions_training_tf    =  None
                self.pieces_training_tf     =  None
            else:
                self.vector_inputs_training = [tf.placeholder(tf.float32, (None, k_step+1)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
                self.visual_inputs_training = [tf.placeholder(tf.float32, (None, k_step+1)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
                self.rewards_tf             =  tf.placeholder(tf.float32, (None, k_step+1, 1), name='reward')
                self.dones_tf               =  tf.placeholder(tf.int32, (None, k_step+1, 1), name='done')
                self.q_training_targets_tf  =  tf.placeholder(tf.float32, (None, 1))
                self.actions_training_tf    =  tf.placeholder(tf.uint8, (None, k_step+1, 2), name='action')
                self.pieces_training_tf     =  tf.placeholder(tf.uint8, (None, k_step+1, 1), name='piece')
                self.time_stamps_tf         =  tf.placeholder(tf.float32, shape=(None,1))
                self.learning_rate_tf       =  tf.placeholder(tf.float32, shape=())
                self.loss_weights_tf        =  tf.placeholder(tf.float32, (None,1), name='loss_weights')
            self.training_tf = tf.placeholder(tf.bool, shape=())
            self.main_q_net  = self.network_type("q-net-main",      output_shape, state_size, self.settings, worker_only=worker_only, training=self.training_tf)
            self.ref_q_net   = self.network_type("q-net-reference", output_shape, state_size, self.settings, worker_only=worker_only, training=self.training_tf)
            #Create trainer and a network to train!
            self.q_tf, self.v_tf, self.a_tf, self.gae_q_target_tf, self.main_scope, self.ref_scope\
                                    = self.create_network(
                                                                        self.vector_inputs,
                                                                        self.visual_inputs,
                                                                        self.vector_inputs_training,
                                                                        self.visual_inputs_training,
                                                                        self.actions_tf,
                                                                        self.pieces_tf,
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
                compute_value=True,
                ):
        vector, visual = inputs
        run_list = [self.q_tf, self.v_tf] if compute_value else [self.a_tf]
        feed_dict = {self.training_tf : False}
        for idx, vec in enumerate(vector):
            feed_dict[self.vector_inputs[idx]] = vec
        for idx, vis in enumerate(visual):
            feed_dict[self.visual_inputs[idx]] = vis
        return_values = self.session.run(run_list, feed_dict=feed_dict)
        if not compute_value:
            return_values.append(np.zeros((len(vec),1)))
        return return_values

    def compute_targets( self,
                vector_states,
                visual_states,
                rewards,
                dones,
                time_stamps=None,
              ):
        run_list = [
                        self.stats_tensors_targets,
                        self.gae_q_target_tf,
                    ]
        feed_dict = {
                        self.time_stamps_tf : time_stamps if time_stamps is not None else [[0.0]],
                        self.rewards_tf : rewards,
                        self.dones_tf : dones,
                        self.training_tf : False,
                    }
        #Add also the state information to the appropriate placeholders..
        for idx, vec in enumerate(vector_states):
            feed_dict[self.vector_inputs_training[idx]] = vec
        for idx, vis in enumerate(visual_states):
            feed_dict[self.visual_inputs_training[idx]] = vis
        stats, targets = self.session.run(run_list, feed_dict=feed_dict)
        return targets, zip(self.stats_tensors_targets, stats)

    # (III) train
    def train(  self,
                vector_states,
                visual_states,
                actions,
                pieces,
                targets,
                weights=None,
                lr=None,
                time_stamps=None,
              ):
        run_list = [
                    self.training_ops,
                    self.new_prios_tf,
                    self.stats_tensors_training,
                    ]
        feed_dict = {
                        self.pieces_tf : pieces,
                        self.actions_tf : actions,
                        self.q_training_targets_tf : targets,
                        self.learning_rate_tf : lr,
                        self.loss_weights_tf : weights,
                        self.training_tf : True,
                    }
        #Add also the state information to the appropriate placeholders..
        for idx, vec in enumerate(vector_states):
            feed_dict[self.vector_inputs[idx]] = vec
        for idx, vis in enumerate(visual_states):
            feed_dict[self.visual_inputs[idx]] = vis
        _, new_prios, stats = self.session.run(run_list, feed_dict=feed_dict)
        return new_prios, zip(self.stats_tensors_training, stats)

    # (IV)
    def create_network(self, vector_states, visual_states, vector_states_training, visual_states_training, actions, pieces, actions_training, pieces_training, rewards, dones):
        with tf.variable_scope("prio_q-net") as vs:
            q_tf, v_tf, a_tf, main_scope = self.main_q_net(vector_states, visual_states)

            #Workers are easy!
            if self.worker_only:
                return q_tf, v_tf, a_tf, None, main_scope, None

            # # #
            # Trainers do Q-updates:
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
            q_t_tf, v_t_tf = [None for _ in range(self.k_step+1)], [None for _ in range(self.k_step+1)]
            for t in estimator_steps:
                # Appy our Q-nets to each time step to yield all V- & Q-estimates!
                s_t_vec = [vec_state[:,t,:] for vec_state in vector_states_training]
                s_t_vis = [vis_state[:,t,:] for vis_state in visual_states_training]
                q,v,_,ref_scope = self.ref_q_net(s_t_vec, s_t_vis)
                q_t_tf[t] = q
                v_t_tf[t] = v

            #2) Do all the V-estimators, k-step style
            extra_decay = 1.0 ** self.time_stamps_tf
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

            #5) Target and predicted values. Also prios for the exp-rep.
            gae_q_target_tf = value_estimator_tf

            #If enabled, lock the Q-value for the actions that were not played
            if self.settings["q_target_locked_for_other_actions"]:
                assert False, "this code is not updated for unified sventon"
                target_values_tf = self.create_fixed_targets(Q_s_all, rtp_mask, target_values_tf)

            #As always - we like some stats!
            self.output_as_stats(gae_q_target_tf, name="target-q")
            self.output_as_stats(done_time_tf, name="done_time")
        return q_tf, v_tf, a_tf, gae_q_target_tf, main_scope, ref_scope

    # (V)
    def create_training_ops(self):
        if self.worker_only:
            return None
        if self.settings["q_target_locked_for_other_actions"]:
            assert False

        #Extract single Q-value
        _q = self.q_tf
        r_mask = tf.reshape(tf.one_hot(self.actions_tf[:,0], self.n_rotations),    (-1, self.n_rotations,    1,  1))
        t_mask = tf.reshape(tf.one_hot(self.actions_tf[:,1], self.n_translations), (-1,  1, self.n_translations, 1))
        p_mask = tf.reshape(tf.one_hot(self.pieces_tf[:,0],  self.n_pieces),       (-1,  1,  1, self.n_pieces     ))
        rtp_mask = r_mask * t_mask * p_mask
        q = tf.expand_dims( tf.reduce_sum(_q * rtp_mask, axis=[1,2,3]), 1)

        #Prios is easy
        self.new_prios_tf = tf.abs(q - self.q_training_targets_tf)
        if self.settings["optimistic_prios"] != 0.0:
            self.new_prios_tf += self.settings["optimistic_prios"] * tf.nn.relu(self.new_prios_tf)

        self.value_loss_tf = tf.losses.mean_squared_error(q, self.q_training_targets_tf, weights=self.loss_weights_tf)
        self.regularizer_tf = self.settings["nn_regularizer"] * tf.add_n([tf.nn.l2_loss(v) for v in self.main_net_vars])
        self.loss_tf = self.value_loss_tf + self.regularizer_tf
        training_ops = self.settings["optimizer"](learning_rate=self.learning_rate_tf).minimize(self.loss_tf)
        #Stats: we like stats.
        self.output_as_stats(self.loss_tf, name='tot_loss', only_mean=True, train_stats=True)
        self.output_as_stats(self.value_loss_tf, name='value_loss', only_mean=True, train_stats=True)
        self.output_as_stats(self.regularizer_tf, name='reg_loss', only_mean=True, train_stats=True)
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

    def output_as_stats(self, tensor, name=None, only_mean=False, train_stats=False):
        stats_tensors = self.stats_tensors_training if train_stats else self.stats_tensors_targets
        if name is None:
            name = tensor.name
        #Corner case :)
        if len(tensor.shape) == 0:
            stats_tensors.append(tf.identity(tensor, name=name))
            return
        stats_tensors.append(tf.reduce_mean(tensor, axis=0, name=name+'_mean'))
        if only_mean:
            return
        stats_tensors.append(tf.reduce_max(tensor, axis=0, name=name+'_max'))
        stats_tensors.append(tf.reduce_min(tensor, axis=0, name=name+'_min'))

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
