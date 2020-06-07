
import tensorflow as tf
import numpy as np

import aux.utils as utils
from agents.networks import network_utils as N
from agents.networks.value_estimator import value_estimator
from agents.networks.network import network

class prio_qnet(network):
    def __init__(self, agent_id, name, state_size, output_shape, session, k_step=1, settings=None, worker_only=False):
        network.__init__(self, agent_id, name, state_size, output_shape, session, k_step=k_step, settings=settings, worker_only=worker_only)
        #Build network!
        with self.scope as scope:
            self.vector_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
            self.visual_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
            self.training_tf = tf.placeholder(tf.bool, shape=())
            self.main_net  = self.network_type(
                                               "main-net",
                                               self.output_shape,
                                               self.settings,
                                               full_network=False,
                                               training=self.training_tf,
                                              )
            self.Q_tf, self.V_tf, self.A_tf = self.main_net(self.vector_inputs, self.visual_inputs)
            if not self.worker_only: #For trainers
                self.rewards_tf             =  tf.placeholder(tf.float32, (None, k_step+1, 1), name='reward')
                self.dones_tf               =  tf.placeholder(tf.int32, (None, k_step+1, 1), name='done')
                self.actions_tf             =  tf.placeholder(tf.uint8, (None, 2), name='action')
                self.pieces_tf              =  tf.placeholder(tf.uint8, (None, 1), name='piece')
                self.learning_rate_tf       =  tf.placeholder(tf.float32, shape=(), name='lr')
                self.loss_weights_tf        =  tf.placeholder(tf.float32, (None,1), name='loss_weights')
                self.estimator_target_tf, self.target_q_value_tf = self.create_targets(self.V_tf)
                self.training_ops           = self.create_training_ops()

            self.main_net_assign_list = self.create_weight_setting_ops(self.main_net.variables)
            self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            self.init_ops = tf.variables_initializer(self.variables)
        #Run init-op
        self.session.run(self.init_ops)

    def evaluate(self, inputs):
        vector, visual = inputs
        run_list = [self.Q_tf, self.V_tf]
        feed_dict = {self.training_tf : False}
        for idx, vec in enumerate(vector):
            feed_dict[self.vector_inputs[idx]] = vec
        for idx, vis in enumerate(visual):
            feed_dict[self.visual_inputs[idx]] = vis
        return_values = self.session.run(run_list, feed_dict=feed_dict)
        return return_values

    def compute_targets( self,
                vector_states,
                visual_states,
                rewards,
                dones,
                time_stamps=None,
              ):
        run_list = self.estimator_target_tf
        feed_dict = {
                        self.rewards_tf : rewards,
                        self.dones_tf : dones,
                        self.training_tf : False,
                    }
        estimator_dict = self.estimator.feed_dict(vector_states, visual_states)
        feed_dict.update(estimator_dict)
        targets = self.session.run(run_list, feed_dict=feed_dict)
        return targets

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
                    self.stats_tensors,
                    ]
        feed_dict = {
                        self.pieces_tf : pieces,
                        self.actions_tf : actions,
                        self.target_q_value_tf : targets,
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
        return new_prios, zip(self.stats_tensors, stats)

    def create_training_ops(self):
        r_mask = tf.reshape(tf.one_hot(self.actions_tf[:,0], self.n_rotations),    (-1, self.n_rotations,    1,  1))
        t_mask = tf.reshape(tf.one_hot(self.actions_tf[:,1], self.n_translations), (-1,  1, self.n_translations, 1))
        p_mask = tf.reshape(tf.one_hot(self.pieces_tf[:,0],  self.n_pieces),       (-1,  1,  1, self.n_pieces     ))
        rtp_mask = r_mask * t_mask * p_mask
        q_rtp = tf.expand_dims( tf.reduce_sum(self.Q_tf * rtp_mask, axis=[1,2,3]), 1)

        #Prios is easy
        self.new_prios_tf = tf.abs(q_rtp - self.target_q_value_tf)
        if self.settings["optimistic_prios"] != 0.0:
            self.new_prios_tf += self.settings["optimistic_prios"] * tf.nn.relu(self.new_prios_tf)

        self.value_loss_tf = tf.losses.mean_squared_error(q_rtp, self.target_q_value_tf, weights=self.loss_weights_tf)
        self.regularizer_tf = self.settings["nn_regularizer"] * tf.add_n([tf.nn.l2_loss(v) for v in self.main_net.variables])
        self.loss_tf = self.value_loss_tf + self.regularizer_tf
        training_ops = self.settings["optimizer"](learning_rate=self.learning_rate_tf).minimize(self.loss_tf)
        #Stats: we like stats.
        self.output_as_stats(q_rtp, name='q_val', only_mean=False)
        self.output_as_stats(self.target_q_value_tf, name='q_target', only_mean=False)
        self.output_as_stats(self.loss_tf, name='tot_loss', only_mean=True)
        self.output_as_stats(self.value_loss_tf, name='value_loss', only_mean=True)
        self.output_as_stats(self.regularizer_tf, name='reg_loss', only_mean=True)
        return training_ops

    def create_targets(self, values):
        target_values_tf = tf.placeholder(tf.float32, (None,1), name='target_val_ph')
        self.ref_net   = self.network_type(
                                           "reference-net",
                                           self.output_shape,
                                           self.settings,
                                           training=False,
                                           kbd_activation=N.action_softmax,
                                           raw_outputs=True
                                          )
        self.estimator = value_estimator(
                                           self.state_size_vec,
                                           self.state_size_vis,
                                           self.ref_net,
                                           self.rewards_tf,
                                           self.dones_tf,
                                           self.k_step,
                                           self.gamma,
                                           self._lambda, # lambda is used for general advantage estimation (see paper for details...)
                                           filter=self.estimator_filter, # ignore certain k-estimates for speed (optional)
                                           truncate_aggregation=self.settings["value_estimator_params"]["truncate_aggregation"], # only do gae until en of episode
                                          )
        self.reference_net_assign_list = self.create_weight_setting_ops(self.ref_net.variables)
        estimator_target_tf = self.estimator.output_tf
        # print(estimator_target_tf, target_values_tf);exit("is this broken?")
        return estimator_target_tf, target_values_tf

    @property
    def output(self):
        return self.Q_tf
