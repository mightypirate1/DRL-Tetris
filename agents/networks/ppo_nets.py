
import tensorflow as tf
import numpy as np

import aux.utils as utils
from agents.networks import network_utils as N
from agents.networks.value_estimator import value_estimator
from agents.networks.network import network

class ppo_nets(network):
    def __init__(self, agent_id, name, state_size, output_shape, session, k_step=1, settings=None, worker_only=False):
        network.__init__(self, agent_id, name, state_size, output_shape, session, k_step=k_step, settings=settings, worker_only=worker_only)
        #Build network!
        with self.scope as scope:
            self.vector_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
            self.visual_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
            self.training_tf = tf.placeholder(tf.bool, shape=())
            self.main_net  = self.network_type(
                                               "main-net",
                                               output_shape,
                                               self.settings,
                                               worker_only=self.worker_only,
                                               training=self.training_tf,
                                               kbd_activation=N.action_softmax,
                                               raw_outputs=True
                                              )
            self.v_tf, self.pi_tf = self.main_net(self.vector_inputs, self.visual_inputs)
            #
            if not self.worker_only: #For trainers
                self.rewards_tf             =  tf.placeholder(tf.float32, (None, k_step+1, 1), name='reward')
                self.dones_tf               =  tf.placeholder(tf.int32, (None, k_step+1, 1), name='done')
                self.actions_training_tf    =  tf.placeholder(tf.uint8, (None, k_step+1, 2), name='action')
                self.pieces_training_tf     =  tf.placeholder(tf.uint8, (None, k_step+1, 1), name='piece')
                self.probabilities_old_tf   =  tf.placeholder(tf.float32, (None, k_step+1, 1), name='probabilities')
                self.ppo_epsilon_tf         =  tf.placeholder(tf.float32, shape=(), name='ppo_epsilon')
                self.learning_rate_tf       =  tf.placeholder(tf.float32, shape=(), name='lr')
                self.ref_net   = self.network_type(
                                                   "reference-net",
                                                   output_shape,
                                                   self.settings,
                                                   worker_only=self.worker_only,
                                                   training=self.training_tf,
                                                   kbd_activation=N.action_softmax,
                                                   raw_outputs=True
                                                  )
                self.value_estimator = value_estimator(
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
                self.training_ops  = self.create_training_ops(
                                                              self.pi_tf,
                                                              self.v_tf,
                                                              self.value_estimator.output_tf,
                                                              self.actions_training_tf,
                                                              self.pieces_training_tf,
                                                              self.probabilities_old_tf,
                                                              self.learning_rate_tf,
                                                              ppo_epsilon=self.ppo_epsilon_tf,
                                                             )
                self.reference_net_assign_list = self.create_weight_setting_ops(self.ref_net.variables)
            self.main_net_assign_list = self.create_weight_setting_ops(self.main_net.variables)
            self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            self.init_ops = tf.variables_initializer(self.variables)
        #Run init-op
        self.session.run(self.init_ops)

    def evaluate(self,
                inputs,
                compute_value=True,
                ):
        vector, visual = inputs
        run_list = [self.pi_tf, self.v_tf] if compute_value else [self.pi_tf]
        feed_dict = {self.training_tf : False}
        for idx, vec in enumerate(vector):
            feed_dict[self.vector_inputs[idx]] = vec
        for idx, vis in enumerate(visual):
            feed_dict[self.visual_inputs[idx]] = vis
        return_values = self.session.run(run_list, feed_dict=feed_dict)
        if not compute_value:
            return_values.append(np.zeros((len(vec),1)))
        return return_values

    def train(  self,
                vector_states,
                visual_states,
                actions,
                pieces,
                probabilities,
                rewards,
                dones,
                lr=None,
                ppo_epsilon=0.0,
                unfiltered_inputs=True,
              ):
        run_list = [
                    self.training_ops,
                    self.stats_tensors,
                    self.debug_tensors,
                    self.visualization_tensors,
                    ]
        feed_dict = {
                        self.pieces_training_tf : pieces,
                        self.probabilities_old_tf : probabilities,
                        self.actions_training_tf : actions,
                        self.rewards_tf : rewards,
                        self.dones_tf : dones,
                        self.learning_rate_tf : lr,
                        self.ppo_epsilon_tf : ppo_epsilon,
                        self.training_tf : True,
                    }
        _filter = lambda x : x[:,0,:] if unfiltered_inputs else lambda x : x
        feed_dict.update(dict(zip(self.vector_inputs, map(_filter,vector_states))))
        feed_dict.update(dict(zip(self.visual_inputs, map(_filter,visual_states))))
        feed_dict.update(self.value_estimator.feed_dict(vector_states, visual_states))
        _, stats, dbg, vis = self.session.run(run_list, feed_dict=feed_dict)
        N.debug_prints(dbg,self.debug_tensors)
        return zip(self.stats_tensors, stats), zip(self.visualization_tensors, vis)

    def create_training_ops(
                            self,
                            policy,
                            values,
                            target_values,
                            actions_training,
                            pieces_training,
                            old_probs,
                            learning_rate,
                            ppo_epsilon=None,
                            ):
        params = self.settings["ppo_parameters"]
        clip_param, c1, c2, c3, e = params["clipping_parameter"], params["value_loss"], params["policy_loss"], params["entropy_loss"], 10**-6
        #current pi(a|s)
        r_mask = tf.reshape(tf.one_hot(actions_training[:,0,0], self.n_rotations),    (-1, self.n_rotations,    1,  1), name='r_mask')
        t_mask = tf.reshape(tf.one_hot(actions_training[:,0,1], self.n_translations), (-1,  1, self.n_translations, 1), name='t_mask')
        p_mask = tf.reshape(tf.one_hot(pieces_training[:,0,:],  self.n_pieces),       (-1,  1,  1, self.n_pieces     ), name='p_mask')
        rtp_mask = r_mask*t_mask*p_mask
        probability = tf.expand_dims(tf.reduce_sum(policy * rtp_mask, axis=[1,2,3]),1)
        if self.settings["separate_piece_values"]:
            values = tf.reduce_sum(values * p_mask, axis=[2,3])
        advantages = target_values - values
        #probability ratio
        old_prob = old_probs[:,0,:]
        r = tf.maximum(probability, e) / tf.maximum(old_prob, e)
        clipped_r = tf.clip_by_value( r, 1-clip_param, 1+clip_param )
        policy_loss = tf.minimum( r * advantages, clipped_r * advantages )
        #entropy
        entropy_bonus = action_entropy = tf.reduce_sum(N.action_entropy(policy + e) * p_mask, axis=3)
        if "entropy_floor_loss" in params:
            eps = ppo_epsilon
            n_actions = self.n_rotations * self.n_translations
            entropy_floor = -eps*tf.math.log( eps/(n_actions-1) ) -(1-eps) * tf.log(1-eps)
            entropy_bonus += params["entropy_floor_loss"] * tf.nn.relu(entropy_floor - action_entropy)
        #tally up
        self.value_loss_tf   =  c1 * tf.losses.mean_squared_error(values, target_values) #reduce loss
        self.policy_loss_tf  = -c2 * tf.reduce_mean(policy_loss) #increase expected advantages
        self.entropy_loss_tf = -c3 * tf.reduce_mean(entropy_bonus) #increase entropy
        self.regularizer_tf = self.settings["nn_regularizer"] * tf.add_n([tf.nn.l2_loss(v) for v in self.main_net.variables])
        self.loss_tf = self.value_loss_tf + self.policy_loss_tf + self.entropy_loss_tf + self.regularizer_tf
        training_ops = self.settings["optimizer"](learning_rate=self.learning_rate_tf).minimize(self.loss_tf)
        #Stats: we like stats.
        self.output_as_stats( action_entropy, name='entropy')
        self.output_as_stats( entropy_bonus, name='entropy_bonus', only_mean=True)
        self.output_as_stats( values, name='values')
        self.output_as_stats( target_values, name='target_values')
        self.output_as_stats( self.loss_tf, name='tot_loss', only_mean=True)
        self.output_as_stats( self.value_loss_tf, name='value_loss', only_mean=True)
        self.output_as_stats(-self.policy_loss_tf, name='policy_loss', only_mean=True)
        self.output_as_stats(-self.entropy_loss_tf, name='entropy_loss', only_mean=True)
        self.output_as_stats( self.regularizer_tf, name='reg_loss', only_mean=True)
        return training_ops

    @property
    def output(self):
        return self.pi_tf
