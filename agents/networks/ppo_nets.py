
import tensorflow as tf
import numpy as np

import tools.utils as utils
from agents.networks import network_utils as N
from agents.networks.value_estimator import value_estimator
from agents.networks.advantage_normalizer import adv_normalizer
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
                                               self.output_shape,
                                               self.settings,
                                               full_network=(not worker_only or self.settings["workers_computes_advantages"]),
                                               training=self.training_tf,
                                               kbd_activation=N.action_softmax,
                                               raw_outputs=True,
                                               param_noiser=self.param_noiser,
                                              )
            self.v_tf, self.pi_tf = self.main_net(self.vector_inputs, self.visual_inputs)
            #
            if not self.worker_only: #For trainers
                self.rewards_tf             =  tf.placeholder(tf.float32, (None, k_step+1, 1), name='reward')
                self.dones_tf               =  tf.placeholder(tf.int32, (None, k_step+1, 1), name='done')
                self.actions_training_tf    =  tf.placeholder(tf.uint8, (None, 2), name='action')
                self.pieces_training_tf     =  tf.placeholder(tf.uint8, (None, 1), name='piece')
                self.probabilities_old_tf   =  tf.placeholder(tf.float32, (None, 1), name='probabilities')
                self.target_value_tf, self.advantages_tf = self.create_targets(self.v_tf)
                #params
                self.params = {
                                'ppo_epsilon'        : tf.placeholder(tf.float32, shape=(), name='ppo_epsilon'),
                                'clipping_parameter' : tf.placeholder(tf.float32, shape=(), name='clipping_parameter'),
                                'value_loss'         : tf.placeholder(tf.float32, shape=(), name='c_value_loss'),
                                'policy_loss'        : tf.placeholder(tf.float32, shape=(), name='c_policy_loss'),
                                'entropy_loss'       : tf.placeholder(tf.float32, shape=(), name='c_entropy_loss'),
                                'entropy_floor_loss' : tf.placeholder(tf.float32, shape=(), name='c_entropy_floor_loss'),
                                'lr'                 : tf.placeholder(tf.float32, shape=(), name='lr'),
                                }
                self.training_ops  = self.create_training_ops(
                                                              self.pi_tf,
                                                              self.v_tf,
                                                              self.target_value_tf,
                                                              self.advantages_tf,
                                                              self.actions_training_tf,
                                                              self.pieces_training_tf,
                                                              self.probabilities_old_tf,
                                                              self.params,
                                                             )

            self.main_net_assign_list = self.create_weight_setting_ops(self.main_net.variables)
            self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            self.init_ops = tf.variables_initializer(self.variables)
        #Run init-op
        self.session.run(self.init_ops)

    def evaluate(self, inputs, disable_noise=False):
        vector, visual = inputs
        run_list = [self.pi_tf, self.v_tf]
        feed_dict = {self.training_tf : False}
        if self.param_noiser is not None:
            feed_dict[self.param_noiser.disable_noise_tf] = disable_noise
        for idx, vec in enumerate(vector):
            feed_dict[self.vector_inputs[idx]] = vec
        for idx, vis in enumerate(visual):
            feed_dict[self.visual_inputs[idx]] = vis
        return_values = self.session.run(run_list, feed_dict=feed_dict)
        return return_values

    def train(  self,
                vector_states,
                visual_states,
                actions,
                pieces,
                probabilities,
                advantages,
                target_vals,
                rewards,
                dones,
                unfiltered_inputs=True,
                #params
                ppo_epsilon=0.0,
                clipping_parameter=0.15,
                value_loss=0.3,
                policy_loss=1.0,
                entropy_loss=0.0001,
                entropy_floor_loss=1.0,
                #
                lr=None,
              ):
        run_list = [
                    self.training_ops,
                    self.stats_tensors,
                    self.debug_tensors,
                    self.visualization_tensors,
                   ]
        if self.settings["workers_computes_advantages"]:
            val_and_adv_dict = { self.target_value_tf : target_vals[:,0,:], self.advantages_tf : advantages[:,0,:] }
        else:
            val_and_adv_dict = self.estimator.feed_dict(vector_states, visual_states)
        feed_dict = {
                        self.pieces_training_tf : pieces[:,0,:],
                        self.probabilities_old_tf : probabilities[:,0,:],
                        self.actions_training_tf : actions[:,0,:],
                        self.rewards_tf : rewards,
                        self.dones_tf : dones,
                        self.training_tf : True,
                        #params
                        self.params['ppo_epsilon']         : ppo_epsilon,
                        self.params['clipping_parameter']  : clipping_parameter,
                        self.params['value_loss']          : value_loss,
                        self.params['policy_loss']         : policy_loss,
                        self.params['entropy_loss']        : entropy_loss,
                        self.params['entropy_floor_loss']  : entropy_floor_loss,
                        self.params['lr']                  : lr,
                    }
        if self.param_noiser is not None:
            feed_dict[self.param_noiser.disable] = False
        _filter = lambda x : x[:,0,:] if unfiltered_inputs else lambda x : x
        feed_dict.update(dict(zip(self.vector_inputs, map(_filter,vector_states))))
        feed_dict.update(dict(zip(self.visual_inputs, map(_filter,visual_states))))
        feed_dict.update(val_and_adv_dict)
        #
        _, stats, dbg, vis = self.session.run(run_list, feed_dict=feed_dict)
        N.debug_prints(dbg,self.debug_tensors)
        return zip(self.stats_tensors, stats), zip(self.visualization_tensors, vis)

    def create_training_ops(
                            self,
                            policy,
                            values,
                            target_values,
                            advantages,
                            actions_training,
                            pieces_training,
                            old_probs,
                            params,
                            ):
        clip_param, c1, c2, c3, e = params["clipping_parameter"], params["value_loss"], params["policy_loss"], params["entropy_loss"], 10**-6
        #current pi(a|s)
        r_mask = tf.reshape(tf.one_hot(actions_training[:,0], self.n_rotations),    (-1, self.n_rotations,    1,  1), name='r_mask')
        t_mask = tf.reshape(tf.one_hot(actions_training[:,1], self.n_translations), (-1,  1, self.n_translations, 1), name='t_mask')
        p_mask = tf.reshape(tf.one_hot(pieces_training[:,:],  self.n_pieces),       (-1,  1,  1, self.n_pieces     ), name='p_mask')
        rtp_mask = r_mask * t_mask * p_mask
        probability = tf.expand_dims(tf.reduce_sum(policy * rtp_mask, axis=[1,2,3]),1)

        values = tf.reduce_sum(values * p_mask, axis=[2,3])
        #probability ratio
        r = tf.maximum(probability, e) / tf.maximum(old_probs, e)
        clipped_r = tf.clip_by_value( r, 1-clip_param, 1+clip_param )
        r_saturation = tf.reduce_mean(tf.cast(tf.not_equal(r, clipped_r),tf.float32))
        advnorm = adv_normalizer(0.01, safety=2.0, clip_val=4.0)
        if self.settings["normalize_advantages"]:
            advantages = advnorm(advantages)

        policy_loss = tf.minimum( r * advantages, clipped_r * advantages )
        #entropy
        entropy_bonus = action_entropy = tf.reduce_sum(N.action_entropy(policy + e) * p_mask, axis=3)
        if "entropy_floor_loss" in params:
            eps = params["ppo_epsilon"]
            n_actions = self.n_rotations * self.n_translations
            entropy_floor = -eps*tf.math.log( eps/(n_actions-1) ) -(1-eps) * tf.log(1-eps)
            extra_entropy = -tf.nn.relu(entropy_floor - action_entropy)
            entropy_bonus += params["entropy_floor_loss"] * extra_entropy
        #tally up
        self.value_loss_tf   =  c1 * tf.losses.mean_squared_error(values, target_values) #reduce loss
        self.policy_loss_tf  = -c2 * tf.reduce_mean(policy_loss) #increase expected advantages
        self.entropy_loss_tf = -c3 * tf.reduce_mean(entropy_bonus) #increase entropy
        self.regularizer_tf = self.settings["nn_regularizer"] * tf.add_n([tf.nn.l2_loss(v) for v in self.main_net.variables])
        self.loss_tf = self.value_loss_tf + self.policy_loss_tf + self.entropy_loss_tf + self.regularizer_tf
        training_ops = self.settings["optimizer"](learning_rate=params['lr']).minimize(self.loss_tf)
        #Stats: we like stats.
        self.output_as_stats( action_entropy, name='entropy')
        self.output_as_stats( entropy_bonus, name='entropy_bonus', only_mean=True)
        self.output_as_stats( values, name='values')
        self.output_as_stats( target_values, name='target_values')
        self.output_as_stats( r_saturation, name='clip_saturation', only_mean=True)
        self.output_as_stats( advnorm.a_mean, name='advantage_normalizer', only_mean=True)
        self.output_as_stats( advnorm.a_max, name='advantage_normalizer_max', only_mean=True)
        self.output_as_stats( advnorm.a_saturation, name='advantage_normalizer_saturation', only_mean=True)
        self.output_as_stats( self.loss_tf, name='tot_loss', only_mean=True)
        self.output_as_stats( self.value_loss_tf, name='value_loss', only_mean=True)
        self.output_as_stats(-self.policy_loss_tf, name='policy_loss', only_mean=True)
        self.output_as_stats(-self.entropy_loss_tf, name='entropy_loss', only_mean=True)
        self.output_as_stats( self.regularizer_tf, name='reg_loss', only_mean=True)
        for param_name in params:
            self.output_as_stats( params[param_name], name='params/'+param_name, only_mean=True)
        return [training_ops, advnorm.update_op]

    def create_targets(self, values):
        if self.settings["workers_computes_advantages"]:
            target_values_tf = tf.placeholder(tf.float32, (None,1), name='target_val_ph')
            advantages_tf    = tf.placeholder(tf.float32, (None,1), name='advantages_ph')
            self.reference_net_assign_list = self.create_weight_setting_ops([])
            self.ref_net = self.main_net
        else:
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
            target_values_tf = self.estimator.output_tf
            advantages_tf    = values - target_values_tf
        return target_values_tf, advantages_tf

    @property
    def output(self):
        return self.pi_tf
