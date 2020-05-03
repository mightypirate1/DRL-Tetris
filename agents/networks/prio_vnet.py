
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import aux
import aux.utils as utils

class prio_vnet:
    def __init__(self, agent_id, name, state_size, sess, k_step=1, settings=None, worker_only=False):
        self.settings = utils.parse_settings(settings)
        self.session = sess
        self.name = name
        self.output_activation = settings["nn_output_activation"]
        self.output_size       = settings["n_value_funcions"]
        self.worker_only = worker_only
        self.stats_tensors = []
        self.scope_name = "agent{}_{}".format(agent_id,name)
        self.state_size_vec, self.state_size_vis = state_size
        self.k_step = k_step
        assert k_step > 0, "k_step AKA n_step_value_estimates has to be greater than 0!"
        #Define tensors/placeholders
        with tf.variable_scope(self.scope_name):
            if worker_only:
                self.input_rewards_tf       = tf.placeholder(tf.float32, (None,1), name='reward')
                self.input_dones_tf         = tf.placeholder(tf.float32, (None,1), name='done')
                self.vector_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
                self.visual_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
                self.vector_inputs_training = None
                self.visual_inputs_training = None
            else:
                self.vector_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
                self.visual_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
                self.vector_inputs_training = [tf.placeholder(tf.float32, (None, k_step+1)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
                self.visual_inputs_training = [tf.placeholder(tf.float32, (None, k_step+1)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
                self.input_rewards_tf       = tf.placeholder(tf.float32, (None, k_step+1, 1), name='reward')
                self.input_dones_tf         = tf.placeholder(tf.float32, (None, k_step+1, 1), name='done')
                self.learning_rate_tf       = tf.placeholder(tf.float32, shape=())

            self.loss_weights_tf        = tf.placeholder(tf.float32, (None,1), name='loss_weights')
            self.output_values_tf, self.training_values_tf, self.target_values_tf, self.new_prios_tf, self.main_scope, self.ref_scope\
                                    = self.create_prio_vnet(
                                                            self.vector_inputs,
                                                            self.visual_inputs,
                                                            self.vector_inputs_training,
                                                            self.visual_inputs_training,
                                                            self.input_rewards_tf,
                                                            self.input_dones_tf,
                                                           )
            self.main_net_vars      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.main_scope.name)
            self.main_net_assign_list      = self.create_weight_setting_ops(self.main_net_vars)
            if not worker_only:
                self.reference_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.ref_scope.name)
                self.reference_net_assign_list = self.create_weight_setting_ops(self.reference_net_vars)
                self.training_ops = self.create_training_ops()
            self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
            self.init_ops = tf.variables_initializer(self.all_vars)
        #Run init-op
        self.session.run(self.init_ops)

    def evaluate(self, inputs):
        vector, visual = inputs
        run_list = self.output_values_tf
        feed_dict = {}
        for idx, vec in enumerate(vector):
            feed_dict[self.vector_inputs[idx]] = vec
        for idx, vis in enumerate(visual):
            feed_dict[self.visual_inputs[idx]] = vis
        return_values = self.session.run(run_list, feed_dict=feed_dict)
        return return_values

    def train(self, vector_states, visual_states, rewards, dones, weights=None, lr=None):
        run_list = [
                    self.training_ops,
                    self.new_prios_tf,
                    self.stats_tensors,
                    ]
        feed_dict = {
                        self.input_rewards_tf : rewards,
                        self.input_dones_tf : dones,
                        self.learning_rate_tf : lr,
                        self.loss_weights_tf : weights,
                    }
        #Add also the state information to the appropriate placeholders..
        for idx, vec in enumerate(vector_states):
            feed_dict[self.vector_inputs_training[idx]] = vec
        for idx, vis in enumerate(visual_states):
            feed_dict[self.visual_inputs_training[idx]] = vis
        for idx, vec in enumerate(vector_states):
            feed_dict[self.vector_inputs[idx]] = vec[:,0,:]
        for idx, vis in enumerate(visual_states):
            feed_dict[self.visual_inputs[idx]] = vis[:,0,:]
        _, new_prios, stats = self.session.run(run_list, feed_dict=feed_dict)
        return new_prios, zip(self.stats_tensors, stats)

    def create_vectorencoder(self, x):
        with tf.variable_scope("vectorencoder", reuse=tf.AUTO_REUSE) as vs:
            for n in range(self.settings['vectorencoder_n_hidden']):
                x = tf.layers.dense(
                                    x,
                                    self.settings['vectorencoder_hidden_size'],
                                    name='vectorencoder_layer{}'.format(n),
                                    activation=tf.nn.elu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                   )
            x = tf.layers.dense(
                                x,
                                self.settings['vectorencoder_output_size'],
                                name='layer{}'.format(self.settings['vectorencoder_n_hidden']+1),
                                activation=self.output_activation,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.contrib.layers.xavier_initializer(),
                               )
        return x

    def create_visualencoder(self, x):
        with tf.variable_scope("visualencoder", reuse=tf.AUTO_REUSE) as vs:
            if self.settings["pad_visuals"]:
                x = self.apply_visual_pad(x)
            for n in range(self.settings['visualencoder_n_convs']):
                y = tf.layers.conv2d(
                                        x,
                                        self.settings["visualencoder_n_filters"][n],
                                        self.settings["visualencoder_filter_sizes"][n],
                                        name='visualencoder_layer{}'.format(n),
                                        padding='same',
                                        activation=tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        bias_initializer=tf.zeros_initializer(),
                                    )
                if n in self.settings["visualencoder_peepholes"] and self.settings["peephole_convs"]:
                    x = tf.concat([y,x], axis=-1)
                else:
                    x = y
                if n in self.settings["visualencoder_poolings"]:
                    y = tf.layers.max_pooling2d(y, 2, 2, padding='same')
        return x

    def apply_visual_pad(self, x):
        #Apply zero-padding on top:
        x = tf.pad(x, [[0,0],[1,0],[0,0],[0,0]], constant_values=0.0)
        #Apply one-padding left, right and bottom:
        x = tf.pad(x, [[0,0],[0,1],[1,1],[0,0]], constant_values=1.0)
        # This makes floor and walls look like it's a piece, and cieling like its free space
        return x

    def create_value_net(self,vectors, visuals, name):
        with tf.variable_scope("valuened_"+name, reuse=tf.AUTO_REUSE) as vs:
            scope = vs
            hidden_vec = [self.create_vectorencoder(vec) for vec in vectors]
            hidden_vis = [self.create_visualencoder(vis) for vis in visuals]
            flat_vec = [tf.layers.flatten(hv) for hv in hidden_vec]
            flat_vis = [tf.layers.flatten(hv) for hv in hidden_vis]
            x = tf.concat(flat_vec+flat_vis, axis=-1)
            for n in range(self.settings['valuenet_n_hidden']):
                x = tf.layers.dense(
                                    x,
                                    self.settings['valuenet_hidden_size'],
                                    name='valuenet_layer{}'.format(n),
                                    activation=tf.nn.elu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                   )
            x = tf.layers.dense(
                                x,
                                self.output_size,
                                name='valuenet_layer{}'.format(self.settings['valuenet_n_hidden']+1),
                                activation=self.output_activation,
                                # kernel_initializer=tf.zeros_initializer(),
                                bias_initializer=tf.zeros_initializer(),
                               )

            ret = x
        return ret, scope

    def create_prio_vnet(self, vector_states, visual_states, vector_states_training, visual_states_training, rewards, dones):
        with tf.variable_scope("prio_vnet") as vs:
            values_tf, main_scope = self.create_value_net(vector_states, visual_states, "main")

            #Workers are easy!
            if self.worker_only:
                return values_tf, None, None, None, main_scope, None

            # # #
            # Trainers also do {1,...,k}-step value estimators in 2 steps:
            # # #

            #1) get all the values and a bool to tell if the round is over
            val_i_tf = [0 for _ in range(self.k_step+1)]
            dones_tf = tf.minimum(1.0, tf.cumsum(dones, axis=1)) #Maximum ensures there is no stray done from an adjacent trajectory influencing us...
            done_time_tf = tf.reduce_sum( 1-dones_tf, axis=1)
            for i in range(self.k_step+1):
                subscope = "main" if i == 0 else "reference" #shared weights in scope!
                subinputs_vec = [vec_state[:,i,:] for vec_state in vector_states_training]
                subinputs_vis = [vis_state[:,i,:] for vis_state in visual_states_training]
                val_i_tf[i], ref_scope = self.create_value_net(subinputs_vec, subinputs_vis, subscope)
            #2) Combine rewards, values and dones into estimates
            gamma = -self.settings["gamma"] if self.settings["single_policy"] else self.settings["gamma"]
            def k_step_estimate(k):
                e = 0
                for t in range(k):
                    e += rewards[:,t,:] * tf.cast((done_time_tf >= t),tf.float32) * gamma**t
                e += val_i_tf[k] * tf.cast((done_time_tf > k),tf.float32) * gamma**k
                return e
            estimators_tf = [k_step_estimate(k) for k in range(1,self.k_step+1)]
            #3) GAE-style aggregation
            weight = 0
            estimator_sum_tf = 0
            gae_lambda = self.settings["gae_lambda"]
            for e in reversed(estimators_tf):
                estimator_sum_tf *= gae_lambda
                weight *= gae_lambda
                estimator_sum_tf += e
                weight += 1
            target_values_tf = tf.stop_gradient(estimator_sum_tf / weight)

            # Sometimes we will also want to output the value from the main-net:
            training_values_tf = val_i_tf[0]

            # Prio-V-net is true to it's name and computes the sample priorities for you!
            if self.settings["optimistic_prios"] == 0.0:
                prios_tf = tf.abs(training_values_tf - target_values_tf) #priority for the experience replay
            else:
                prios_tf = tf.abs(training_values_tf - target_values_tf) + self.settings["optimistic_prios"] * tf.nn.relu(target_values_tf - training_values_tf)

            #As always - we like some stats!
            self.output_as_stats(target_values_tf, name="gae-val")
            self.output_as_stats(done_time_tf, name="done_time")
            for i,e in enumerate(estimators_tf):
                self.output_as_stats(e, name='{}_step_val_est'.format(i+1))
                self.output_as_stats(tf.abs(target_values_tf-e), name='{}-step_val_est_diff'.format(i+1))
        return values_tf, training_values_tf, target_values_tf, prios_tf, main_scope, ref_scope

    def create_training_ops(self):
        if self.worker_only:
            return None
        self.value_loss_tf = tf.losses.mean_squared_error(self.target_values_tf, self.training_values_tf, weights=self.loss_weights_tf)
        self.regularizer_tf = self.settings["nn_regularizer"] * tf.add_n([tf.nn.l2_loss(v) for v in self.main_net_vars])
        self.loss_tf = self.value_loss_tf + self.regularizer_tf
        training_ops = self.settings["optimizer"](learning_rate=self.learning_rate_tf).minimize(self.loss_tf)
        #Stats: we like stats.
        self.output_as_stats(self.loss_tf, name='tot_loss', only_mean=True)
        self.output_as_stats(self.value_loss_tf, name='value_loss', only_mean=True)
        self.output_as_stats(self.regularizer_tf, name='reg_loss', only_mean=True)
        return training_ops

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

    def check_gpu(self):
        for dev in device_lib.list_local_devices():
            if "GPU" in dev.name:
                return True
        return False

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

    @property
    def output(self):
        return self.output_values_tf
