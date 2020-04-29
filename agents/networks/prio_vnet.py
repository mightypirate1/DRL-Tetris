
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
        self.worker_only = worker_only
        self.scope_name = "agent{}_{}".format(agent_id,name)
        self.state_size_vec, self.state_size_vis = state_size
        self.k_step = k_step
        assert k_step > 0, "k_step AKA n_step_value_estimates has to be greater than 0!"
        #Define tensors/placeholders
        with tf.variable_scope(self.scope_name):
            if worker_only:
                self.vector_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
                self.visual_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
                self.input_rewards_tf       = tf.placeholder(tf.float32, (None,1), name='reward')
                self.input_dones_tf         = tf.placeholder(tf.float32, (None,1), name='done')
            else:
                self.vector_inputs          = [tf.placeholder(tf.float32, (None, k_step)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
                self.visual_inputs          = [tf.placeholder(tf.float32, (None, k_step)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
                self.input_rewards_tf       = tf.placeholder(tf.float32, (None, k_step, 1), name='reward')
                self.input_dones_tf         = tf.placeholder(tf.float32, (None, k_step, 1), name='done')
                self.learning_rate_tf       = tf.placeholder(tf.float32, shape=())

            self.loss_weights_tf        = tf.placeholder(tf.float32, (None,1), name='loss_weights')
            self.output_values_tf, self.target_values_tf, self.new_prios_tf, self.main_scope, self.ref_scope\
                                    = self.create_prio_vnet(
                                                            self.vector_inputs,
                                                            self.visual_inputs,
                                                            self.input_rewards_tf,
                                                            self.input_dones_tf
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

    def check_gpu(self):
        for dev in device_lib.list_local_devices():
            if "GPU" in dev.name:
                return True
        return False

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

    def compute_prios(self, s, s_prime, rewards, dones):
        assert not self.worker_only, "IMPLEMENT COMPUTE PRIOS"
        vector_states, visual_states = s
        vector_s_primes, visual_s_primes = s_prime
        run_list = self.new_prios_tf
        feed_dict = {
                        self.input_rewards_tf : rewards,
                        self.input_dones_tf : dones,
                    }
        #Add also the state information to the appropriate placeholders..
        for idx, vec in enumerate(vector_states):
            feed_dict[self.vector_inputs[idx]] = vec
        for idx, vis in enumerate(visual_states):
            feed_dict[self.visual_inputs[idx]] = vis
        new_prios = self.session.run(run_list, feed_dict=feed_dict)
        return new_prios

    def train(self, vector_states, visual_states, vector_s_primes, visual_s_primes, rewards, dones, weights=None, lr=None):
        assert not self.worker_only, "IMPLEMENT TRAIN"
        if weights is None:
            weights = np.ones((input_states.shape[0],1))
        run_list = [
                    self.training_ops,
                    self.new_prios_tf,
                    self.loss_tf,
                    self.value_loss_tf,
                    self.regularizer_tf,
                    ]
        feed_dict = {
                        self.input_rewards_tf : rewards,
                        self.input_dones_tf : dones,
                        self.learning_rate_tf : lr,
                        self.loss_weights_tf : weights,
                    }
        #Add also the state information to the appropriate placeholders..
        for idx, vec in enumerate(vector_states):
            feed_dict[self.vector_inputs[idx]] = vec
        for idx, vis in enumerate(visual_states):
            feed_dict[self.visual_inputs[idx]] = vis
        for idx, vec in enumerate(vector_s_primes):
            feed_dict[self.vector_s_primes[idx]] = vec
        for idx, vis in enumerate(visual_s_primes):
            feed_dict[self.visual_s_primes[idx]] = vis
        _, new_prios, loss, v_loss, reg_loss = self.session.run(run_list, feed_dict=feed_dict)
        return new_prios, (loss, v_loss, reg_loss)

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
                                1,
                                name='valuenet_layer{}'.format(self.settings['valuenet_n_hidden']+1),
                                activation=self.output_activation,
                                # kernel_initializer=tf.zeros_initializer(),
                                bias_initializer=tf.zeros_initializer(),
                               )

            ret = x
        return ret, scope

    def create_prio_vnet(self, vector_states, visual_states, rewards, dones):
        with tf.variable_scope("prio_vnet") as vs:
            if self.worker_only:
                #Workers are easy!
                values_tf, main_scope = self.create_value_net(vector_states, visual_states, "main")
                return values_tf, None, None, main_scope, None
            else:
                #k-step value estimator in 2 steps:
                #1) get all the values and a bool to tell if the round is over
                done = tf.constant(0, shape=(1,1))
                values_tf = [0 for _ in range(self.k_step+1)]
                dones_tf =  [0 for _ in range(self.k_step+1)]
                for i in range(self.k_step+1):
                    subscope = "main" if i == 0 else "reference"
                    subinputs_vec = [vs[:,i,] for fs in vector_states]
                    subinputs_vis = [vs[:,i,] for fs in visual_states]
                    values_tf[i], scope_i = self.create_value_net(subinputs_vec, subinputs_vis, subscope)
                    dones_tf[i] = tf.maximum(done, dones[:,i-1,:]) if i>0 else 0
                    main_scope = scope_i if i == 0 else main_scope
                    ref_scope = scope_i
                #2) Combine rewards, values and dones into estimates
                estimators_tf = []
                for K in range(1, self.k_step+1):
                    #k-step esimate:
                    e = 0
                    for i in range(K):
                        d = dones_tf[i] #== (game over BEFORE i)
                        e += (1-d)*rewards[:,i,:]*gamma**i
                    e += (1-dones_tf[K])*(gamma**K)*values_tf[K]

            target_values_tf = rewards + gamma * tf.multiply(
                                                              tf.stop_gradient(
                                                                               v_sprime_tf #we treat the target values as constant!
                                                                               ),
                                                              (1-dones)
                                                              ) #1-step empirical estimate
            if self.settings["optimistic_prios"] == 0.0:
                prios_tf = tf.abs(values_tf - target_values_tf) #priority for the experience replay
            else:
                prios_tf = tf.abs(values_tf - target_values_tf) + self.settings["optimistic_prios"] * tf.nn.relu(target_values_tf - values_tf)
        return values_tf, target_values_tf, prios_tf, main_scope, ref_scope

    def create_training_ops(self):
        if self.worker_only:
            return None
        self.value_loss_tf = tf.losses.mean_squared_error(self.target_values_tf, self.output_values_tf, weights=self.loss_weights_tf)
        self.regularizer_tf = self.settings["nn_regularizer"] * tf.add_n([tf.nn.l2_loss(v) for v in self.main_net_vars])
        self.loss_tf = self.value_loss_tf + self.regularizer_tf
        training_ops = self.settings["optimizer"](learning_rate=self.learning_rate_tf).minimize(self.loss_tf)
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
        #This makes it equivalent to:
        # tmp = reference_net
        # reference_net = main_net
        # main_net = tmp
        #if only I could do it that ez... :)
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

    @property
    def output(self):
        return self.output_values_tf
