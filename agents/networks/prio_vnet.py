
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import aux
import aux.utils as utils

class prio_vnet:
    def __init__(self, agent_id, name, state_size, sess, on_cpu=False, settings=None, output_activation=tf.nn.tanh, reuse_nets=False, disable_reference_net=False):
        self.settings = utils.parse_settings(settings)
        self.session = sess
        self.name = name
        self.output_activation = output_activation
        self.disable_reference_net = disable_reference_net
        self.scope_name = "agent{}_{}".format(agent_id,name)
        if settings is not None:
            for x in settings:
                self.settings[x] = settings[x]
        self.state_size_vec, self.state_size_vis = state_size
        #Define tensors/placeholders
        reuse = True if reuse_nets else None
        # device = "/cpu:0" if on_cpu or not self.check_gpu() else "/device:GPU:0" #Previously this was to force CPU-placed nets on workers. Now this is regulated when creating sessions in worker threads. Unsure if current solution is good, but it runs... If you know how to do this, please contact me :) //mightypirate1
        with tf.variable_scope(self.scope_name, reuse=reuse):
            self.vector_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
            self.visual_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
            self.vector_s_primes        = [tf.placeholder(tf.float32, (None,)+s[1:], name='vector_s_prime_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
            self.visual_s_primes        = [tf.placeholder(tf.float32, (None,)+s[1:], name='visual_s_prime_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
            self.input_rewards_tf       = tf.placeholder(tf.float32, (None,1), name='reward')
            self.input_dones_tf         = tf.placeholder(tf.float32, (None,1), name='done')
            self.learning_rate_tf       = tf.placeholder(tf.float32, shape=())
            self.loss_weights_tf        = tf.placeholder(tf.float32, (None,1), name='loss_weights')
            self.output_values_tf, self.target_values_tf, self.new_prios_tf, main_scope, ref_scope\
                                    = self.create_prio_vnet(
                                                            self.vector_inputs,
                                                            self.visual_inputs,
                                                            self.vector_s_primes,
                                                            self.visual_s_primes,
                                                            self.input_rewards_tf,
                                                            self.input_dones_tf
                                                           )

            self.ref_scope  = ref_scope
            self.main_scope = main_scope
            self.main_net_vars      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=main_scope.name)
            self.reference_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=ref_scope.name)

            self.main_net_assign_list      = self.create_weight_setting_ops(self.main_net_vars)
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

    def train(self, vector_states, visual_states, vector_s_primes, visual_s_primes, rewards, dones, weights=None, lr=None):
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
            for n in range(self.settings['visualencoder_n_convs']):
                x = tf.layers.conv2d(
                                        x,
                                        self.settings["visualencoder_n_filters"][n],
                                        self.settings["visualencoder_filter_sizes"][n],
                                        name='visualencoder_layer{}'.format(n),
                                        padding='same',
                                        activation=tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        bias_initializer=tf.zeros_initializer(),
                                    )
                if n in self.settings["visualencoder_poolings"]:
                    x = tf.layers.max_pooling2d(x, 2, 2, padding='same')
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
                                kernel_initializer=tf.zeros_initializer,
                                bias_initializer=tf.zeros_initializer,
                               )

            ret = x
        return ret, scope

    def create_prio_vnet(self, vector_states, visual_states, vector_s_primes, visual_s_primes,  rewards, dones):
        with tf.variable_scope("prio_vnet") as vs:
            values_tf, main_scope = self.create_value_net(vector_states, visual_states, "main")
            v_sprime_tf, ref_scope = self.create_value_net(vector_s_primes, visual_s_primes, "reference")
            target_values_tf = rewards -tf.multiply(
                                                    tf.stop_gradient(
                                                                     self.settings["gamma_extrinsic"]*v_sprime_tf #we treat the target values as constant!
                                                                    ),
                                                    (1-dones)
                                                   ) #1-step empirical estimate
            prios_tf = tf.abs(values_tf - target_values_tf) #priority for the experience replay
        return values_tf, target_values_tf, prios_tf, main_scope, ref_scope

    def create_training_ops(self):
        self.value_loss_tf = tf.losses.mean_squared_error(self.target_values_tf, self.output_values_tf, weights=self.loss_weights_tf)
        self.regularizer_tf = self.settings["nn_regularizer"] * tf.add_n([tf.nn.l2_loss(v) for v in self.main_net_vars])
        self.loss_tf = self.value_loss_tf + self.regularizer_tf
        training_ops = tf.train.AdamOptimizer(learning_rate=self.learning_rate_tf).minimize(self.loss_tf)
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
