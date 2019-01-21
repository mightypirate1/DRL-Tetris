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
        if output_activation == "elu_plus1":
            def ep1(x):
                return tf.nn.elu(x)+1
            self.output_activation = ep1
        self.scope_name = "agent{}_{}".format(agent_id,name)
        if settings is not None:
            for x in settings:
                self.settings[x] = settings[x]
        self.state_size = state_size
        #Define tensors/placeholders
        reuse = True if reuse_nets else None
        # device = "/cpu:0" if on_cpu or not self.check_gpu() else "/device:GPU:0" #Previously this was to force CPU-placed nets on workers. Now this is regulated when creating sessions in worker threads. Unsure if current solution is good, but it runs... If you know how to do this, please contact me :) //mightypirate1
        with tf.variable_scope(self.scope_name, reuse=reuse):
            self.input_states_tf        = tf.placeholder(tf.float32, (None,)+self.state_size, name='input_state')
            self.input_s_primes_tf      = tf.placeholder(tf.float32, (None,)+self.state_size, name='input_state')
            self.input_rewards_tf       = tf.placeholder(tf.float32, (None,1), name='input_state')
            self.input_dones_tf         = tf.placeholder(tf.float32, (None,1), name='input_state')
            self.learning_rate_tf       = tf.placeholder(tf.float32, shape=())
            self.loss_weights_tf        = tf.placeholder(tf.float32, (None,1), name='loss_weights')
            self.output_values_tf, self.target_values_tf, self.new_prios_tf, main_scope, ref_scope\
                                    = self.create_prio_vnet(
                                                            self.input_states_tf,
                                                            self.input_rewards_tf,
                                                            self.input_s_primes_tf,
                                                            self.input_dones_tf
                                                           )

            self.main_scope = main_scope
            self.ref_scope = ref_scope
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

    def evaluate(self, input_states, temperature=1.0):
        run_list = self.output_values_tf
        feed_dict = {
                        self.input_states_tf : input_states,
                    }
        return_values = self.session.run(run_list, feed_dict=feed_dict)
        return return_values

    def train(self, states, rewards, s_primes, dones, weights=None, lr=None):
        if weights is None:
            weights = np.ones((input_states.shape[0],1))
        run_list = [
                    self.training_ops,
                    self.new_prios_tf,
                    ]
        feed_dict = {
                        self.input_states_tf : states,
                        self.input_rewards_tf : rewards,
                        self.input_s_primes_tf : s_primes,
                        self.input_dones_tf : dones,
                        self.learning_rate_tf : lr,
                        self.loss_weights_tf : weights,
                    }
        _, new_prios = self.session.run(run_list, feed_dict=feed_dict)
        return new_prios

    def create_value_net(self,x, name):
        with tf.variable_scope("value_net_"+name, reuse=tf.AUTO_REUSE) as vs:
            scope = vs
            for n in range(self.settings['value_head_n_hidden']):
                x = tf.layers.dense(
                                    x,
                                    self.settings['value_head_hidden_size'],
                                    name='layer{}'.format(n),
                                    activation=tf.nn.elu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                   )
            x = tf.layers.dense(
                                x,
                                1,
                                name='layer{}'.format(self.settings['value_head_n_hidden']+1),
                                activation=self.output_activation,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.contrib.layers.xavier_initializer(),
                               )
        return x, scope

    def create_prio_vnet(self, states, rewards, s_primes, dones):
        with tf.variable_scope("prio_vnet") as vs:
            values_tf, main_scope = self.create_value_net(states - 0.5, "main") #"normalize" by -.5
            v_sprime_tf, ref_scope = self.create_value_net(s_primes - 0.5, "reference")
            target_values_tf = rewards -tf.multiply(
                                                    tf.stop_gradient(
                                                                     self.settings["gamma_extrinsic"]*v_sprime_tf #we treat the target values as constant!
                                                                    ),
                                                    (1-dones)
                                                   ) #1-step empirical estimate
            prios_tf = tf.abs(values_tf - target_values_tf) #priority for the experience replay
        return values_tf, target_values_tf, prios_tf, main_scope, ref_scope

    def create_training_ops(self):
        value_loss_tf = tf.losses.mean_squared_error(self.target_values_tf, self.output_values_tf, weights=self.loss_weights_tf)
        training_ops = tf.train.AdamOptimizer(learning_rate=self.learning_rate_tf).minimize(value_loss_tf)
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
