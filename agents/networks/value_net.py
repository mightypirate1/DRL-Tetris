import tensorflow as tf
import numpy as np
import aux
class value_net:
    def __init__(self, agent_id, name, state_size, sess, settings=None, output_activation=tf.nn.tanh, reuse_nets=False):
        self.settings = aux.settings.default_settings.copy()
        self.session = sess
        self.name = name
        self.output_activation = output_activation
        if output_activation == "elu_plus1":
            def ep1(x):
                return tf.nn.elu(x)+1
            self.output_activation = ep1
        self.scope_name = "agent{}_{}".format(agent_id,name)
        if settings is not None:
            for x in settings:
                self.settings[x] = settings[x]
        self.state_size = state_size
        self.dummy_state = np.zeros((1,)+state_size)
        #Define tensors/placeholders
        reuse = True if reuse_nets else None
        with tf.variable_scope(self.scope_name, reuse=reuse) as vs:
            self.softmax_temperature_tf = tf.placeholder(tf.float32, (1,), name='softmax_temperature')
            self.input_states_tf = tf.placeholder(tf.float32, (None,)+self.state_size, name='input_state')
            self.target_values_tf = tf.placeholder(tf.float32, (None,1), name='target_value')
            self.output_values_tf, self.output_probabilities_tf = self.create_value_net(self.input_states_tf)
            self.learning_rate_tf = tf.placeholder(tf.float32, shape=())
            self.loss_weights_tf = tf.placeholder(tf.float32, (None,1), name='loss_weights')
            self.training_ops = self.create_training_ops()
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)
            self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
            self.init_ops = tf.variables_initializer(self.trainable_vars)
            self.assign_placeholder_dict = self.create_weight_setting_ops()
        #Run init-op
        self.session.run(self.init_ops)

    def evaluate(self, input_states, temperature=1.0):
        run_list = [
                    self.output_values_tf,
                    self.output_probabilities_tf,
                    ]
        feed_dict = {
                        self.input_states_tf : input_states,
                        self.softmax_temperature_tf : np.array([temperature])
                    }
        return_values = self.session.run(run_list, feed_dict=feed_dict)
        return return_values

    def train(self, input_states, target_values, weights=None, lr=None):
        if weights is None:
            weights = np.ones((input_states.shape[0],1))
        n_states = len(input_states)
        run_list = [
                    self.training_ops,
                    ]
        feed_dict = {
                        self.input_states_tf : input_states,
                        self.learning_rate_tf : lr,
                        self.target_values_tf : target_values,
                        self.loss_weights_tf : weights,
                    }
        self.session.run(run_list, feed_dict=feed_dict)

    def create_value_head(self,x):
        with tf.variable_scope("value_head", reuse=tf.AUTO_REUSE) as vs:
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
        return x

    def create_value_net(self, input):
        with tf.variable_scope("value_net") as vs:
            x = input - 0.5 #"normalization"
            values_tf = self.create_value_head(x)
            probabilities_tf = tf.nn.softmax( tf.multiply(values_tf, self.softmax_temperature_tf), axis=0 )
            output_probabilities_tf = tf.div(probabilities_tf, tf.reduce_sum(probabilities_tf))
        return values_tf, output_probabilities_tf

    def create_training_ops(self):
        value_loss_tf = tf.losses.mean_squared_error(self.target_values_tf, self.output_values_tf, weights=self.loss_weights_tf)
        training_ops = tf.train.AdamOptimizer(learning_rate=self.learning_rate_tf).minimize(value_loss_tf)
        return training_ops

    def create_weight_setting_ops(self):
        assign_placeholder_dict = {}
        for var in self.all_vars:
            shape, dtype = var.shape, var.dtype
            assign_val_placeholder_tf = tf.placeholder(shape=shape, dtype=dtype)
            assign_op_tf = var.assign(assign_val_placeholder_tf)
            assign_placeholder_dict[var] = {
                                                "assign_op" : assign_op_tf,
                                                "assign_val_placeholder" : assign_val_placeholder_tf,
                                            }
        return assign_placeholder_dict

    def get_weights(self, collection):
        output = self.session.run(collection)
        ret = {}
        for i,var in enumerate(collection):
            ret[var.name] = output[i]
        return (ret, self.scope_name)

    def set_weights(self, collection, input):
        weight_dict, old_scope_name = input
        run_list = []
        feed_dict = {}
        for var in collection:
            old_var_name = var.name.replace(self.scope_name, old_scope_name)
            run_list.append(self.assign_placeholder_dict[var]['assign_op'])
            feed_dict[self.assign_placeholder_dict[var]['assign_val_placeholder']] = weight_dict[old_var_name]
        self.session.run(run_list, feed_dict=feed_dict)
