import tensorflow as tf
from tensorflow.python.client import device_lib
import aux.utils as utils
import numpy as np
import aux
class pg_net:
    def __init__(self, agent_id, name, state_size, sess, settings=None):
        self.settings = utils.parse_settings(settings)
        self.session = sess
        self.name = name
        self.n_actions = n = self.settings["n_actions"]
        self.scope_name = "agent{}_{}".format(agent_id,name)
        if settings is not None:
            for x in settings:
                self.settings[x] = settings[x]
        self.state_size = state_size
        #Define tensors/placeholders
        with tf.variable_scope(self.scope_name):
            #Input tensors
            self.input_future_states_tf  = tf.placeholder(dtype=tf.float32, shape=(None, n)+self.state_size, name='input_state'  )
            self.n_future_states_mask_tf = tf.placeholder(dtype=tf.float32, shape=(None, n),                 name='states_mask'  )
            self.actions_tf              = tf.placeholder(dtype=tf.uint8,   shape=(None, 1),                 name='actions'      )
            self.advantages_tf           = tf.placeholder(dtype=tf.float32, shape=(None, 1),                 name='advantages'   )
            self.target_values_tf        = tf.placeholder(dtype=tf.float32, shape=(None, 1),                 name='target_values')
            self.old_probabilities_tf    = tf.placeholder(dtype=tf.float32, shape=(None, 1),                 name='old_probs'    )
            self.probabilities_tf, self.values_tf = self.create_net(self.input_future_states_tf)
            #Parameter tensors
            self.lr_tf      = tf.placeholder(dtype=tf.float32, shape=(), name='lr')
            self.epsilon_tf = tf.placeholder(dtype=tf.float32, shape=(), name='epsilon')

            self.training_ops = self.create_training_ops()
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)
            self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
            self.init_ops = tf.variables_initializer(self.all_vars)
            self.assign_placeholder_dict = self.create_weight_setting_ops()
        #Run init-op
        self.session.run(self.init_ops)

    def check_gpu(self):
        for dev in device_lib.list_local_devices():
            if "GPU" in dev.name:
                return True
        return False

    def evaluate(self, future_states, future_states_mask):
        run_list = [
                    self.probabilities_tf,
                    self.values_tf
                    ]
        feed_dict = {
                        self.input_future_states_tf : future_states,
                        self.n_future_states_mask_tf : future_states_mask,
                    }
        return_values = self.session.run(run_list, feed_dict=feed_dict)
        return return_values

    def train(self, future_states, future_states_mask, actions, advantages, target_values, old_probabilities, epsilon, lr):
            run_list = [self.training_ops, self.loss_clip_tf, self.loss_entropy_tf, self.loss_value_tf, self.loss_tf]
            feed_dict = {
                            #Inputs
                            self.input_future_states_tf : future_states,
                            self.n_future_states_mask_tf : future_states_mask,
                            self.actions_tf : actions,
                            self.advantages_tf : advantages,
                            self.target_values_tf : target_values,
                            self.old_probabilities_tf : old_probabilities,
                            #Parameters
                            self.lr_tf : lr,
                            self.epsilon_tf : epsilon,
                        }
            return self.session.run(run_list, feed_dict=feed_dict)

    def create_state_head(self,x):
        with tf.variable_scope("value_head", reuse=tf.AUTO_REUSE) as vs:
            for n in range(self.settings['state_head_n_hidden']):
                x = tf.layers.dense(
                                    x,
                                    self.settings['state_head_hidden_size'],
                                    name='layer{}'.format(n),
                                    activation=tf.nn.elu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                   )
            x = tf.layers.dense(
                                x,
                                self.settings['state_head_output_size'],
                                name='layer{}'.format(self.settings['state_head_n_hidden']+1),
                                activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.contrib.layers.xavier_initializer(),
                               )
        return x

    def create_net(self, input):
        with tf.variable_scope("pg_net") as vs:
            hidden = [tf.reshape(self.n_future_states_mask_tf[:,i], [-1, 1]) * self.create_state_head(input[:,i,:]) for i in range(self.n_actions)]
            hidden = tf.concat(hidden, axis=-1)
            x = hidden
            for n in range(self.settings['joined_n_hidden']):
                x = tf.layers.dense(
                                    x,
                                    self.settings['joined_hidden_size'],
                                    name='joining_layer{}'.format(n),
                                    activation=tf.nn.elu, # <--- NOTE: ELU
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                   )
            p_head = tf.layers.dense(
                                     x,
                                     self.n_actions,
                                     name='policy_unmasked',
                                     activation=tf.nn.softmax,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer(),
                                    )
            v_head = tf.layers.dense(
                                     x,
                                     1,
                                     name='value_head',
                                     activation=tf.nn.elu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer(),
                                    )
            _probabilities_tf = p_head * self.n_future_states_mask_tf
            probabilities_tf = tf.div(_probabilities_tf, tf.reduce_sum(_probabilities_tf, axis=1,keepdims=True))
            values_tf = v_head
        return probabilities_tf, values_tf

    def create_training_ops(self):
        with tf.variable_scope("Training_ops"):
            #Fudge it up so it doesnt inf/nan...
            e = 10**-7
            probs = tf.maximum(self.probabilities_tf, e) * self.n_future_states_mask_tf
            old_probs = tf.maximum(self.old_probabilities_tf, e)
            action_1h = tf.one_hot(self.actions_tf, self.n_actions)
            #Define some intermediate tensors...
            log_proof_probs = probs + (1-self.n_future_states_mask_tf) #This makes probabilities 1 if they were masked out. log(1)=0, so they dont contribute to the entropy anyway...
            entropy_tf = tf.reduce_sum(-tf.multiply(probs, tf.log(log_proof_probs)), axis=1)
            action_prob_tf = tf.reduce_sum(tf.multiply(action_1h, probs), axis=1, keepdims=True)
            ratio_tf = tf.div( action_prob_tf , old_probs )
            ratio_clipped_tf = tf.clip_by_value(ratio_tf, 1-self.epsilon_tf, 1+self.epsilon_tf)
            #Define the loss tensors!
            self.loss_clip_tf = tf.reduce_mean(tf.minimum( tf.multiply(ratio_tf,self.advantages_tf), tf.multiply(ratio_clipped_tf,self.advantages_tf) ) )
            self.loss_entropy_tf = tf.reduce_mean(entropy_tf)
            self.loss_value_tf = tf.losses.mean_squared_error(self.values_tf, self.target_values_tf)
            self.loss_tf = - self.settings["weight_loss_policy"]  * self.loss_clip_tf      \
                           - self.settings["weight_loss_entropy"] * self.loss_entropy_tf   \
                           + self.settings["weight_loss_value"]   * self.loss_value_tf
        #Minimize loss!
        return tf.train.AdamOptimizer(learning_rate=self.lr_tf).minimize(self.loss_tf)

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
