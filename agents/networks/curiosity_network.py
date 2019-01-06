import tensorflow as tf
import numpy as  np
import aux
from aux.parameter import *

class curiosity_network:
    def __init__(self, id, state_size, session, settings=None, reuse_nets=False):
        self.settings = aux.settings.default_settings.copy()
        if settings is not None:
            for x in settings:
                self.settings[x] = settings[x]
        self.name = "agent{}_curiosity".format(id)
        self.scope_name = self.name
        self.sess = session
        self.state_size = state_size

        #Whitening of the inputs/outputs
        self.input_mu = 0#None
        self.input_sigma = np.ones((1,)+self.state_size)#None
        self.output_avg = 0
        self.output_max = 1
        self.sigma_min = 10**-1

        with tf.variable_scope(self.scope_name):
            self.input_tf = tf.placeholder(tf.float32, (None,)+self.state_size, name='input_state')
            self.lr_placeholder_tf = tf.placeholder(tf.float32, shape=())
            self.loss_weights_tf = tf.placeholder(tf.float32, (None,1) , name='loss_weights')
            self.curiosity_network1 = self.create_random_network(self.input_tf, "curiosity1", trainable=False)
            self.curiosity_network2 = self.create_random_network(self.input_tf, "curiosity2", trainable=True)
            self.curiosity_reward_tf = tf.norm(self.curiosity_network1 - self.curiosity_network2, ord=self.settings["curiosity_norm"], axis=1, keepdims=True)
            self.avg_curiosity_reward_tf = tf.reduce_mean(self.curiosity_reward_tf, axis=0)
            self.max_curiosity_reward_tf = tf.reduce_max (self.curiosity_reward_tf, axis=0)
            self.input_mu_tf, _input_sigma_tf = tf.nn.moments(self.input_tf, axes=0, keep_dims=True)
            self.input_sigma_tf = tf.maximum(_input_sigma_tf, self.sigma_min)
            self.training_ops = self.create_training_ops()
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)
            self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
            self.init_ops = tf.variables_initializer(self.all_vars)
        #Run init-op
        self.sess.run(self.init_ops)

    def evaluate(self, state):
        run_list = [
                    self.curiosity_reward_tf,
                    self.avg_curiosity_reward_tf,
                    self.max_curiosity_reward_tf,
                    self.input_mu_tf,
                    self.input_sigma_tf,
                    ]
        feed_dict = {
                    self.input_tf : (state-self.input_mu)/(self.input_sigma),
                    }
        difference, out_avg, out_max, in_mu, in_sig = self.sess.run(run_list, feed_dict=feed_dict)
        # difference = difference*np.sqrt(2*self.settings["output_size"])
        reward = np.clip((difference-self.output_avg)/(self.output_max-self.output_avg) ,0,1)
        self.update_normalization(out_avg, out_max, in_mu, in_sig)
        return reward

    def train(self, state, lr=10**-6, weights=None):
        run_list = self.training_ops
        if weights is None:
            weights = [[1]]
        feed_dict = {
                    self.input_tf : (state-self.input_mu)/(self.input_sigma),
                    self.lr_placeholder_tf: lr,
                    self.loss_weights_tf : weights,
                    }
        return self.sess.run(run_list, feed_dict=feed_dict)

    def update_normalization(self, out_avg, out_max, in_mu, in_sig):
        self.output_max = max(self.output_avg, out_max)
        self.output_avg = (1-self.settings["output_mu_lr"]) * self.output_avg + self.settings["output_mu_lr"] * out_avg
        self.input_mu = (1-self.settings["input_normalization_lr"]) * self.input_mu + self.settings["input_normalization_lr"] * in_mu if self.input_mu is not None else in_mu
        self.input_sigma = (1-self.settings["input_normalization_lr"]) * self.input_sigma + self.settings["input_normalization_lr"] * in_sig if self.input_sigma is not None else in_sig

    def create_random_network(self, x, name, trainable):
        with tf.variable_scope(name):
            for i in range(self.settings["n_hidden_layers"]):
                x = tf.layers.dense(
                                    x,
                                    self.settings["layer_size"],
                                    name='layer{}'.format(i),
                                    activation=self.settings["RDN_activation"],
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                    trainable=trainable,
                                   )
            x = tf.layers.dense(
                                x,
                                self.settings["output_size"],
                                name='layer{}'.format(self.settings["n_hidden_layers"]),
                                activation=self.settings["RDN_activation"],
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=trainable,
                               )
            return x

    def create_training_ops(self):
        value_loss_tf = tf.losses.mean_squared_error(
                                                        self.curiosity_network1,
                                                        self.curiosity_network2,
                                                        weights=self.loss_weights_tf
                                                    )
        training_ops = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder_tf).minimize(value_loss_tf)
        return training_ops

    ''' Maybe implement this? '''
    # def get_weights(self, collection):
    #     output = self.session.run(collection)
    #     ret = {}
    #     for i,var in enumerate(collection):
    #         ret[var.name] = output[i]
    #     return (ret, self.scope_name)
    #
    # def set_weights(self, collection, input):
    #     weight_dict, old_scope_name = input
    #     run_list = []
    #     feed_dict = {}
    #     for var in collection:
    #         old_var_name = var.name.replace(self.scope_name, old_scope_name)
    #         # run_list.append(var.assign(weight_dict[old_var_name]))
    #         run_list.append(self.assign_placeholder_dict[var]['assign_op'])
    #         feed_dict[self.assign_placeholder_dict[var]['assign_val_placeholder']] = weight_dict[old_var_name]
    #     self.session.run(run_list, feed_dict=feed_dict)
