import tensorflow as tf

default_settings = {
                    "n_hidden_layers" : 4,
                    "layer_size" : 429,
                    "output_size" : 40,
                    "activation" : tf.nn.tanh,
                    "curiosity_norm" : 2,
                    "lr" : 0.001,
                    }

class curiosity_network:
    def __init__(self, state_size, session, name="ape", settings=None):
        if settings is None:
            settings = default_settings
        self.settings = settings
        self.name = name
        self.scope_name = self.name+"curiosity"
        self.sess = session
        self.state_size = state_size

        with tf.variable_scope(self.scope_name):
            self.input_tf = tf.placeholder(tf.float32, (None,)+self.state_size, name='input_state')
            self.loss_weights_tf = tf.placeholder(tf.float32, (None,1) , name='loss_weights')
            self.curiosity_network1 = self.create_random_network(self.input_tf, "curiosity1", trainable=False)
            self.curiosity_network2 = self.create_random_network(self.input_tf, "curiosity2", trainable=True)
            self.curiosity_reward = tf.norm(self.curiosity_network1 - self.curiosity_network2, ord=self.settings["curiosity_norm"], axis=1)
            self.training_ops = self.create_training_ops()
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)
            self.init_ops = tf.global_variables_initializer()#tf.variables_initializer(self.trainable_vars)
        #Run init-op
        self.sess.run(self.init_ops)

    def evaluate(self, state):
        run_list = self.curiosity_reward
        feed_dict = {
                    self.input_tf : state,
                    }
        return self.sess.run(run_list, feed_dict=feed_dict)

    def train(self, state, weights=None):
        run_list = self.training_ops
        if weights is None:
            weights = [[1]]
        feed_dict = {
                    self.input_tf : state,
                    self.loss_weights_tf : weights,
                    }
        return self.sess.run(run_list, feed_dict=feed_dict)

    def create_random_network(self, x, name, trainable):
        with tf.variable_scope(name):
            for i in range(self.settings["n_hidden_layers"]):
                x = tf.layers.dense(
                                    x,
                                    self.settings["layer_size"],
                                    name='layer{}'.format(i),
                                    activation=self.settings["activation"],
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                    trainable=trainable,
                                   )
            x = tf.layers.dense(
                                x,
                                self.settings["output_size"],
                                name='layer{}'.format(self.settings["n_hidden_layers"]),
                                activation=self.settings["activation"],
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
        training_ops = tf.train.AdamOptimizer(learning_rate=self.settings['lr']).minimize(value_loss_tf)
        return training_ops
