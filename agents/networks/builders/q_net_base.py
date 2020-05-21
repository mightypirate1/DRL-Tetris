import tensorflow as tf
import numpy as np

from agents.networks.builders import build_blocks as blocks
from agents.networks import network_utils as N

# This file survives only due to backwards compatibility (I can not compare to old training settings unless I keep it..).
# I wish I knew then what I know now :)

class q_net_base:
    def __init__(self, name, output_shape, state_size, settings, worker_only=False, training=tf.constant(False, dtype=tf.bool)):
        self.name = name
        self.settings = settings
        self.worker_only = worker_only
        self.training_tf = training
        self.output_shape = self.n_rotations, self.n_translations, self.n_pieces = output_shape
        self.output_size = self.n_rotations * self.n_translations * self.n_pieces
        self.output_activation = settings["nn_output_activation"]
        self.state_size_vec, self.state_size_vis = state_size
        self.advantage_range = self.settings["advantage_range"]
        self.n_used_pieces = len(self.settings["pieces"])
        self.used_pieces_mask_tf = self.create_used_pieces_mask()
        self.initialize_variables()
    def initialize_variables(self):
        pass
    def __call__(self,vectors, visuals, *args, **kwargs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            scope = vs
            Q, V, A = self.Q_V_A(vectors, visuals, *args, **kwargs)
            return Q, V, A, scope

    def Q_V_A(self, *args, **kwargs):
        assert False, "Don't apply the base-class!"

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
                x = N.apply_visual_pad(x)
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
                if "visualencoder_dropout" in self.settings:
                    x = tf.keras.layers.SpatialDropout2D(self.settings["visualencoder_dropout"])(x,self.training_tf)
                if n in self.settings["visualencoder_peepholes"] and self.settings["peephole_convs"]:
                    x = N.peephole_join(x,y,mode=self.settings["peephole_join_style"])
                else:
                    x = y
                if n in self.settings["visualencoder_poolings"]:
                    x = tf.layers.max_pooling2d(x, (2,1), (2,1), padding='same')
        return x

    def create_kbd_visual(self,x):
        x = tf.layers.max_pooling2d(x, 2, 2, padding='valid')
        for i in range(self.settings["kbd_vis_n_convs"]):
            x = N.layer_pool(
                            x,
                            n_filters=self.settings["kbd_vis_n_filters"][i],
                            filter_size=(3,3),
                            dropout=(self.settings["visualencoder_dropout"],self.training_tf)
                            )
        x = tf.reduce_mean(x, axis=[1,2])
        return x

    def create_kbd(self, x):
        for i in range(self.settings["keyboard_n_convs"]-1):
            x = tf.layers.conv2d(
                                    x,
                                    self.settings["keyboard_n_filters"][i],
                                    (5,5),
                                    name='keyboard_conv{}'.format(i),
                                    padding='same',
                                    activation=tf.nn.elu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    bias_initializer=tf.zeros_initializer(),
                                )
            if "visualencoder_dropout" in self.settings:
                x = tf.keras.layers.Dropout(self.settings["visualencoder_dropout"])(x,self.training_tf)
        x = blocks.keyboard_conv(x, self.n_rotations, self.n_pieces, name='keyboard_conv{}'.format(self.settings["keyboard_n_convs"]-1))
        return x
        # #Interpret with of field as translations for the piece W ~> T, then:
        # # [?, 1, T, R*P] -> [?, T, R, P] -> [?, R, T, P]
        # x = tf.reshape(x, [-1, 10, 4, 7])
        # x = tf.transpose(x, perm=[0,2,1,3])
        # return x

    def create_value_head(self, x):
        for n in range(self.settings['valuenet_n_hidden']):
            x = tf.layers.dense(
                                x,
                                self.settings['valuenet_hidden_size'],
                                name='layer{}'.format(n),
                                activation=tf.nn.elu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer(),
                               )
        v = tf.layers.dense(
                            x,
                            1,
                            name='values',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            activation=self.settings["nn_output_activation"],
                            bias_initializer=tf.zeros_initializer(),
                           )
        if not self.settings["separate_piece_values"]:
            return v
        v_p = tf.layers.dense(
                              x,
                              7,
                              name='piece_values',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              activation=None,
                              bias_initializer=tf.zeros_initializer(),
                             )
        v_p = self.settings["piece_advantage_range"] * N.advantage_activation_sqrt(v_p - tf.reduce_mean(v_p, axis=1, keepdims=True))
        return v + v_p

    def create_used_pieces_mask(self):
        used_pieces = [0, 0, 0, 0, 0, 0, 0]
        for i in range(7):
            if i in self.settings["pieces"]:
                used_pieces[i] = 1
        return tf.constant(np.array(used_pieces).reshape((1,1,1,7)).astype(np.float32))
