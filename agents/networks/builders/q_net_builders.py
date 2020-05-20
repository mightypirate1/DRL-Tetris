import tensorflow as tf
import numpy as np

from agents.networks import network_utils as N

class q_net_base:
    def __init__(self, name, output_shape, state_size, settings, worker_only=False, training=tf.constant(False, dtype=tf.bool)):
        self.name = name
        self.settings = settings
        self.worker_only = worker_only
        self.training_tf = training
        #
        self.keyboard_range = self.settings["keyboard_range"]
        used_pieces = [0, 0, 0, 0, 0, 0, 0]
        for i in range(7):
            if i in self.settings["pieces"]:
                used_pieces[i] = 1
                self.used_pieces_mask_tf = tf.constant(np.array(used_pieces).reshape((1,1,1,7)).astype(np.float32))
        self.output_shape = self.n_rotations, self.n_translations, self.n_pieces = output_shape
        self.output_size = self.n_rotations * self.n_translations * self.n_pieces
        self.output_activation = settings["nn_output_activation"]
        self.state_size_vec, self.state_size_vis = state_size
        self.n_used_pieces = len(self.settings["pieces"])

    def __call__(self,vectors, visuals):
        return self.Q_V_A(vectors, visuals)

    def Q_V_A(self,vectors, visuals):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            scope = vs
            #1) create visual- and vector-encoders for the inputs!
            hidden_vec = [self.create_vectorencoder(vec) for vec in vectors]
            if self.settings["keyboard_conv"]:
                _visuals = [self.create_visualencoder(vis) for vis in visuals]
                hidden_vis = [self.create_kbd_visual(vis) for vis in _visuals]
                A_kbd = self.create_kbd(_visuals[0]) #"my screen -> my kbd"
            else:
                hidden_vis = [self.create_visualencoder(vis) for vis in visuals]
            flat_vec = [tf.layers.flatten(hv) for hv in hidden_vec]
            flat_vis = [tf.layers.flatten(hv) for hv in hidden_vis]

            #2) Take some of the data-stream and compute a value-estimate
            x = tf.concat(flat_vec+flat_vis, axis=-1)
            V = self.create_value_head(x)
            V_qshaped = tf.reshape(V,[-1,1,1,V.shape.as_list()[-1]]) #Shape for Q-calc!

            #3) Compute advantages
            if self.settings["keyboard_conv"]:
                #Or if we just have them... (if I move that line down, I break some backward compatibility)
                A = self.keyboard_range * A_kbd
            else:
                _A = tf.layers.dense(
                                    x,
                                    self.n_rotations * self.n_translations * self.n_pieces,
                                    name='advantages_unshaped',
                                    activation=N.advantage_activation_sqrt,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(),
                                   )
                A = self.keyboard_range * tf.reshape(_A, [-1, self.n_rotations, self.n_translations, self.n_pieces])

            #4) Combine values and advantages to form the Q-fcn (Duelling-Q-style)
            if self.settings["advantage_type"] == "max":
                a_maxmasked = self.used_pieces_mask_tf * A + (1-self.used_pieces_mask_tf) * tf.reduce_min(A, axis=[1,2,3], keepdims=True)
                _max_a   = tf.reduce_max(a_maxmasked,  axis=[1,2],     keepdims=True ) #We max over the actions which WE control (rotation, translation) and average over the ones we dont control (piece)
                if self.settings["separate_piece_values"]:
                    max_a = _max_a
                else:
                    max_a   = tf.reduce_sum(_max_a * self.used_pieces_mask_tf,  axis=3,     keepdims=True ) / self.n_used_pieces #We max over the actions which WE control (rotation, translation) and average over the ones we dont control (piece)
                A  = (A - max_a)  #A_q(s,r,t,p) = advantage of applying rotation r and translation t to piece p in state s; compared to playing according to the argmax-policy
                Q = V_qshaped + A
            elif self.settings["advantage_type"] == "mean":
                _mean_a = tf.reduce_mean(A,      axis=[1,2], keepdims=True )
                # mean_a  = tf.reduce_mean(_mean_a, axis=3,     keepdims=True )
                mean_a  = tf.reduce_sum(_mean_a * self.used_pieces_mask_tf, axis=3,     keepdims=True ) / self.n_used_pieces
                A = (A - mean_a)
                Q = V_qshaped + A
            else:
                Q = A
        V = self.q_to_v(Q)
        return Q, V, A, scope

    ###
    ### Above is the blue-print, below the details
    ###

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
        x = tf.layers.conv2d(
                                x,
                                self.n_rotations * self.n_pieces,
                                (x.shape.as_list()[1],3),
                                name='keyboard_conv{}'.format(self.settings["keyboard_n_convs"]-1),
                                padding='valid',
                                activation=N.advantage_activation_sqrt,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                bias_initializer=tf.zeros_initializer(),
                            )
        X = [ x[:,:,:,p*self.n_pieces:(p+1)*self.n_pieces ] for p in range(self.n_rotations) ]
        x = tf.concat(X, axis=1)
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

    def q_to_v(self,q):
        q_p = tf.reduce_max(q, axis=[1,2], keepdims=True)
        v = tf.reduce_sum(q_p * self.used_pieces_mask_tf, axis=3, keepdims=True) / self.n_pieces
        return tf.reshape(v, [-1, 1])

class q_net(q_net_base):
    '''
    TODO:
    1) Split apart the q_net_base builder into components
    2) Build K and noK as derived classes using those components
    3) Build the clever design from Silver et al
    4) Profit
    '''
    pass
