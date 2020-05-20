import tensorflow as tf
import numpy as np

from agents.networks import network_utils as N
from agents.networks.builders.q_net_base import q_net_base
from agents.networks.builders import build_blocks as blocks

### Easiest way to do this:
### 1) Create a class that inherits the base-class.
### 2) Forward constructor arguments to base constructor.
### 3) Implement Q_V_A(vec, vis) -> Q, V, A.
### Note: Some supporting variables are served for you via base-class constructor (look there),
###  together with a bunch of functions for building nets.
### (Base class is currently a mess, but I'm working on it!)

class q_net_silver(q_net_base):
    def __init__(self, name, output_shape, state_size, settings, worker_only=False, training=tf.constant(False, dtype=tf.bool)):
        super().__init__(self, name, output_shape, state_size, settings, worker_only=worker_only, training=training)
    def Q_V_A(self, vec, vis, one_eyed_advantage=False):
        #1) Pad visuals
        vis = [N.apply_visual_pad(v) for v in vis]
        #2) Pass visuals thru 1 res-block:
        visual_stream_resblock_settings = {
                                            n_layers : 3,
                                            n_filters : 128,
                                            filter_size : (3,3),
                                            strides : (1,1),
                                            peepholes : True,
                                            pools : False,
                                            dropout : 0.15,
                                            training : self.training_tf,
                                          }
        hidden_vis = [blocks.residual_block(v, **visual_stream_settings) for v in vis]
        #3) Make feature-planes out of vector-data and stack it on hidden stream:
        vec = [N.conv_shape_vector(v, hv.shape) for v,hv in zip(vec, hidden_vis)]
        visvec = [N.peephole_join(_vec,   _vis,   mode="concat") for _vec,_vis in zip(vec,hidden_vis)]

        #4) Another res-block before we split off in V- and A- streams.
        joined_stream_resblock_settings = {
                                            n_layers : 3,
                                            n_filters : 128,
                                            filter_size : (5,5),
                                            strides : (1,1),
                                            peepholes : True,
                                            pools : False,
                                            dropout : 0.15,
                                            training : self.training_tf,
                                          }
        joined = [blocks.residual_block(v, **joined_stream_resblock_settings) for v in visvec]

        #5) More res-blocks!
        adv_stream_resblock_settings = {
                                        n_layers : 3,
                                        n_filters : 128,
                                        filter_size : (3,3),
                                        strides : (1,1),
                                        peepholes : True,
                                        pools : False,
                                        dropout : 0.15,
                                        training : self.training_tf,
                                        }
        #Add your vector-data to my processed joined stream to compute advantges
        adv = blocks.residual_block(N.peephole_join(joined[0], vec[1]), **adv_stream_resblock_settings)
        _A = blocks.keyboard_conv(adv, self.n_rotations, self.n_pieces)

        #If we are not a worker, we want to compute values!
        if not self.worker_only:
            val_stream_resblock_settings = {
                                             n_layers : 3,
                                             n_filters : 256,
                                             filter_size : (5,5),
                                             strides : (1,1),
                                             peepholes : True,
                                             pools : True,
                                             dropout : 0.15,
                                             training : self.training_tf,
                                           }
            _V = blocks.residual_block(N.peephole_join(*joined), **val_stream_resblock_settings)
            _V = self.settings["piece_advantage_range"] * N.normalize_advantages(_V, axis=1) #This works here too!
        else:
            _V = tf.constant([0.0], dtype=tf.float32)
        V_qshaped = tf.reshape(_V,[-1,1,1,_V.shape.as_list()[-1]]) #Shape for Q-calc!

        #Finnishing touch
        A = self.settings["advantage_range"] * N.normalize_advantages(_A, separate_piece_values=self.settings["separate_piece_values"], mode=self.settings["advantage_type"], piece_mask=self.used_pieces_mask_tf)
        Q = V_qshaped + A
        V = N.q_to_v(Q, mask=self.used_pieces_mask_tf, n_pieces=self.n_pieces)
        return Q, V, A

class q_net_keyboard(q_net_base):
    def __init__(self, name, output_shape, state_size, settings, worker_only=False, training=tf.constant(False, dtype=tf.bool)):
        super().__init__(self, name, output_shape, state_size, settings, worker_only=worker_only, training=training)
    def Q_V_A(self, vector, visuals):
        #1) create visual- and vector-encoders for the inputs!
        hidden_vec = [self.create_vectorencoder(vec) for vec in vectors]
        _visuals = [self.create_visualencoder(vis) for vis in visuals]
        hidden_vis = [self.create_kbd_visual(vis) for vis in _visuals]
        flat_vec = [tf.layers.flatten(hv) for hv in hidden_vec]
        flat_vis = [tf.layers.flatten(hv) for hv in hidden_vis]
        #Prepare A. It's up here for backwards-compatibility reasons...
        A_kbd = self.create_kbd(_visuals[0]) #"my screen -> my kbd"
        #2) Take some of the data-stream and compute a value-estimate
        x = tf.concat(flat_vec+flat_vis, axis=-1)
        #By hypothesis, workers don't need the value!
        if self.worker_only:
            V = tf.constant([[0.0]], dtype=tf.float32)
        else:
            V = self.create_value_head(x)
        V_qshaped = tf.reshape(V,[-1,1,1,V.shape.as_list()[-1]]) #Shape for Q-calc!
        A = self.advantage_range * A_kbd
        #4) Combine values and advantages to form the Q-fcn (Duelling-Q-style)
        A = N.normalize_advantages(_A, separate_piece_values=self.settings["separate_piece_values"], mode=self.settings["advantage_type"], piece_mask=self.used_pieces_mask_tf)
        Q = V_qshaped + A
        V = N.q_to_v(Q, mask=self.used_pieces_mask_tf, n_pieces=self.n_pieces)
        return Q, V, A

class q_net_vanilla(q_net_base):
    def __init__(self, name, output_shape, state_size, settings, worker_only=False, training=tf.constant(False, dtype=tf.bool)):
        super().__init__(self, name, output_shape, state_size, settings, worker_only=worker_only, training=training)
    # # # # # # # # # # # # #
    def Q_V_A(self,vectors, visuals):
        #1) create visual- and vector-encoders for the inputs!
        hidden_vec = [self.create_vectorencoder(vec) for vec in vectors]
        hidden_vis = [self.create_visualencoder(vis) for vis in visuals]
        flat_vec = [tf.layers.flatten(hv) for hv in hidden_vec]
        flat_vis = [tf.layers.flatten(hv) for hv in hidden_vis]

        #2) Take some of the data-stream and compute a value-estimate
        x = tf.concat(flat_vec+flat_vis, axis=-1)
        V = self.create_value_head(x)
        V_qshaped = tf.reshape(V,[-1,1,1,V.shape.as_list()[-1]]) #Shape for Q-calc!

        #3) Compute advantages
        _A = tf.layers.dense(
                            x,
                            self.n_rotations * self.n_translations * self.n_pieces,
                            name='advantages_unshaped',
                            activation=N.advantage_activation_sqrt,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer(),
                           )
        _A = self.advantage_range * tf.reshape(_A, [-1, self.n_rotations, self.n_translations, self.n_pieces])

        #4) Combine values and advantages to form the Q-fcn (Duelling-Q-style)
        A = N.normalize_advantages(_A, separate_piece_values=self.settings["separate_piece_values"], mode=self.settings["advantage_type"], piece_mask=self.used_pieces_mask_tf)
        Q = V_qshaped + A
        V = N.q_to_v(Q, mask=self.used_pieces_mask_tf, n_pieces=self.n_pieces)
        return Q, V, A
