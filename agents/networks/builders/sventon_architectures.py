import tensorflow.compat.v1 as tf
import numpy as np

from agents.networks import network_utils as N
from agents.networks.builders.base_architecture import base_architecture
from agents.networks.builders import build_blocks as blocks
from agents.networks.builders import legacy_build_blocks as legacy_blocks

### Easiest way to do this:
### 1) Create a class that inherits the base-class.
### 2) Forward constructor arguments to base constructor.
### 3) Implement output_streams(vec, vis) -> f(s), g(s,a).
###
### Note1: Normally f(s) is V, and f(s,a) is A, but more options are possible.
### For instance, PPO uses g(s,a) as it's policy head: g(s,a) = pi(a|s).
###
### Note2: Some supporting variables are served for you via base-class constructor (look there),
###  together with a bunch of functions for building nets.
### (Base class is currently a mess, but I'm working on it!)
###
### Note3: Names are misleading; this is not just for Q-learning!

class resblock(base_architecture):
    def output_streams(self, vec, vis):
        #1) Pad visuals
        # TODO: Get more data into the stacks! (Height? )
        vis = [N.apply_visual_pad(v) for v in vis]
        if "visual_stack" in self.settings:
            vis = [N.visual_stack(v, items=self.settings["visual_stack"]) for v in vis]
        #2) Pass visuals thru 1 res-block:
        hidden_vis = [blocks.residual_block(v, **self.resblock_settings["visual"]) for v in vis]
        #3) Make feature-planes out of vector-data and stack it on hidden stream:
        vec = [N.conv_shape_vector(v, hv.shape) for v,hv in zip(vec, hidden_vis)]
        visvec = [N.peephole_join(_vec,   _vis,   mode="concat") for _vec,_vis in zip(vec,hidden_vis)]
        #4) Another res-block before we split off in V- and A- streams.
        joined = [blocks.residual_block(v, **self.resblock_settings["visvec"]) for v in visvec]
        #5) Add your vector-data to my processed joined stream then compute advantges
        # TODO: Make a little tiny resblock that gets just some data from the other screen and joins it here
        _adv_stream = N.peephole_join(joined[0], vec[1])
        _A = blocks.residual_block(_adv_stream, output_activation=None, **self.resblock_settings["adv_stream"])
        _A = self.advantage_output_fcn(_A)
        #7) If we are not a worker, we want to compute values!
        _V = tf.constant([[[[0.0]]]], dtype=tf.float32)
        if self.full_network: #Worker's need only to compute an arg-max, so we don't need the value for them :)
            #merge main-stream with visual-inputs
            vstream_in = N.peephole_join(*joined, *vis, mode="concat")
            _V = blocks.residual_block(vstream_in, **self.resblock_settings["val_stream"])
            _V = N.pool_spatial_dims_until_singleton(_V, warning=True)
            # if self.settings["separate_piece_values"]: #Treat pieces like actions
            _V = N.normalize_advantages(_V, activation=tf.nn.tanh, axis=3, inplace=True, piece_mask=self.used_pieces_mask_tf, n_used_pieces=self.n_used_pieces)
        return _V, _A
    def advantage_output_fcn(self, x):
        if self.advantage_activation_fcn is None:
            return x
        return self.advantage_activation_fcn(x)
    def initialize_variables(self):
        n = self.n_pieces+1 if (self.settings["separate_piece_values"] and self.n_pieces>1) else 1
        resb_default      = {'n_layers' : 3, 'n_filters' : 128,  'dropout' : 0.0, 'training' : self.training_tf, 'normalization' : None}
        val_resb_settings = {'n_layers' : 3, 'n_filters' : 1024, 'dropout' : 0.0, 'training' : self.training_tf, 'normalization' : None, 'output_n_filters' : n, 'filter_size' : (5,5), 'pools' : True, 'output_activation' : None, 'output_layer' : True,}
        if "residual_block_settings" not in self.settings:
                self.resblock_settings = {"visual": resb_default, "visvec": resb_default, "adv_stream" : resb_default, "val_stream": val_resb_settings,}
                return
        resb_default.update(self.settings["residual_block_settings"]["default"])
        self.resblock_settings = {"visual": resb_default.copy(), "visvec": resb_default.copy(), "adv_stream" : resb_default.copy(), "val_stream": val_resb_settings,}

        if "residual_block_settings" in self.settings:
            for key in self.resblock_settings:
                if key in self.settings["residual_block_settings"]:
                    self.resblock_settings[key].update(self.settings["residual_block_settings"][key])

class resblock_kbd(resblock):
    def advantage_output_fcn(self, x):
        return blocks.keyboard_conv(x, self.n_rotations, self.n_pieces, activation=self.kbd_activation)

class convkeyboard(base_architecture):
    def output_streams(self, vectors, visuals):
        #1) create visual- and vector-encoders for the inputs!
        hidden_vec = [legacy_blocks.create_vectorencoder(vec, self.settings) for vec in vectors]
        _visuals = [legacy_blocks.create_visualencoder(vis, self.settings) for vis in visuals]
        hidden_vis = [legacy_blocks.create_kbd_visual(vis, self.settings) for vis in _visuals]
        flat_vec = [tf.layers.flatten(hv) for hv in hidden_vec]
        flat_vis = [tf.layers.flatten(hv) for hv in hidden_vis]
        #Prepare A. It's up here for backwards-compatibility reasons...
        _A = legacy_blocks.create_kbd(_visuals[0], self.settings) #"my screen -> my kbd"
        #2) Take some of the data-stream and compute a value-estimate
        x = tf.concat(flat_vec+flat_vis, axis=-1)
        #By hypothesis, workers don't need the value!
        if self.full_network:
            V = legacy_blocks.create_value_head(x, self.settings)
        else:
            V = tf.constant([[0.0]], dtype=tf.float32)
        _V = tf.reshape(V,[-1,1,1,V.shape.as_list()[-1]]) #Shape for Q-calc!
        return _V, _A

class convthendense(base_architecture):
    def output_streams(self,vectors, visuals):
        #1) create visual- and vector-encoders for the inputs!
        hidden_vec = [legacy_blocks.create_vectorencoder(vec, self.settings) for vec in vectors]
        hidden_vis = [legacy_blocks.create_visualencoder(vis, self.settings) for vis in visuals]
        flat_vec = [tf.layers.flatten(hv) for hv in hidden_vec]
        flat_vis = [tf.layers.flatten(hv) for hv in hidden_vis]

        #2) Take some of the data-stream and compute a value-estimate
        x = tf.concat(flat_vec+flat_vis, axis=-1)
        V = legacy_blocks.create_value_head(x, self.settings)
        _V = tf.reshape(V,[-1,1,1,V.shape.as_list()[-1]]) #Shape for Q-calc!

        #3) Compute advantages
        _A = tf.layers.dense(
                            x,
                            self.n_rotations * self.n_translations * self.n_pieces,
                            name='advantages_unshaped',
                            activation=None,
                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                            bias_initializer=tf.zeros_initializer(),
                           )
        _A = self.advantage_range * tf.reshape(_A, [-1, self.n_rotations, self.n_translations, self.n_pieces])
        return _V, _A
