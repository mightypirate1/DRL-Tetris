import tensorflow.compat.v1 as tf
import numpy as np
from agents.networks import network_utils as N

# Theis classes exists to be the interface between architectures and networks:
#  - Networks are the inputs/outputs and training-rules that agents/trainers used.
#  - Architectures are how data is transformed within a network.
#
# A network can use an architecture to map it's states (vis-vec pairs) to outputs.
# Outputs have one state-dependent part, and one or 2 state-action dependent,
# parts depending on whether raw_outputs is True or False.

tf_false = tf.constant(False, dtype=tf.bool) #if this is in-line below, it messes up my syntax highlighting

class base_architecture:
    def __init__(
                self,
                name,
                output_shape,
                settings,
                full_network=False,
                training=tf_false,
                advantage_activation_fcn=None,
                kbd_activation=None,
                raw_outputs=False,
                ):
        self.name = name
        self.settings = settings
        self.full_network = full_network
        self.training_tf = training
        self.output_shape = self.n_rotations, self.n_translations, self.n_pieces = output_shape
        self.output_size = self.n_rotations * self.n_translations * self.n_pieces
        self.output_activation = settings["nn_output_activation"]
        self.advantage_activation_fcn = advantage_activation_fcn
        self.advantage_range = self.settings["advantage_range"]
        self.n_used_pieces, self.used_pieces_mask_tf = self.create_used_pieces_mask()
        self.kbd_activation = kbd_activation
        self.raw_outputs = raw_outputs
        self.scope = tf.variable_scope(self.name, reuse=tf.AUTO_REUSE)
        self.initialize_variables()
    def initialize_variables(self):
        pass
    def output_streams(self, *args, **kwargs):
        raise Exception("base_architecture is an ABC. please instantiate a derived class!")
    def __call__(self,vectors, visuals, *args, **kwargs):
        with self.scope as scope:
            outputs = self.output_streams(vectors, visuals, *args, **kwargs) #outputs = f(s,a), g(s)
            self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            if self.raw_outputs:
                return outputs
            else:
                return self.Q_V_A(*outputs)
    def Q_V_A(self, _V, _A):
        Q, V, A = N.qva_from_raw_streams(
            _V,
            self.advantage_range * _A,
            mask=self.used_pieces_mask_tf,
            n_used_pieces=self.n_used_pieces,
            separate_piece_values=self.settings["separate_piece_values"],
            mode=self.settings["advantage_type"]
        )
        return Q, V, A
    def create_used_pieces_mask(self):
        assert self.n_pieces in [1,7]
        if self.n_pieces == 1:
            return 1, 1.0
        used_pieces = [0 for _ in range(self.n_pieces)]
        for i in range(self.n_pieces):
            if i in self.settings["pieces"]:
                used_pieces[i] = 1
        return len(self.settings["pieces"]), tf.constant(np.array(used_pieces).reshape((1,1,1,7)).astype(np.float32))
