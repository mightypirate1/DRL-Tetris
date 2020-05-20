import tensorflow as tf
from agents.networks import network_utils as N

# Game plan: write good code and put it here. Migrate the mess in q_net_base into here and network utils; but tidy please...

# This is a resudual-block a-la ResNet etc. Silver used it. I trust it.
def residual_block(x,
                    n_layers=4,
                    n_filters=128,
                    filter_size=(3,3),
                    strides=(1,1),
                    peepholes=True,
                    pools=False,
                    dropout=0.0,
                    training=tf.constant(False),
                    ):
    for _ in range(n_layers):
        y = x
        x = tf.layers.conv2d(
                              x,
                              n_filters,
                              filter_size,
                              padding='same',
                              activation=tf.nn.elu,
                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              bias_initializer=tf.zeros_initializer(),
                            )
        x = N.peephole_join(x,y, mode="add")
        if dropout > 0:
            x = tf.keras.layers.SpatialDropout2D(dropout)(x,training)
        if pools:
            x = tf.layers.max_pooling2d(x, (2,2), (2,2), padding='same')
    return x

# This is an idea to preserve game-geometry when producing an action-space.
def keyboard_conv(x, n_rot, n_p, name='keyboard_conv'):
    x = tf.layers.conv2d(
                            x,
                            n_rot * n_p,
                            (x.shape.as_list()[1],3),
                            name=name,
                            padding='valid',
                            activation=N.advantage_activation_sqrt,
                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            bias_initializer=tf.zeros_initializer(),
                        )
    X = [ x[:,:,:,p*n_p:(p+1)*n_p ] for p in range(n_rot) ]
    x = tf.concat(X, axis=1)
    return x
