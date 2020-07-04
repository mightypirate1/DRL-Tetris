import tensorflow as tf
from agents.networks import network_utils as N

#This messes up my syntax-highlighting if it's in-line below...
tf_rn_init = tf.random_normal_initializer(0,0.01)

# This is a resudual-block a-la ResNet etc. Silver used it. I trust it.
def residual_block(x,
                    n_layers=4,
                    n_filters=128,
                    filter_size=(3,3),
                    strides=(1,1),
                    peepholes=True,
                    pools=False,
                    dropout=0.0,
                    pool_size=(3,2),
                    pool_strides=(3,2),
                    output_n_filters=None,
                    output_activation=tf.nn.elu,
                    output_initializer=tf_rn_init,
                    normalization=None,
                    training=False,
                    output_layer=False,
                    param_noiser=None,
                    ):
    for i in range(n_layers):
        #default params
        y, n, activation, join_mode, initializer, normalize, last_layer, second_last_layer = x, n_filters, tf.nn.elu, "add", tf.contrib.layers.xavier_initializer_conv2d(), False, (i==n_layers-1), (i==n_layers-2)
        #last layer sometimes different params
        if last_layer:
            activation = output_activation
            if output_n_filters is not None:
                n = output_n_filters
                join_mode = "truncate_add"
                normalize = True if normalization is not None else False
        if output_layer:
            if last_layer or second_last_layer:
                initializer = output_initializer
            if last_layer:
                normalize = False

        #Build block!
        if last_layer or param_noiser is None:
            x = tf.layers.conv2d(
                                  x,
                                  n,
                                  filter_size,
                                  padding='same',
                                  activation=None,
                                  kernel_initializer=initializer,
                                  bias_initializer=tf.zeros_initializer(),
                                )
        else:
            #TIDY
            conv_layer = tf.keras.layers.Conv2D(n, filter_size, padding='same', activation=None, kernel_initializer=initializer)
            conv_layer.build(x.shape)
            param_noiser.add_noise(conv_layer)
            x = conv_layer(x)
        if peepholes:
            x = N.peephole_join(x,y, mode=join_mode)
        if normalize:
            x = N.normalization_layer(x, mode=normalization)
        if activation is not None:
            x = activation(x)
        if dropout > 0:
            x = tf.keras.layers.SpatialDropout2D(dropout)(x,training)
        if pools:
            _w,_h = x.shape.as_list()[1:3]
            w, h = min(pool_size[0],_w), min(pool_size[1],_h)
            x = tf.layers.average_pooling2d(x, (w,h), (w,h), padding='valid')
    return x


# This is an idea to preserve game-geometry when producing an action-space.
def keyboard_conv(x, n_rot, n_p, name='keyboard_conv', activation=None):
    x = tf.layers.conv2d(
                            x,
                            n_rot * n_p,
                            (x.shape.as_list()[1],3),
                            name=name,
                            padding='valid',
                            activation=None,
                            kernel_initializer=tf.zeros_initializer(),
                            bias_initializer=tf.random_normal_initializer(0,0.00001),
                        )
    X = [ x[:,:,:,p*n_p:(p+1)*n_p ] for p in range(n_rot) ]
    x = tf.concat(X, axis=1)
    if activation is not None:
        x = activation(x)
    return x
