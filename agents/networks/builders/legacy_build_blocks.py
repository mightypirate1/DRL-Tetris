import tensorflow as tf
from agents.networks import network_utils as N

# This file is to provide backwards compatibility mainly.
# Once SVENton is stable and mature, it will be tidied up, but for now I need it like this :(

def create_vectorencoder(x, settings):
    with tf.variable_scope("vectorencoder", reuse=tf.AUTO_REUSE) as vs:
        for n in range(settings['vectorencoder_n_hidden']):
            x = tf.layers.dense(
                                x,
                                settings['vectorencoder_hidden_size'],
                                name='vectorencoder_layer{}'.format(n),
                                activation=tf.nn.elu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.contrib.layers.xavier_initializer(),
                               )
        x = tf.layers.dense(
                            x,
                            settings['vectorencoder_output_size'],
                            name='layer{}'.format(settings['vectorencoder_n_hidden']+1),
                            activation=self.output_activation,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.contrib.layers.xavier_initializer(),
                           )
    return x

def create_visualencoder(x, settings):
    with tf.variable_scope("visualencoder", reuse=tf.AUTO_REUSE) as vs:
        if settings["pad_visuals"]:
            x = N.apply_visual_pad(x)
        for n in range(settings['visualencoder_n_convs']):
            y = tf.layers.conv2d(
                                    x,
                                    settings["visualencoder_n_filters"][n],
                                    settings["visualencoder_filter_sizes"][n],
                                    name='visualencoder_layer{}'.format(n),
                                    padding='same',
                                    activation=tf.nn.elu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    bias_initializer=tf.zeros_initializer(),
                                )
            if "visualencoder_dropout" in settings:
                x = tf.keras.layers.SpatialDropout2D(settings["visualencoder_dropout"])(x,self.training_tf)
            if n in settings["visualencoder_peepholes"] and settings["peephole_convs"]:
                x = N.peephole_join(x,y,mode=settings["peephole_join_style"])
            else:
                x = y
            if n in settings["visualencoder_poolings"]:
                x = tf.layers.max_pooling2d(x, (2,1), (2,1), padding='same')
    return x

def create_kbd_visual(x, settings):
    x = tf.layers.max_pooling2d(x, 2, 2, padding='valid')
    for i in range(settings["kbd_vis_n_convs"]):
        x = N.layer_pool(
                        x,
                        n_filters=settings["kbd_vis_n_filters"][i],
                        filter_size=(3,3),
                        dropout=(settings["visualencoder_dropout"],self.training_tf)
                        )
    x = tf.reduce_mean(x, axis=[1,2])
    return x

def create_kbd(x, settings):
    for i in range(settings["keyboard_n_convs"]-1):
        x = tf.layers.conv2d(
                                x,
                                settings["keyboard_n_filters"][i],
                                (5,5),
                                name='keyboard_conv{}'.format(i),
                                padding='same',
                                activation=tf.nn.elu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                bias_initializer=tf.zeros_initializer(),
                            )
        if "visualencoder_dropout" in settings:
            x = tf.keras.layers.Dropout(settings["visualencoder_dropout"])(x,self.training_tf)
    x = blocks.keyboard_conv(x, self.n_rotations, self.n_pieces, name='keyboard_conv{}'.format(settings["keyboard_n_convs"]-1))
    return x
    # #Interpret with of field as translations for the piece W ~> T, then:
    # # [?, 1, T, R*P] -> [?, T, R, P] -> [?, R, T, P]
    # x = tf.reshape(x, [-1, 10, 4, 7])
    # x = tf.transpose(x, perm=[0,2,1,3])
    # return x

def create_value_head(x, settings):
    for n in range(settings['valuenet_n_hidden']):
        x = tf.layers.dense(
                            x,
                            settings['valuenet_hidden_size'],
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
                        activation=settings["nn_output_activation"],
                        bias_initializer=tf.zeros_initializer(),
                       )
    if not settings["separate_piece_values"]:
        return v
    v_p = tf.layers.dense(
                          x,
                          7,
                          name='piece_values',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          activation=None,
                          bias_initializer=tf.zeros_initializer(),
                         )
    v_p = 0.5 * N.advantage_activation_sqrt(v_p - tf.reduce_mean(v_p, axis=1, keepdims=True))
    return v + v_p
