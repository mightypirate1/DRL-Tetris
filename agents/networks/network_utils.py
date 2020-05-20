import tensorflow as tf
from tensorflow.python.client import device_lib

###
### Builder utilities!
###

def conv_shape_vector(vec, shape_to_match):
    dims = vec.shape.as_list()
    if type(shape_to_match) is not list:
        #Hope it's a tensor-shape!
        shape_to_match = shape_to_match.as_list()
    assert len(dims) == 2, "Give me a vector ([?, K] tensor)"
    x = tf.reshape(vec, [-1, 1, 1, dims[-1]])
    return tf.tile(x, [1, shape_to_match[1], shape_to_match[2], 1])

def peephole_join(x,y, mode="concat"):
    if mode == "add":
        nx,ny = x.shape[3], y.shape[3]
        larger = x if nx > ny else y
        smaller = y if nx > ny else x
        x = larger[:,:,:,:smaller.shape[3]] + smaller
        y = larger[:,:,:,smaller.shape[3]:]
    return tf.concat([y,x], axis=-1)

def advantage_activation_sqrt(x):
    alpha = 0.01
    ret = tf.sign(x) * (tf.sqrt( tf.abs(x) + alpha**2) - alpha)
    return ret

def apply_visual_pad(x):
    #Apply zero-padding on top:
    x = tf.pad(x, [[0,0],[1,0],[0,0],[0,0]], constant_values=0.0)
    #Apply one-padding left, right and bottom:
    x = tf.pad(x, [[0,0],[0,1],[1,1],[0,0]], constant_values=1.0)
    # This makes floor and walls look like it's a piece, and cieling like its free space
    return x

###
### Regularizers & metrics
###

def argmax_entropy_reg(q, mask=1.0, n_pieces=7.0):
    q_max = tf.reduce_max(q, axis=[1,2], keepdims=True)
    q_argmax_mask = tf.cast(q-q_max>=0, tf.float32)
    #distribution of the argmaxes
    distribution = tf.reduce_sum( q_argmax_mask * mask, axis=3, keepdims=True ) / n_pieces
    p = 10**-8 + tf.reduce_mean(q_argmax_mask, axis=[0,3])
    argmax_entropy = tf.reduce_sum( p * tf.math.log(p) )
    print("argmax_entropy_reg: ERROR CHECK ME!")
    return argmax_entropy

###
### Layer builders
###
conv_default_kwargs = {
                        'padding' : 'same',
                        'activation' : tf.nn.elu,
                        'kernel_initializer' : tf.contrib.layers.xavier_initializer_conv2d(),
                        'bias_initializer' : tf.zeros_initializer(),
                      }
drop_default_kwargs = {}
pool_default_kwargs = {'padding' : 'same'}

def layer_pool(
                X,
                n_filters=16,
                filter_size=(3,3),
                pool_size=2,
                pool_strides=2,
                name=None,
                dropout=None,
                drop_kwargs={},
                conv_kwargs={},
                pool_kwargs={}
                ):
    conv_kwargs = dict(conv_kwargs, **conv_default_kwargs)
    if "name" not in conv_kwargs: conv_kwargs["name"] = name
    drop_kwargs = dict(drop_kwargs, **drop_default_kwargs)
    pool_kwargs = dict(pool_kwargs, **pool_default_kwargs)
    y = tf.layers.conv2d(
                            X,
                            n_filters,
                            filter_size,
                            **conv_kwargs,
                        )
    if dropout is not None:
        #dropout = (rate, training_tensor)
        y = tf.keras.layers.SpatialDropout2D(dropout[0], **drop_kwargs)(y,dropout[1])
    return tf.layers.max_pooling2d(y, pool_size, pool_strides, **pool_kwargs) #comment out for old

###
### Other utilities
###

def check_gpu():
    for dev in device_lib.list_local_devices():
        if "GPU" in dev.name:
            return True
    return False

def debug_prints(vals,tensors):
    for val,tensor in zip(vals,tensors):
        v = val if tensor.shape.rank == 0 else val[0]
        print(tensor,v)
