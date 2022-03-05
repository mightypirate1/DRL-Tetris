import numpy as np
import tensorflow.compat.v1 as tf

###
### Builder utilities!
###

def normalize_advantages(A, activation=None, inplace=False, mode="mean", axis=[1,2], piece_axis=3, separate_piece_values=True, piece_mask=1.0, n_used_pieces=7):
    # En garde! - This is a duelling network.
    if inplace:
        if A.shape.as_list()[-1] == 1: #Corner-case handling
            return (A if activation is None else activation(A))
        V = tf.expand_dims(A[:,:,:,0],3)
        A = A[:,:,:,1:]
    #piece_mask is expected to be shape [1,1,1,7] or constant. 1 for used pieces, 0 for unused.
    if piece_mask == 1.0:
        n_used_pieces = 1
    if  mode == "max":
        all_axis = [i+1 for i in range(A.shape.rank-1)]
        a_maxmasked = piece_mask * A + (1-piece_mask) * tf.reduce_min(A, axis=all_axis, keepdims=True)
        _max_a   = tf.reduce_max(a_maxmasked,  axis=axis,     keepdims=True ) #We max over the actions which WE control (rotation, translation) and average over the ones we dont control (piece)
        if separate_piece_values:
            max_a = _max_a
        else:
            max_a   = tf.reduce_sum(_max_a * piece_mask,  axis=piece_axis,     keepdims=True ) / n_used_pieces #We max over the actions which WE control (rotation, translation) and average over the ones we dont control (piece)
        A  = (A - max_a)  #A_q(s,r,t,p) = advantage of applying rotation r and translation t to piece p in state s; compared to playing according to the argmax-policy
    elif mode == "mean":
        _mean_a = tf.reduce_mean(A,      axis=axis, keepdims=True )
        mean_a  = tf.reduce_sum(_mean_a * piece_mask, axis=piece_axis,     keepdims=True ) / n_used_pieces
        A = (A - mean_a)
    if inplace:
        A = V + A
    if activation is not None:
        A = activation(A)
    return A

def conv_shape_vector(vec, shape_to_match):
    dims = vec.shape.as_list()
    if type(shape_to_match) is not list:
        #Hope it's a tensor-shape!
        shape_to_match = shape_to_match.as_list()
    assert len(dims) == 2, "Give me a vector ([?, K] tensor)"
    x = tf.reshape(vec, [-1, 1, 1, dims[-1]])
    return tf.tile(x, [1, shape_to_match[1], shape_to_match[2], 1])

def normalization_layer(x, mode="layer"):
    assert mode in ["layer",]
    if mode == "layer":
        x = tf.keras.layers.LayerNormalization()(x)
    return x

def peephole_join(*inputs, mode="concat"):
    if mode in ["add", "truncate_add"]:
        assert len(inputs) == 2, "len>2 not implemented"
        x, y = inputs
        nx,ny = x.shape[3], y.shape[3]
        larger = x if nx > ny else y
        smaller = y if nx > ny else x
        x = larger[:,:,:,:smaller.shape[3]] + smaller
        y = larger[:,:,:,smaller.shape[3]:]
        out_list = [x,y] if mode == "add" else [x]
    if mode == "concat":
        out_list = inputs
    return tf.concat(out_list, axis=-1)

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

def visual_stack(x, items=None): #x assumed to be of shape [N,H,W,1]
    x_cumsum = tf.cumsum(x, axis=1)
    x_shadow = tf.minimum(x_cumsum, 1.0)
    _x_height = np.arange(x.shape.as_list()[1], dtype=np.float32).reshape((1,-1,1,1))
    x_height = tf.constant( np.tile(_x_height, (1,1,x.shape.as_list()[2],1)) ) + tf.zeros_like(x)
    x_holes = x_shadow - x
    #
    item_dict = {"cumsum":x_cumsum, "shadow":x_shadow, "height":x_height, "holes":x_holes,}
    if items == True:
        #temporary backwards comp..
        items = ["cumsum", "shadow", "height"]
    if items is None:
        items = item_dict.keys()
    x_stack = [x] + [item_dict[key] for key in items]
    return tf.concat(x_stack, axis=3)

def q_to_v(q, mask=1.0, n_used_pieces=7):
    q_p = tf.reduce_max(q, axis=[1,2], keepdims=True)
    v = tf.reduce_sum(q_p * mask, axis=3, keepdims=True) / n_used_pieces
    return tf.reshape(v, [-1, 1])

def qva_from_raw_streams(_V,_A, mask=1.0, n_used_pieces=7, separate_piece_values=True, mode="mean"):
    A = normalize_advantages(_A, activation=tf.nn.tanh, separate_piece_values=separate_piece_values, mode=mode, piece_mask=mask, n_used_pieces=n_used_pieces)
    Q = _V + A
    V = q_to_v(Q, mask=mask, n_used_pieces=n_used_pieces)
    return Q, V, A

def pool_spatial_dims_until_singleton(x, warning=False):
    for axis in [1, 2]:
        if x.shape[axis] > 1:
            if warning:
                print("applying reduce_mean to ", x, "along axis", axis)
            x = tf.reduce_mean(x, axis=axis, keepdims=True)
    return x

def action_entropy(pi):
    #pi.shape == [?, R, T, P]
    log_pi = tf.math.log( tf.maximum(10**-6, pi) )
    entropy = -tf.reduce_sum(pi * log_pi, axis=[1,2], keepdims=True)
    return entropy

def action_softmax(x, clip=None):
    #x.shape == [?, R, T, P] or [?, W, H, P]
    max_x = tf.reduce_max(x, axis=[1,2], keepdims=True)
    pi_unnormalized = tf.math.exp(x - max_x) #for numerical stability
    pi = pi_unnormalized / tf.reduce_sum(pi_unnormalized, axis=[1,2], keepdims=True)
    return pi

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
                        'kernel_initializer' : tf.keras.initializers.glorot_uniform(),
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
