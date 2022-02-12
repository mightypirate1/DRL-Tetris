import tensorflow.compat.v1 as tf
import numpy as np

class param_noiser:
    def __init__(
                    self,
                    type="additive_gaussian",
                    distance="ratio",
                    target_distance=1.02,
                    scale_to_weight=False,
                    update_frequency=100,
                    lr=0.01,
                ):
        _type_dict = {
                        "multiplicative_gaussian" : self.gauss_mult,
                        "additive_gaussian" : self.gauss_add,
                     }
        _dist_dict = {
                        "ratio" : self.ratio_distance,
                    }
        self.type = type
        self.distance = distance
        self.lr = lr
        self.target_distance = target_distance
        self.scale_to_weight = scale_to_weight
        self.noise_fcn = _type_dict[type]
        self.distance_fcn = _dist_dict[distance]
        self.volume = 0.01
        self._noised_vars = {}
        self.disable_noise_tf = tf.placeholder(tf.bool, shape=[])
        self.update_frequency = update_frequency
        self.time_to_update = 0
        self.activated = False

    def update_noise(self, session, seed=None):
        self.activated = True
        if seed is not None:
            self.set_seed(seed)
        feed_dict = {}
        run_list = []#[self.update_ops]
        for key in self._noised_vars:
            var_dict = self._noised_vars[key]
            new_noise = np.random.normal(loc=0.0, scale=self.volume, size=var_dict["noise"].shape.as_list())
            run_list.append(var_dict["update_noise_op"])
            feed_dict[var_dict["update_noise_placeholder"]] = new_noise
        session.run(run_list, feed_dict=feed_dict)
    def set_seed(self, seed):
        np.random.seed(seed)
    def adjust_vol(self, current_distance):
        gain = 1.03 if current_distance < self.target_distance else 0.96
        self.volume *= gain
        print("DBG: volume", self.volume, "current_distance", current_distance)
    def volume_adjust_callback(self, model_fcn, model_args, model_kwargs, training=True):
        self.time_to_update -= 1
        if self.time_to_update > 0 or not self.activated or not training:
            return
        self.time_to_update = self.update_frequency
        pi1, _, _ = model_fcn(*model_args, **model_kwargs)
        model_kwargs["disable_noise"] = not model_kwargs["disable_noise"]
        pi2, _, _ = model_fcn(*model_args, **model_kwargs)
        self.adjust_vol( self.distance_fcn(pi1,pi2) )
    def add_noise(self, layer):
        layer.kernel = self._add_noise_to_weight(layer.kernel)
        layer.bias   = self._add_noise_to_weight(layer.bias)
    def _add_noise_to_weight(self, tensor):
        shape = tensor.shape.as_list()
        # shape = [1,*tensor.shape.as_list()[1:]]
        pre, suff = tensor.name.split(":")
        noise_name = pre + "-NOISE"
        #The actual noise
        var_noise = tf.Variable(initial_value=np.zeros(shape), name=noise_name,dtype=tf.float32, trainable=False)
        #Keeping track of weight-amplitutes in case we want to scale the noise to it
        avg_amp = tf.Variable(initial_value=tf.ones_like(tensor), name=noise_name+"amplitude",dtype=tf.float32, trainable=False)
        update_op = avg_amp.assign( (1-self.lr) * avg_amp + self.lr * tf.abs(tensor) ) if self.scale_to_weight else tf.no_op()
        #Placeholder for feeding a new value
        noise_feed_placeholder_tf = tf.placeholder(tf.float32, shape=shape)
        #The restult:
        effective_weight = self.noise_fcn(tensor, tf.cond( self.disable_noise_tf, lambda : tf.zeros_like(var_noise), lambda : avg_amp * var_noise ))
        self._noised_vars[var_noise.name] = {
                                                "weight"                   : tensor,
                                                "noise"                    : var_noise,
                                                "avg_amplitude"            : avg_amp,
                                                "update_op"                : update_op,
                                                "update_noise_op"          : var_noise.assign(noise_feed_placeholder_tf),
                                                "update_noise_placeholder" : noise_feed_placeholder_tf,
                                                "effective_weight"         : effective_weight,
                                            }
        # return self.noise_fcn(tensor, avg_amp * (1-tf.cast( self.disable_noise_tf, tf.float32)) * var_noise )
        return effective_weight
    #
    ##
    def gauss_add(self,weight, noise):
        return weight + noise
    def gauss_mult(self,weight, noise):
        return weight * (1+noise)
    #
    ##
    def ratio_distance(self, p,q):
        e = 1e-3
        p,q = p+e,q+e
        return np.mean(np.maximum( p / q, q / p ))
    @property
    def update_ops(self):
        return [ self._noised_vars[var]["update_op"] for var in self._noised_vars ]
    @property
    def disable(self):
        return self.disable_noise_tf
    @property
    def variables(self):
        return [ self._noised_vars[var]["weight"] for var in self._noised_vars ]
