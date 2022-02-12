import tensorflow.compat.v1 as tf

class adv_normalizer:
    def __init__(self, lr, clip_val=2.0, safety=2.0):
        self.a_mean = tf.Variable(1.0, trainable=False)
        self.a_max  = tf.Variable(1.0, trainable=False)
        self.lr = lr
        self.clip_val = clip_val
        self.update_op = tf.no_op()
        self.safety = safety
    def __call__(self, A):
        self.input  = A
        batch_mean = tf.reduce_mean(tf.abs(A))
        batch_max  = tf.reduce_max(tf.abs(A))
        norm = tf.maximum(self.a_mean, batch_mean)
        clip = tf.minimum(self.safety * self.a_max / self.a_mean, self.clip_val)
        # clip = tf.minimum(self.a_max / self.a_mean, self.safety) #I-Z07 used this line, with safety=3.0
        a_normalized = A / ( norm + 10**-6)
        self.output = tf.clip_by_value( a_normalized, -clip, clip )
        self.update_op = [self.a_mean.assign( (1-self.lr) * self.a_mean + self.lr * batch_mean ), self.a_max.assign( (1-self.lr) * self.a_max + self.lr * batch_max )]
        self.a_saturation = tf.reduce_mean(tf.cast(tf.not_equal(a_normalized, self.output),tf.float32))
        return self.output
