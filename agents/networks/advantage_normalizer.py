import tensorflow as tf

class adv_normalizer:
    def __init__(self, lr, clip_val=2.0, safety=2.0):
        self.a_norm = tf.Variable(1.0, trainable=False)
        self.a_max  = tf.Variable(1.0, trainable=False)
        self.lr = lr
        self.clip_val = clip_val
        self.update_op = tf.no_op()
        self.safety = safety
    def __call__(self, A):
        self.input  = A
        batch_mean = tf.reduce_mean(tf.abs(A))
        batch_max  = tf.reduce_max(tf.abs(A))
        norm = tf.maximum(self.a_norm, batch_mean)
        clip = tf.minimum(self.a_max / self.a_norm, self.safety)
        self.output = tf.clip_by_value( A / ( norm + 10**-6), -clip, clip )
        self.update_op = [self.a_norm.assign( (1-self.lr) * self.a_norm + self.lr * batch_mean ), self.a_max.assign( (1-self.lr) * self.a_max + self.lr * batch_max )]
        return self.output
