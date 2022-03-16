import tensorflow.compat.v1 as tf

class compressor:
    def __init__(self, lr=0.01, clip_val=3.0, safety=8.0, cautious=True):
        self.x_mean = tf.Variable(1.0, trainable=False)
        self.x_max  = tf.Variable(1.0, trainable=False)
        self.lr = lr
        self.clip_val = clip_val
        self.update_op = tf.no_op()
        self.safety = safety
        self.cautious = cautious
    def __call__(self, X):
        epsilon = 10**-6
        self.input  = X
        # Batch stats
        batch_mean = tf.reduce_mean(tf.abs(X))
        batch_max  = tf.reduce_max(tf.abs(X))

        # Normalization parameters
        floor = tf.maximum(batch_mean, epsilon) if self.cautious else epsilon
        norm = tf.maximum(self.x_mean, floor)
        clip = tf.minimum(self.safety * self.x_max / self.x_mean, self.clip_val)

        # Normalization
        x_normalized = X / norm
        self.output = tf.clip_by_value( x_normalized, -clip, clip )
        self.update_op = [self.x_mean.assign( (1-self.lr) * self.x_mean + self.lr * batch_mean ), self.x_max.assign( (1-self.lr) * self.x_max + self.lr * batch_max )]
        self.x_saturation = tf.reduce_mean(tf.cast(tf.not_equal(x_normalized, self.output),tf.float32))
        return self.output
