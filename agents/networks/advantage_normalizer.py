import tensorflow as tf

class adv_normalizer:
    def __init__(self, lr, clip_val=2.0):
        self.a_norm = tf.Variable(1.0, trainable=False)
        self.lr = lr
        self.clip_val = clip_val
        self.update_op = tf.no_op()
    def __call__(self, A):
        self.input  = A
        self.output = tf.clip_by_value( A / (self.a_norm + 10**-6), -self.clip_val, self.clip_val )
        self.update_op = self.a_norm.assign((1-self.lr)*self.a_norm + self.lr*tf.reduce_mean(tf.abs(A)))
        return self.output
