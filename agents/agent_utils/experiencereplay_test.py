import numpy as np
import tensorflow as tf
#
from keras.datasets import mnist
from keras.utils import np_utils
#
from curiosity_network import curiosity_network

''''''
np.set_printoptions(linewidth=212, precision=2)
size = (28**2,)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
X_train = X_train / 255
X_test = X_test / 255
num_classes = 10
data = {}
siev = np.random.permutation(np.arange(X_train.shape[1]))[:220]
for n in range(num_classes):
    idx = np.where(y_train == n)
    data[n] = X_train[idx[0][siev]]
def get_data(mix):
    ret = []
    for n in range(num_classes):
        idx = np.random.permutation( np.arange(data[n].shape[0]) )[:mix[n]]
        ret.append(data[n][idx,:])
    return np.concatenate(ret,axis=0)
''''''


train_mix = [ max(0,100-20*x)  for x in range(10) ]
test_mix1 = [ max(0,-100+20*x) for x in range(10) ]
test_mix2 = [ max(0,8)         for x in range(10) ]

breakpoints = [100*x for x in range(1,100)]
with tf.Session() as sess:
    ape = curiosity_network("curiosity_test", size, sess)
    #
    for t in range(breakpoints[-1]):
        d_train = get_data(train_mix)
        d_test1 = get_data(test_mix1)
        d_test2 = get_data(test_mix2)
        print("5to9", ape.evaluate(d_train).mean(), ape.evaluate(d_train).max(), ape.evaluate(d_train).min())
        print("0to4", ape.evaluate(d_test1).mean(), ape.evaluate(d_test1).max(), ape.evaluate(d_test1).min())
        print("even", ape.evaluate(d_test2).mean(), ape.evaluate(d_test2).max(), ape.evaluate(d_test2).min())
        print("---")
        for i in range(5):
            perm = np.random.permutation(np.arange( sum(train_mix) ))
            ape.train(d_train[perm,:])
        #Sometimes print and ask for a new sample-mixture
        if t in breakpoints:
            input()
