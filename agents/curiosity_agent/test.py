import numpy as np
import tensorflow as tf
from curiosity_network import curiosity_network
# from agents.curiosity_agent.curiosity_network import curiosity_network

size = (220,)
def generate_data(n, hi, lo):
    ret = []
    if n < 1:
        return None
    for i in range(n):
        k = np.random.randint(low=lo, high=hi)
        x = np.zeros((1,)+size)
        idx = np.random.permutation(np.arange(size[0], dtype=np.int))[:k].reshape(-1)
        x[0,idx] = 1
        ret.append(x)
    return np.concatenate(ret, axis=0)

d1 = generate_data(1000,30,0)
d2 = generate_data(1000,100,30)
d3 = generate_data(1000,200,100)
d  = generate_data(1000,220,0)

m1,m2,m3 = (300,100,0) #"mix"
breakpoints = [100*x for x in range(1,100)]
with tf.Session() as sess:
    ape = curiosity_network(size, sess)
    for t in range(breakpoints[-1]):
        print("iteration {}!".format(t))
        #Test evaluation
        eval1 = ape.evaluate(d1)
        eval2 = ape.evaluate(d2)
        eval3 = ape.evaluate(d3)
        #See some samples
        a1 = generate_data(m1,30,0)
        a2 = generate_data(m2,100,30)
        a3 = generate_data(m3,200,100)
        cat = [x for x in (a1,a2,a3) if x is not None]
        a = np.concatenate(cat, axis=0)
        perm = np.random.permutation(np.arange(m1+m2+m3))
        ape.train(a[perm,:])
        #Sometimes print and ask for a new sample-mixture
        if t in breakpoints:
            print(m1, eval1.mean(), eval1.var())
            print(m2, eval2.mean(), eval2.var())
            print(m3, eval3.mean(), eval3.var())
            z = input("mix??? ")
            if z != "" : m1,m2,m3 = [int(x) for x in z.split(",")]
