import numpy as np
from numpy.lib.stride_tricks import as_strided

def k_step_view(x, k):
    #reshapes shape (N, d1,d2,..,di) -> (n,k,d1,d2,...,di) with full data-sharing.
    #... basically (s0,s1,...,sn) -> ((s0,..,sk),(s1,...,s(k+1)),...(s(n-k),...,sn))
    shape = (x.shape[0]-k, k, *x.shape[1:])
    strides = x.strides[0], *x.strides
    return as_strided(x, shape=shape, strides=strides)


if __name__ == "__main__":
    data = np.zeros((100,2,2,), dtype=np.int64)
    w = k_step_view(data, 2)
    # exit()
    for i in range(100):
        data[i,:,:] = i
    for i in range(98):
        print("d",data[i,:])
        print("w",w[i,:])
        print("---")
    print("====")
    print(data[-3,:])
    print("====")
    print(w[-1,:])
