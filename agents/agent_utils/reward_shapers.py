import numpy as np

### ## ## # # #
### THIS WHOLE FILE EXPECTS INPUTS r TO BE LIST-LIKE; E.G [0,0,0,0,1]
### ## ## # # #

#This smears out the reward from the final step onto the previous steps
def linear_reshaping(ammount, single_policy=True):
    def r_tilde(r):
        if len(r) < 3:
            return r
        T = len(r)-1
        rT = r[-1]
        idxs = np.arange(T+1)
        signs = np.power(-1,idxs+T) if single_policy else 1.0
        coeff = 2 * ammount * rT / (T*T-T)
        ret = coeff * idxs * signs
        ret[-1] = (1-ammount) * rT
        ret[:-1] += r[:-1]
        return ret
    return r_tilde

def exp_reshaping(gamma, single_policy=True):
    if not single_policy: gamma = -gamma
    def r_tilde(r):
        ret = np.zeros_like(r, dtype=np.float)
        weight = 0
        sum = 0
        for i in reversed(range(len(r))):
            sum *= gamma
            weight *= gamma
            sum += r[i]
            weight += 1
            ret[i] = sum
            ret = ret / weight
            ret[:-1] += r[:-1]
        return ret
    return r_tilde

if __name__ == "__main__":
    r1 = [0,0,0,0,1]
    r2 = [0,0,0,0,0,1]
    hat = linear_reshaping(0.5)
    r1_hat = hat(r1)
    r2_hat = hat(r2)
    print(r1_hat)
    print(r2_hat)
