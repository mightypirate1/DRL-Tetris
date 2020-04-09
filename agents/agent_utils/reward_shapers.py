import numpy as np

### ## ## # # #
### THIS WHOLE FILE EXPECTS INPUTS r TO BE LIST-LIKE; E.G [0,0,0,0,1]
### ## ## # # #

#This smears out the reward from the final step onto the previous steps
def linear_reshaping(ammount):
    def r_tilde(r):
        if len(r) < 3:
            return r
        T = len(r)-1
        rT = r[-1]
        coeff = 2 * ammount * rT / (T*T-T)
        ret = coeff * np.arange(T+1)
        ret[-1] = (1-ammount) * rT
        return ret
    return r_tilde

def exp_reshaping(gamma):
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
        return ret / weight
    return r_tilde
