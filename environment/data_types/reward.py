import numpy as np

###
# This looks complicated, but I have some ideas I want to test, and I need this flexibility to keep code clean elsewhere!
#

class reward:
    def __init__(self, *args, **kwargs):
        if len(args) > 2:
            raise ValueError("reward needs to have 0, 1 or 2 arguments: [<extrinsic>] [<intrinsic>]")
        elif len(args) > 0:
            self._extrinsic = np.array(args[0])
            if 'extrinsic' in kwargs:
                raise ValueError("Doubly specified extrinsic :(")
        elif len(args) == 2:
            self._intrinsic = np.array(args[1])
            if 'intrinsic' in kwargs:
                raise ValueError("Doubly specified intrinsic :(")
        else:
            self._extrinsic  = np.array(kwargs.pop('extrinsic'))
        if len(args) < 2:
            self._intrinsic  = np.array(kwargs.pop('intrinsic' , np.zeros((1,))))
    def ext_rule(self,*args, **kwargs):
        raise ValueError("Dont use base-class!")
    def int_rule(self,*args, **kwargs):
        raise ValueError("Dont use base-class!")
    @property
    def reward(self): #For those unaware we do both :)
        return self.extrinsic
    @property
    def intrinsic(self):
        return self._intrinsic
    @property
    def extrinsic(self):
        return self._extrinsic
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "reward<R="+str(self())+"=("+str(self.extrinsic.tolist())+", " + str(self.intrinsic.tolist()) + " )>"
    def __call__(self, separate_components=False):
        return self._extrinsic.sum()+self._intrinsic.sum() if not separate_components else (self._extrinsic, self._extrinsic)
    def __add__(self,other):
        return type(self)(
                            extrinsic=self.ext_rule(self._extrinsic, other._extrinsic, add=True),
                            intrinsic=self.int_rule(self._intrinsic, other._intrinsic, add=True),
                          )
    def __sub__(self,other):
        return type(self)(
                            extrinsic=self.ext_rule(self._extrinsic, other._extrinsic, sub=True),
                            intrinsic=self.int_rule(self._intrinsic, other._intrinsic, sub=True),
                          )

#This is just a vector of numbers
class standard_reward(reward):
    def ext_rule(self, x, y, add=False, sub=False):
        if add: return x+y
        if sub: return x-y
    def int_rule(self, x, y, add=False, sub=False):
        if add: return x+y
        if sub: return x-y

class maingoal_reward(standard_reward):
    def ext_rule(self,x,y, add=False, sub=False):
        tmp = np.zeros_like(y)
        tmp[0] = y[0]
        if add: return x+tmp
        if sub: return x-tmp

class coopintrinsic_reward(maingoal_reward):
    def int_rule(self, x,y, **kwargs):
        return x+y
