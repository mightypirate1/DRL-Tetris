import numpy as np

class reward:
    def __init__(self, *args, **kwargs):
        if len(args) > 2:
            raise ValueError("reward needs to have 0, 1 or 2 arguments: [<extrinsic>] [<intrinsic>]")
        elif len(args) > 0:
            self._extrinsic = args[0]
            if 'extrinsic' in kwargs:
                raise ValueError("Doubly specified extrinsic :(")
        elif len(args) == 2:
            self._intrinsic = args[1]
            if 'intrinsic' in kwargs:
                raise ValueError("Doubly specified intrinsic :(")
        else:
            self._extrinsic  = kwargs.pop('extrinsic')
        self._ext_weigts = kwargs.pop('ext_weight', np.zeros_like(self._extrinsic).put([0],[1.0]))
        self.ext_rule    = kwargs.pop('ext_rule'  , self._default_rule)
        if len(args) < 2:
            self._intrinsic  = kwargs.pop('intrinsic' , np.zeros((1,)))
        self._int_weigts = kwargs.pop('int_weight', np.zeros_like(self._intrinsic)               )
        self.int_rule    = kwargs.pop('int_rule'  , self._default_rule)
    def _default_rule(self,other):
        pass
    @property
    def reward(self): #For those unaware we do both :)
        return self.extrinsic
    @property
    def intrinsic(self):
        return self._int_weigts * self._intrinsic
    @property
    def extrinsic(self):
        return self._ext_weigts * self._extrinsic
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "reward<R="+str(self())+"=("+str(self.r)+", w="+str(self.w)+")>"
    def __call__(self):
        return self.extrinsic if self._intrinsic is None else self.extrinsic, self.intrinsic
    def __add__(self,other):
        r = [a+b for a,b in zip(self.r, other.r)]
        return reward(r, extra_rewards=self.extra, weights=self.w)
