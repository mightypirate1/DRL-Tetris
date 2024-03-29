import math

def param_eval(x,t):
    if issubclass(type(x), parameter):
        return x(t)
    return x

class parameter:
    def __init__(self, value, final_val=None, time_horizon=None, min=0, max=math.inf):
        self.init_val = value
        self.final_val = final_val
        self.time_horizon = time_horizon
        self.max = max
        self.min = min
    def __eval__(self, t):
        return self.get_value(t)
    def __call__(self, t):
        return max( min(self.get_value(t), self._eval(self.max,t)), self._eval(self.min,t) )
    def get_value(self, t):
        return self._eval(self.init_val, t)
    def _eval(self,x,t):
        return param_eval(x,t)
    def __eq__(self, other):
        if isinstance(other, type(self)):
            if self.init_val == other.init_val and self.final_val == other.final_val and self.time_horizon == other.time_horizon:
                return True
        return False
    def __str__(self):
        return "{}(init:{}, final:{}, time_horizon:{}, min/max:{}/{}, {})".format(type(self).__name__,self.init_val,self.final_val,self.time_horizon,self.min,self.max,self.__str2__())
    def __str2__(self):
        return ""
    def __repr__(self):
        return self.__str__()

class exp_parameter(parameter): #Very crude exp-decay :)
    def __init__(self, value, **kwargs):
        self.decay = kwargs.pop('decay', 10**-3)
        self.base  = kwargs.pop('base',  math.e)
        parameter.__init__(self, value, **kwargs)
        self.__init_vars__()
    def __init_vars__(self):
        assert self.time_horizon is None, "exp_parameter uses parameters init base and decay (and min,max): f(t) = init*base**(-decay*t) clipped to range [min, max]"
    def get_value(self, t):
        eval = lambda x : self._eval(x,t)
        init, base, dec = eval(self.init_val), eval(self.base), eval(self.decay)
        return init * base**(-dec*t)
    def __eq__(self, other):
        if isinstance(other, type(self)):
            if self.init_val == other.init_val and self.final_val == other.final_val and self.min == other.min and self.max == other.max and self.decay == other.decay and self.base == other.base:
                return True
        return False
    def __str2__(self):
        return ", min:{}, max:{}, decay:{}, base:{}".format(self.min, self.max, self.decay, self.base)

class linear_parameter(parameter):
    def __init__(self, value, **kwargs):
        parameter.__init__(self, value, **kwargs)
        self.__init_vars__()
    def __init_vars__(self):
        pass
    def get_value(self,t):
        eval = lambda x : self._eval(x,t)
        x = max( min(t,self.time_horizon), 0 ) / self.time_horizon
        return x * eval(self.final_val) + (1-x) * eval(self.init_val)

constant_parameter = parameter
