class reward:
    def __init__(self, rewards, extra_rewards=False, weights=(1.0,)):
        self.r = rewards
        self.w = weights if extra_rewards else (1.0,)
        self.extra = extra_rewards
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return "reward<R="+str(self())+"=("+str(self.r)+", w="+str(self.w)+")>"
    def __call__(self):
        return sum([r*w for r,w in zip(self.r, self.w)])
    def __add__(self,other):
        r = [a+b for a,b in zip(self.r, other.r)]
        return reward(r, extra_rewards=self.extra, weights=self.w)
    @property
    def base(self):
        return self.r[0] * self.w[0]
