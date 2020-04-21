class ape:
    def __init__(self,x, k=None):
        self.x = x
        if k is None:
            self.k = -1 if x%2 == 1 else 0
        else:
            self.k = k
    def __add__(self,other):
        print(self.x, "+"if self.k==1 else "-", other.x, "=", self.x+self.k*other.x)
        return ape(self.x+self.k*other.x, k=self.k)

A = [ape(i) for i in range(0,7,1)]
sum = ape(0,k=-1)
for a in reversed(A):
    sum += a
print(sum.x)
