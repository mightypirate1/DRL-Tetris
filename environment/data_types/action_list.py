class action_list(list):
    def __init__(self, container=None, remove_null=False):
        self.remove_null = remove_null
        if container is None or len(container) == 0:
            container = [[0]]
        if [0] not in container:
            self.container = [[0]]
        else:
            self.container = []
        for x in container:
            assert type(x) is list, "attempted to create action list with non-list type actions (type={}). An actions list is of the form [s_1,...,s_n] where each s_i is a list of actions [a_1,...,a_m], and each a_j an integer.".format(type(x))
            if x not in self.container:
                self.container.append(x)
        if self.remove_null:
            self.remove_nulls()
    def remove_nulls(self):
        while len(self.container) > 0 and [0] in self.container:
            self.container.remove([0])
    def __add__(self,n):
        return action_list(n.container+self.container, remove_null=(n.remove_null or self.remove_null))
    def __str__(self):
        return "action_list("+str(self.container)+")"
    def __repr__(self):
        return self.__str__()
    def __getitem__(self, key):
        return self.container[key]
    def __len__(self):
        return len(self.container)
    def __iter__(self):
        return self.container.__iter__()
