class action_list(list):
    def __init__(self, container=None):
        if container is None:
            container = [[0]]
        if [0] not in container:
            self.container = [[0]]
        else:
            self.container = []
        for x in container:
            assert type(x) is list, "attempted to create action list with non-list type actions (type={}). An actions list is of the form [s_1,...,s_n] where each s_i is a list of actions [a_1,...,a_m], and each a_j an integer.".format(type(x))
            self.container.append(x)
    def __add__(self,n):
        for x in n.container:
            if x not in self.container:
                self.container.append(x)
        return self
    def __str__(self):
        return str(self.container)
    def __repr__(self):
        return str(self.container)
    def __getitem__(self, key):
        return self.container[key]
    def __len__(self):
        return len(self.container)
    def __iter__(self):
        return self.container.__iter__()
