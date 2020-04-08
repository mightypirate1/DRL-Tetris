from environment.data_types.action import action, null_action

class action_list(list):
    def __init__(self, container=None, remove_null=False):
        self.remove_null = remove_null
        if container is None or len(container) == 0:
            container = [null_action]
        if null_action not in container:
            self.container = [null_action]
        else:
            self.container = []
        for x in container:
            assert type(x) in [action, list], "attempted to create action list with non-list type actions (x={} type(x)={}). An actions list is of the form [s_1,...,s_n] where each s_i is a list of actions [a_1,...,a_m], and each a_j an integer.".format(x,type(x))
            if x not in self.container:
                self.container.append(action(x))
        if self.remove_null:
            self.remove_nulls()
    def remove_nulls(self):
        while len(self.container) > 1 and null_action in self.container:
            self.container.remove(null_action)
    def __add__(self,n):
        return action_list(n.container+self.container, remove_null=(n.remove_null or self.remove_null))
    def __str__(self):
        return "action_list("+str(self.container)+")"
    def __repr__(self):
        return self.__str__()
    def __getitem__(self, key):
        if key > len(self.container):
            print("I HOPE THAT ACTION WAS NOT NECESSARY......")
            print("actions:", self.container)
            print("requested idx:", key)
            return null_action
        return self.container[key]
    def __len__(self):
        return len(self.container)
    def __iter__(self):
        return self.container.__iter__()
