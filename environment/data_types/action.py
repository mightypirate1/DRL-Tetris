#This class exists currently for doing asserts in other classes. If you think this inefficient or a bad solution: please contact me! //mightypirate1
class action(list):
    def __init__(self, container):
        assert type(container) in [list, action], "action created from non-list type object...: x={}, type(x)={}".format(container,type(container))
        super().__init__(container)
    def __str__(self):
        return "action("+super().__str__()+")"


null_action = action([0])
