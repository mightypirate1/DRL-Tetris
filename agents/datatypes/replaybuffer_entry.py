class replaybuffer_entry: #Wrapper class for entries, so that sort algorithms can get handy comparisons
    def __init__(self, data, prio, time_stamp):
        self.data = data
        self.prio = prio
        self.time_stamp = time_stamp
        self.is_weight = None
    def set_prio(self, p):
        self.prio = p
    def set_isw(self, w):
        self.is_weight = w
    def set_time(self,t):
        self.time_stamp = t
    def __getvalue__(self):
        return self.data
    def __lt__(self, e):
        if type(e) is replaybuffer_entry:
            e = e.prio
        return e > self.prio
    def __le__(self, e):
        if type(e) is replaybuffer_entry:
            e = e.prio
        return e >= self.prio
    def __gt__(self, e):
        if type(e) is replaybuffer_entry:
            e = e.prio
        return self.prio > e
    def __ge__(self, e):
        if type(e) is replaybuffer_entry:
            e = e.prio
        return self.prio >= e
    def __eq__(self, e):
        if type(e) is replaybuffer_entry:
            e = e.prio
        return self.prio == e
    def __str__(self):
        return "replaybuffer_entry(prio:{}, time:{}, isw:{}, data:{})".format(self.prio, self.time_stamp, self.is_weight, self.data)
