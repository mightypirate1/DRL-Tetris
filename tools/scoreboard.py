from itertools import combinations, repeat

def div_str(x,y):
    return "{}".format(0.0 if y == 0 else x/y)
def frac_str(x,y):
    return "{} / {}".format(x,y)

class scoreboard:
    def __init__(self, ids, width=200000000):
        self._x = dict( zip(combinations(sorted(ids), 2), repeat(0) )) #match-up score diff
        self._n = dict( zip(combinations(sorted(ids), 2), repeat(0) )) #match-up count
        self._stats = dict(zip(ids, repeat(0)))     #individual scores
        self._stats_tot = dict(zip(ids, repeat(0))) #individual n_games
        self._ids = ids
        self._width = width
        max_len = max([len(str(id)) for id in ids])
        self._strlen = min( max(max_len,12) ,width//(len(ids)+1))
        self._strlen_first = min(20, max_len)
        self._current = (None,None)
    def set_current_players(self, *args):
        self._current = tuple(sorted(self._p_from_args(*args)))
    def declare_winner(self, winner):
        assert winner in self._current, "winner must be currenty playing. call set_current_players first"
        self._n[self._current] += 1
        self._stats_tot[self._current[0]] += 1
        self._stats_tot[self._current[1]] += 1
        self._stats[winner] += 1
        if winner == self._current[0]:
            self._x[self._current] += 1
    def individual_score(self, x, frac=True):
        if frac:
            return frac_str(self._stats[x], self._stats_tot[x])
        return  div_str(self._stats[x], self._stats_tot[x])
    def score(self, *args, frac=True):
        a,b = self._p_from_args(args, distinct=False)
        if a == b:
            return "-"
        elif a > b:
            n = self._n[(b,a)]
            x = n-self._x[(b,a)]
        else:
            n = self._n[(a,b)]
            x = self._x[(a,b)]
        return frac_str(x,n) if frac else div_str(x,n)
    def score_table(self, frac=True):
        #Helper fcn for formatting
        def adjust(x, right=False, cutoff=None):
            if type(x) is not str: x = str(x)
            if cutoff is None: cutoff = self._strlen
            x = x[:cutoff-3] + (x[cutoff-3:], '...')[len(x) > cutoff]
            if right:
                return str(x).rjust(cutoff+1, " ")
            return str(x).ljust(cutoff+1, " ")
        #Actual table
        s, s_intro = "", "  "+adjust("vs: ", right=True, cutoff=self._strlen_first)
        for a in self._ids:
            s += adjust(a,right=True, cutoff=self._strlen_first)+": "
            s_intro += adjust(a)
            for b in self._ids:
                s += adjust(self.score(a,b, frac=frac))
            s += adjust(self.individual_score(a, frac=frac))+"\n"
        table = s_intro+" Total \n"+s
        return table
    def _p_from_args(self, *args, distinct=True):
        assert len(args) in [1,2], "give me 2 players out of the ones specified: " + str(self_ids)
        a,b = args[0] if len(args) == 1 else args
        assert a in self._ids and b in self._ids
        assert not distinct or a != b, "give me 2 DIFFERENT players"
        return a,b

if __name__ == "__main__":
    a,b,c = "ape", "bacon", "charlie"
    s = scoreboard([a,b,c], name_len=5)
    print(s.score_table())
    s.set_current_players(a,b)
    s.declare_winner(a)
    s.declare_winner(a)
    s.declare_winner(a)
    s.declare_winner(b)
    s.declare_winner(b)
    s.set_current_players(c,b)
    s.declare_winner(b)
    s.declare_winner(b)
    s.declare_winner(b)
    s.declare_winner(c)
    s.declare_winner(b)
    print(s.score_table())
