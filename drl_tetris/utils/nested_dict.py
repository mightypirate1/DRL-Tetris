from drl_tetris.utils.scope import keyjoin

class PathDict:
    #######
    ### Core functionality
    #####

    def set_source(self, newsource):
        self._source = newsource

    def __init__(self, scope='', source=None):
        self.set_source(source)
        if source is None:
            self._source = dict()
        self._scope = scope

    def __getitem__(self, key):
        key = self.unscope(key)
        if key in self._source:
            return self._source[key]
        return PathDict(source=self._source, scope=key)

    def __setitem__(self, key, value):
        key = self.unscope(key)
        if key in self._source:
            self._source[key] = value
        PathDict(source=self._source, scope=key).assign(value)

    def assign(self, assignobj):
        if type(assignobj) in [PathDict, dict]:
            for key, val in assignobj.items():
                self._source[self.unscope(key)] = val
        else:
            self._source[self._scope] = assignobj
        return self

    def add(self, addobject):
        if type(addobject) in [PathDict, dict]:
            for key, val in addobject.items():
                if self.unscope(key) in self._source:
                    self._source[self.unscope(key)] += val
                else:
                    self._source[self.unscope(key)] = val
        else:
            if self.unscope(key) in self._source:
                self._source[self.unscope(key)] += addobject
            else:
                self._source[self.unscope(key)] = addobject
        return self

    def copy(self):
        return PathDict(source=dict(self.items()))

    def map(self, function):
        for k,v in self.items():
            self[k] = function(v)
        return self


    #######
    ### Utils for joining strings so that we can pretend to do 'cd <key>' and 'cd ..'
    #####

    def unscope(self, key):
        return keyjoin(self._scope, key)
    def scope(self, key):
        if self._scope == '':
            return key
        if not key.startswith(self._scope):
            raise KeyError(key)
        return key[len(self._scope)+1:]
    def isinscope(self, key):
        return key.startswith(self._scope)
    def keys(self):
        return [self.scope(key) for key in self._source.keys() if self.isinscope(key)]
    def values(self):
        return [self._source[key] for key in self._source.keys()]
    def items(self):
        return list(zip(self.keys(), self.values()))
    def __repr__(self):
        return f'ND[{self._scope}]:[{dict(zip(self.keys(), self.values()))}]'



x = PathDict()
x.set_source ({
    'ape/x' : 1,
    'ape/y' : 2,
    'ape/z' : 3,
    'bacon/sub/x' : 4,
    }
)
y = x['ape']
print(y)
print(y['x'])

y['x'] = 7

print(x)
