class state:
    def __init__(self, backend_state,state_processor,unlocked=True):
        self.unlocked=unlocked
        self.state_processor=state_processor
        self.backend_state=backend_state.copy()
        self.is_dead = [x.dead[0] for x in backend_state.states]
        if not unlocked:
            self.lock()

    def lock(self):
        for s in self.backend_state.states:
            s.dead[0] = 1 #This prevents the state to be changed by performing actions...
        self.unlocked = False

    def unlock(self):
        for i,s in enumerate(self.backend_state.states):
            s.dead[0] = self.is_dead[i]
        self.unlocked = True

    def __getitem__(self, idx):
        if isinstance(idx,slice):
            start,stop,step = idx.indices(len(self.backend_state.states))
            return [self.state_processor(self.backend_state.states,x) for x in range(start,stop,step)]
        if not hasattr(idx,'__iter__'):
            return self.state_processor(self.backend_state.states,idx)
        else:
            return [self.state_processor(self.backend_state.states,i) for i in idx]

    def __iter__(self):
         self.current = -1
         return self

    def __next__(self):
        self.current += 1
        if self.current == len(self.backend_state.states):
            raise StopIteration
        return self.backend_state.states[self.current]
