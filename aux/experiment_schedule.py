class experiment_schedule:
    def __init__(self, experiments, total_steps=None, only_last=False):
        assert total_steps is not None
        self.total_steps = total_steps
        self._scheduled_settings = []
        #Check that all files are found as expected
        for experiment in experiments:
            settings, patches = self.load_file(experiment)
            patches = [{}] + patches #Add a null-patch so we always loop over the vanilla-setting when we loop over patches
            s = settings.copy()
            for p in patches:
                s.update(p)
                if not only_last:
                    self._scheduled_settings.append(s.copy())
            if only_last:
                self._scheduled_settings.append(s.copy())
    def load_file(self, file):
        with open(file,'r') as f:
            raw_code = f.read()
        #This is so ugly I will forever deny writing it. I could have done an import, but then I get no tab-completion without complicating things elsewhere....
        raw_code += "\nret[\"settings\"] = settings\nret[\"patches\"] = [] if \"patches\" not in dir() else patches"
        code, ret = compile(raw_code, file, 'exec'), {}
        exec(code,{},{'ret':ret, 'total_steps' : self.total_steps}) #We expect this to give us settings and possible patches
        return ret["settings"], ret["patches"]
    def __getitem__(self, idx):
        return self._scheduled_settings[idx]
    def __iter__(self):
        self.idx = -1
        return self
    def __next__(self):
        self.idx += 1
        if self.idx < len(self._scheduled_settings):
            return self._scheduled_settings[self.idx].copy()
        raise StopIteration
