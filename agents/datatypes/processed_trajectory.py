from agents.datatypes import replaybuffer_entry, trajectory

class processed_trajectory(replaybuffer_entry.replaybuffer_entry):
    def __init__(self, s,r,d,p,tv,prio):
        self.s, self.r, self.d, self.p, self.target_value, self.prio, self.max_len, self.length = s,r,d,p,tv,prio,len(r),len(r)
    def update_value(self, model):
        self.target_value, self.prio = trajectory.trajectory.process_trajectory(self, model, update=True)
    def get_data(self):
        return self.s[0], self.p[0], self.target_value, self.is_weight
    def __len__(self):
        return len(self.r)
