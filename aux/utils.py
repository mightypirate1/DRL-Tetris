from aux.settings import default_settings
import numpy as np

def parse_arg(entry_idx, data, fill_up=None, indices=False):
    #This tries to encapsulate the general pattern of passing data vectorized...
    #TODO: Write a more thorough comment on the two ways this method can be usedself.
    #TODO: Move this function to some utility plays (maybe) if e.g. vector_agent needs it too...

    #Depending on what the indexing was: int, [int_0, ... , int_k] or None (None is alias for ALL), we get the data vector and possibly indices too
    if entry_idx is None:
        ret = data
        entry_idx = [idx for idx in range(len(data))]
    elif type(entry_idx) in [list, np.ndarray]:
        ret = [data[i] for i in entry_idx]
    else:
        if fill_up is None:
            entry_idx = [entry_idx]
        else:
            entry_idx = [entry_idx for _ in range(fill_up)]
        ret = [data[i] for i in entry_idx]
    if indices:
        #Output is indices and data as lists...
        return entry_idx, ret
    #Output is a list of data...
    return ret

#We pass default settings around everywhere to ensure all hyper-parameters are uniformly customizable
def parse_settings(settings):
    s = default_settings.copy()
    for x in settings:
        s[x] = settings[x]
    return s
