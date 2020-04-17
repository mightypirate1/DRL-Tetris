import os
import pickle
import numpy as np
import itertools
from scipy.stats import rankdata

from aux.settings import default_settings

def parse_arg(_entry_idx, data, fill_up=None, indices=False):
    #This tries to encapsulate the general pattern of passing data vectorized...
    #TODO: Write a more thorough comment on the two ways this method can be usedself.

    #Depending on what the indexing was: int, [int_0, ... , int_k] or None (None is alias for ALL), we get the data vector and possibly indices too
    if _entry_idx is None:
        ret = data if type(data) is list else [d for d in data]
        entry_idx = [idx for idx in range(len(data))]
    elif type(_entry_idx) in [list, np.ndarray]:
        ret = [data[i] for i in _entry_idx]
        entry_idx = _entry_idx
    else:
        if fill_up is None:
            entry_idx = [_entry_idx]
        else:
            entry_idx = [_entry_idx for _ in range(fill_up)]
        ret = [data[i] for i in entry_idx]
    if indices:
        #Output is indices and data as lists...
        return entry_idx, ret
    #Output is a list of data...
    return ret

#We pass default settings around everywhere to ensure all hyper-parameters are uniformly customizable
def parse_settings(settings):
    s = default_settings.copy()
    if settings is not None:
        for x in settings:
            s[x] = settings[x]
        #Here we get a chance to add some derived properties
        s["game_area"] = s["game_size"][0] * s["game_size"][1]
    return s

def find_weight_settings(weight_str):
    for name in reversed(weight_str.split("/")):
        path,_,_ = weight_str.rpartition(name)
        if os.path.exists(path+"settings"):
            return path+"settings"
    assert False, "No settings-file found for " + weight_str

def test_setting_compatibility(*settings):
    def test(y,x):
        equals = ["game_size"]
        for field in equals:
            if x[field] != y[field]:
                print(field," is required to be equal for all settings")
                return False
        return True
    for pair in itertools.product(settings,settings):
        if not test(*pair): return False
    return True

def load_settings(file):
    with open(file, 'rb') as f:
        return parse_settings(pickle.load(f))

# This takes a bunch of lists: [a1,a2,...], [b1,..], [c1, ...] and maps onto [[a1,b1,c1,...], [a2,b2,c2,..]...]
def merge_lists(*e):
    return list(zip(*e))

def weight_location(s, idx=""):
    if type(idx) is not str:
        idx = str(idx)
    if type(s) is dict: #s is a settings dictionary
        project = s["run-id"]
        folder = "models/"+project
        file   = folder+"/weights"+idx+".w"
        return folder, file
    if type(s) is str: #s is a string
        #assume s is path to weight file... (.../.../*.w)
        f = s.split("/")[-1]
        folder,_,_ = s.rpartition(f)
        return folder, s

def pareto(x, temperature=1.0):
    p_unnormalized = 1/rankdata(x, method='ordinal')**temperature
    return p_unnormalized / p_unnormalized.sum()

def entropy(x):
    return -np.sum(x * np.log(x))
