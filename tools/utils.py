import os
import pickle
import numpy as np
import itertools
import tensorflow as tf
from scipy.stats import rankdata
import experiments.presets
from tools.settings import default_settings
from collections import Collection, Mapping

def parse_arg(_entry_idx, data, fill_up=None, indices=False):
    #This tries to encapsulate the general pattern of passing data vectorized...
    #TODO: Write a more thorough comment on the two ways this method can be used.

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
    if "presets" in settings:
        preset_keys = settings["presets"]
        s = {"presets" : preset_keys}
        for key in preset_keys:
            s.update(experiments.presets.presets[key])
    else: #This is the old way
        s = default_settings.copy()
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
    n = x.size
    p_unnormalized = rankdata(n-x, method='ordinal')**-temperature
    return p_unnormalized / p_unnormalized.sum()

def entropy(x):
    return -np.sum(x * np.log(x))

def replace_nan_with_value(tensor, value):
    return tf.where(
            tf.is_nan(tensor),
            value*tf.ones_like(tensor),
            tensor
            )

def progress_bar(current, total, length=30, start="[", stop="]", done="|", remaining="-"):
    progress = current / total
    done_ticks = round(progress * length)
    remaining_ticks = length - done_ticks
    return start + done * done_ticks + remaining * remaining_ticks + stop

def recursive_map(data, func):
    apply = lambda x: recursive_map(x, func)
    if isinstance(data, Mapping):
        return type(data)({k: apply(v) for k, v in data.items()})
    elif isinstance(data, Collection):
        return type(data)(apply(v) for v in data)
    else:
        return func(data)
#The following 2 are to provide a quick way of evaluating parameter-types! evaluate_params(param_dict, t) = params_evaluated_at_t_dict
def param_evaluater(t):
    def evaluater(p):
        if type(p) in [int, float]: #this is intended to be all non param-type objects
            return p
        return p(t)
    return evaluater
def evaluate_params(params, t):
    return recursive_map(params, param_evaluater(t))
