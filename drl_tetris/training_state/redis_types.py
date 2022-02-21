from abc import ABC, abstractmethod
import pickle
import logging
from pathlib import Path

import redis

cache = redis.Redis(host='redis', port=6379)
logger = logging.getLogger(__name__)

#######
### These are "datatypes":
# entry      - generic data, probably a string, int, float or so
# clock      - anything you would want to count or increment
# byte_block - anyt pickelable object
# queue      - just a queue
# claim_flag - used to thread-safe memory recovery
# flag       - like a boolean but as int
# dictionary - pretend dict
#####

CLAIM_TIME = 20 # Number of seconds a flag is claimed for. (All threads need to have started in this ammount of time)

### Path-like joins for keys
def keyjoin(x,y):
    return str(Path(x)/Path(y))

class redis_obj(ABC):
    def __init__(self, key, scope=None, as_type=None, replacement=None):
        self._key = key if scope is None else keyjoin(scope, key)
        self.scope = scope
        self.as_type = as_type
        self.replacement = replacement

    @abstractmethod
    def get(self):
        pass
    @abstractmethod
    def set(self, value):
        pass
    def encode(self, x):
        return x
    def decode(self, x):
        return x

class entry(redis_obj):
    def encode(self, x):
        if self.as_type is not None:
            x = self.as_type(x)
        return x
    def decode(self, x, replacement=None):
        replacement = replacement or self.replacement
        if (found := x is not None):
            if type(x) is bytes:
                x = x.decode()
            if self.as_type is not None:
                x = self.as_type(x)
        else: # Found no data
            if replacement is not None:
                return False, replacement
        return found, x
    def get(self, replacement=None):
        replacement = replacement or self.replacement
        found, value = self.decode(cache.get(self._key), replacement=replacement)
        if replacement and not found:
            value = replacement
        return found, value
    def set(self, value):
        return cache.set(self._key, self.encode(value))

class byte_block(entry):
    def encode(self, x):
        return pickle.dumps(x)
    def decode(self, x, replacement=None):
        replacement = replacement or self.replacement
        if (found := x is not None):
            x = pickle.loads(x)
        if not found and replacement:
                return False, x
        return found, x

class queue(byte_block):
    def __init__(self, key, **kwargs):
        super().__init__(key, **kwargs)
        scope = kwargs.get("scope")
        self.incount  = clock(f"{key}-incount",  scope=scope)
        self.outcount = clock(f"{key}-outcount", scope=scope)
    def set(self):
        raise IOError(f"set called on queue: {self._key}")
    def get(self):
        raise IOError(f"get called on queue: {self._key}")
    def pop(self):
        found, data = self.decode(cache.lpop(self._key))
        if found:
            self.outcount.tick(1)
        return found, data
    def push(self, data):
        if type(data) not in (list, tuple):
            data = [data]
        ret = cache.rpush(self._key, *map(self.encode, data))
        self.incount.tick(len(data))
    def pop_iter(self):
        while (found_data := self.pop())[0]: # found, data = found_data
            yield found_data[1]

class clock(entry):
    def __init__(self, key, replacement=0, as_type=int, **kwargs):
        super().__init__(key, replacement=replacement, as_type=as_type, **kwargs)
    def tick(self, increment):
        return self.decode(cache.incrby(self._key, amount=int(increment)))[1]
    def get(self):
        return super().get()[1]

class claim_flag(clock):
    def claim(self):
        if (my_ticket := self.tick(1)) > 1:
            return False
        cache.expire(self._key, CLAIM_TIME)
        return True

class flag(entry):
    def __init__(self, key, replacement=0, **kwargs):
        super().__init__(key, replacement=replacement, **kwargs)
    def set(self, expire=None):
        super().set(1)
        if expire:
            cache.expire(self._key, expire)
    def get(self):
        return super().get()[1]

### Helper fcn for dictionary-class
operation_dict = {
    "set": (cache.set, cache.set),
    "increment": (cache.incrbyfloat, cache.set),
    "debug": (print, print),
}
def _recurse(curr_dict, prefix, operation):
    key_list = []
    main_op, fallback_op = operation_dict[operation]
    for k, v in curr_dict.items():
        assert type(k) is str, "stats_dicts' keys must be strings"
        key = keyjoin(prefix, k)
        if type(v) is dict:
            keys = _recurse(v, key, operation)
            key_list += keys
        else:
            try:
                main_op(key, v)
            except redis.exceptions.ResponseError as e:
                fallback_op(key, v)
            finally:
               key_list.append(key)
    return key_list

class dictionary(redis_obj):
    def __init__(self, key, update_op="set", **kwargs):
        super().__init__(key, **kwargs)
        keys_key = f"{key}-keys"
        self.scope = kwargs.get("scope")
        self.update_op = update_op
        self.keys = byte_block(keys_key, scope=self.scope)
    def get_keys(self, *args, **kwargs):
        found, keys = self.keys.get(*args, **kwargs)
        if not found: return []
        return keys
    def get(self, key, replacement=None):
        val = entry(key, scope=self._key, as_type=self.as_type, replacement=replacement)
        return val.get(replacement=replacement)[1]
    def set(self, key, val):
        return entry(key, scope=self._key, as_type=self.as_type).set(val)
    def get_all(self):
        keys = self.get_keys(replacement=[])
        return {key: entry(key).get()[1] for key in keys}
    def update(self, update_dict, update_op=None):
        update_op = update_op or self.update_op
        keys_in = self.get_keys(replacement=[])
        new_keys = _recurse(update_dict, self._key, update_op)
        keys_out = list(set(keys_in+new_keys))
        self.keys.set(keys_out)
