import redis
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

cache = redis.Redis(host='redis', port=6379)


def serialize(x):
    return pickle.dumps(x)


def deseriealize(x):
    return pickle.loads(x)

#######
### Root-scope vars:
#####
WORKERS_CLOCK     = "workers-clock"
WORKERS_ID_TICKER = "workers-id-ticker"

class training_state:
    def __init__(self, me=None, trainer="trainer", dummy=False):
        self.me = me or f"worker-{get_next_worker_id()}"
        self.trainer = trainer
        ## If we have no name, we are assigned a worker number
        if not dummy:
            cache.set(self.ALIVE_FLAG, 1)
    #######
    ### Sub-scope keys:
    #####
    @property
    def TRAINER_WEIGHTS_DATA(self):
        return f"{self.trainer}/latest-weights-data"
    @property
    def TRAINER_WEIGHTS_INDEX(self):
        return f"{self.trainer}/latest-weights-index"
    @property
    def WEIGHTS_INDEX(self):
        return f"{self.me}/latest-weights-index"
    @property
    def TRAINER_DATA_QUEUE(self):
        return f"{self.trainer}/data-queue"
    @property
    def TRAINER_CLOCK(self):
        return f"{self.trainer}/clock"
    @property
    def SEND_COUNT(self):
        return f"{self.me}/send-count"
    @property
    def RECIEVE_COUNT(self):
        return f"{self.me}/recieve-count"
    @property
    def STATS_ROOT(self):
        return f"{self.me}/stats/"
    @property
    def STATS_KEYS(self):
        return f"{self.me}/stats-keys"
    @property
    def ALIVE_FLAG(self):
        return f"{self.me}/alive"

    #######
    ### Clocks
    #####
    def get_worker_clock(self):
        return self.get_clock(WORKERS_CLOCK)

    def get_trainer_clock(self):
        return self.get_clock(self.TRAINER_CLOCK)

    def get_clock(self, clock):
        if (clock_bytes := cache.get(clock)):
            return int(clock_bytes.decode())
        return 0

    def tick_worker_clock(self, tics):
        self.tick_clock(WORKERS_CLOCK, tics)

    def tick_trainer_clock(self, tics):
        self.tick_clock(self.TRAINER_CLOCK, tics)

    def tick_clock(self, clock, tics):
        cache.incrby(clock, amount=tics)

    #######
    ### Data bus: workers -> trainer
    #####
    def push_worker_data(self, data):
        dbg = cache.rpush(self.TRAINER_DATA_QUEUE, *map(serialize, data))
        self.log_send(len(data))

    def pop_worker_data(self):
        raw_data = cache.lpop(self.TRAINER_DATA_QUEUE)
        found_data = raw_data is not None
        if found_data:
            data_bytes = raw_data
            data = deseriealize(data_bytes)
            self.log_recieve(1)
            return found_data, data
        return False, None

    def pop_all_worker_data_it(self):
        while (fd := self.pop_worker_data())[0]:
            yield fd[1]

    #######
    ### Weights: trainer -> workers
    #####
    def publish_weights(self, weights):
        cache.set(self.TRAINER_WEIGHTS_DATA, serialize(weights))
        cache.incr(self.TRAINER_WEIGHTS_INDEX)

    def get_weights(self, suffix=None, newer_than=None):
        idx_key = self.TRAINER_WEIGHTS_INDEX
        weight_key = self.TRAINER_WEIGHTS_DATA
        if suffix:
            idx_key += "/" + suffix
            weight_key += "/" + suffix
        if (current_index_bytes := cache.get(idx_key)) is None:
            return False, -1, None
        current_index = int(current_index_bytes.decode())
        if newer_than and current_index <= newer_than:
            return False, current_index, None
        weights = deseriealize(cache.get(weight_key))
        self.log_recieve(1)
        return True, current_index, weights

    def get_current_weight_index(self):
        curr_bytes = cache.get(self.TRAINER_WEIGHTS_INDEX)
        curr = -1 if curr_bytes is None else int(curr_bytes.decode())
        return curr

    #######
    ### Logging and stats
    #####
    def log_send(self, amount):
        cache.incrby(self.SEND_COUNT, amount=amount)

    def log_recieve(self, amount):
        cache.incrby(self.RECIEVE_COUNT, amount=amount)

    def increment_stats(self, stats_dict):
        stats_keys_in = self.get_stats_keys()
        new_keys = _recurse(stats_dict, self.STATS_ROOT, "increment")
        stats_keys_out = list(set(stats_keys_in+new_keys))
        self.set_stats_keys(stats_keys_out)

    def set_stats(self, stats_dict):
        stats_keys_in = self.get_stats_keys()
        new_keys = _recurse(stats_dict, self.STATS_ROOT, "set")
        stats_keys_out = list(set(stats_keys_in+new_keys))
        self.set_stats_keys(stats_keys_out)

    def fetch_stats(self):
        return {key: cache.get(key).decode() for key in self.get_stats_keys()}

    def _stats_key(self, key):
        return keyjoin(self.STATS_ROOT, key)

    def get_stat_by_key(self, key, replacement=None):
        as_type = None if replacement is None else type(replacement)
        return safe_get(self._stats_key(key), replacement=replacement, as_type=as_type)

    def set_stat_by_key(self, key, value):
        return cache.set(self._stats_key(key), value)

    def get_stats_keys(self):
        keys_bytes = cache.get(self.STATS_KEYS)
        if keys_bytes is None:
            return []
        stats_keys = deseriealize(keys_bytes)
        return stats_keys

    def set_stats_keys(self, stats_keys):
        cache.set(self.STATS_KEYS, serialize(stats_keys))


#######
### Recursive function for stats-operations
#####
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
            _recurse(v, key, operation)
        else:
            try:
                main_op(key, v)
            except redis.exceptions.ResponseError as e:
                fallback_op(key, v)
            finally:
                key_list.append(key)
    return key_list

#######
### Short-hand for path-like joins
#####

def safe_get(key, as_type=None, replacement=None):
    bytes = cache.get(key)
    if bytes is None:
        return replacement
    raw = bytes.decode()
    if as_type is None:
        return raw
    return as_type(raw)

def keyjoin(x,y):
    return str(Path(x)/Path(y))

### Worker id-ticker
def get_next_worker_id():
    return cache.incr(WORKERS_ID_TICKER)

def is_agent_root(maybe_agent_root):
    test_key = keyjoin(maybe_agent_root, 'alive')
    return cache.exists(test_key)
