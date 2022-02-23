from time import time
from collections import defaultdict
from functools import partial, wraps
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

### Measure time for calls
# Use timekeeper defined below unless you wanna fine-grain sub process-granularity
class timekeeper_type:
    def __init__(self):
        self.flush()
    def measure_time(self, func, prefix="", debug=False):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if debug: logger.info(func)
            t_start = time()
            self._entered_tag[prefix] = func.__name__
            ret = func(*args, **kwargs)
            time_elapsed = time() - t_start
            if prefix:
                key = str(Path(prefix) / Path(func.__name__))
            else:
                key = func.__name__
            self._time_stats[key] += time_elapsed
            self._last_tag[prefix] = func.__name__
            return ret
        return wrapper
    def flush(self):
        self._time_stats  = defaultdict(float)
        self._last_tag    = defaultdict(str)
        self._entered_tag = defaultdict(str)
    @property
    def stats(self):
        return {
            'time': dict(self._time_stats),
            'latest': dict(self._last_tag),
            'current': dict(self._entered_tag),
            }
    @property
    def last_tag(self):
        return self.last_tag
    def __call__(self, *prefixes, debug=False):
        prefix = '/'.join(prefixes)
        return partial(self.measure_time, prefix=prefix, debug=False)
timekeeper = timekeeper_type()
