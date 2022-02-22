from datetime import datetime
from functools import wraps

def now_str():
    return datetime.now().isoformat(sep='@', timespec='seconds')

class logstamp:
    def __init__(self, loggerfunc, name=None, only_new=True, on_entry=False, on_exit=False):
        self.loggerfunc = loggerfunc
        self.on_entry = on_entry
        self.on_exit = on_exit
        self.only_new = only_new
        self.name = name
        self._last_ret = None
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.on_entry and not self.only_new:
                self.loggerfunc(f"{now_str()} [o] {self.name or func.__name__}")
            ret = func(*args, **kwargs)
            if self.on_exit or self.only_new:
                if not self.only_new or type(ret) == type(self._last_ret) and ret == self._last_ret:
                    self.loggerfunc(f"{now_str()} [x] {self.name or func.__name__}")
            return ret
        return wrapper
