import redis
import pickle
import logging
from pathlib import Path

from drl_tetris.training_state.redis_types import clock, byte_block, queue, dictionary, entry, flag, cache
from drl_tetris.utils.scope import keyjoin

logger = logging.getLogger(__name__)

trainer_scope = "trainer"
def worker_scope(id):
    return f"worker-{id}"

### Cheese queue
def get_next_worker_id():
    for id in range(1000):
        scope = worker_scope(id)
        candidate_flag = flag("alive", scope=scope)
        if (success := candidate_flag.claim()):
            logger.info(f'scope [{scope}] claimed')
            return id
    raise IOError(f"creating too many workers!")

class training_state:
    def __init__(self, me=None, scope='', trainer="trainer", dummy=False):
        init_weights = None  # this means do not overwrite any stored value for init_weights
        if me != "trainer":
            init_weights = -1  # so forces re-load of weights
            me = worker_scope(get_next_worker_id())
        self.me = keyjoin(scope, me)
        self.trainer = trainer
        self.cache = cache
        ### Data:
        self.workers_clock         =       clock("workers-clock")
        self.runner_state          =  byte_block("runner-state", scope=me)
        self.trainer_weights       =  byte_block("latest-weights-data", scope=trainer_scope)
        self.trainer_weights_index =       clock("latest-weights-index", scope=trainer_scope, replacement=0)
        self.weights               =  byte_block("latest-weights-data", scope=me)
        self.weights_index         =       clock("latest-weights-index", scope=me, init=init_weights)
        self.data_queue            =       queue("data-queue", scope=trainer_scope)
        self.trainer_clock         =       clock("clock", scope=trainer_scope)
        self.stats                 =  dictionary("stats", scope=me, replacement=0.0, as_type=float, update_op="increment")
        self.stats_str             =  dictionary("stats-str", scope=me, update_op="increment")
        self.alive_flag            =        flag("alive", scope=me)
        self.validation_artifact   =  byte_block("validation-artifact", scope=me)
        self.validation_checksum   =       entry("validation-checksum", scope=me)
