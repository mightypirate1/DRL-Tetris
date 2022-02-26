import redis
import pickle
import logging
from pathlib import Path

from drl_tetris.training_state.redis_types import clock, byte_block, queue, dictionary, entry, flag, cache

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
    def __init__(self, me=None, trainer="trainer", dummy=False):
        self.me = me or worker_scope(get_next_worker_id())
        self.trainer = trainer
        self.cache = cache
        ### Data:
        self.workers_clock         =       clock("workers-clock")
        self.runner_state          =  byte_block("runner-state", scope=self.me)
        self.trainer_weights       =  byte_block("latest-weights-data", scope=trainer_scope)
        self.trainer_weights_index =       clock("latest-weights-index", scope=trainer_scope, replacement=0)
        self.weights               =  byte_block("latest-weights-data", scope=self.me)
        self.weights_index         =       clock("latest-weights-index", scope=self.me, init=0)
        self.data_queue            =       queue("data-queue", scope=trainer_scope)
        self.trainer_clock         =       clock("clock", scope=trainer_scope)
        self.stats                 =  dictionary("stats", scope=self.me, replacement=0.0, as_type=float, update_op="increment")
        self.stats_str             =  dictionary("stats-str", scope=self.me, update_op="increment")
        self.alive_flag            =        flag("alive", scope=self.me)
        self.validation_artifact   =  byte_block("validation-artifact", scope=self.me)
        self.validation_checksum   =       entry("validation-checksum", scope=self.me)
