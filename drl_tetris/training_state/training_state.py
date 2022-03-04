import redis
import pickle
import logging
from pathlib import Path

from drl_tetris.training_state.redis_types import clock, byte_block, queue, dictionary, entry, flag, cache
from drl_tetris.utils.scope import keyjoin

logger = logging.getLogger(__name__)


class training_state:
    def __init__(self, me=None, scope='', trainer="trainer", dummy=False):
        init_weights = None  # this means do not overwrite any stored value for init_weights
        if me is None:
            init_weights = -1  # so forces re-load of weights
            me = get_next_worker_id(scope=scope)

        ### Name & scope:
        self.me = me
        self.my_scope      = keyjoin(scope, me)
        self.trainer_scope = keyjoin(scope, trainer)
        self.shared_scope  = keyjoin(scope, "shared")
        self.cache = cache

        ### Data:
        self.workers_clock         =       clock("workers-clock", scope=self.shared_scope)
        self.runner_state          =  byte_block("runner-state", scope=self.my_scope)
        self.trainer_weights       =  byte_block("latest-weights-data", scope=self.trainer_scope)
        self.trainer_weights_index =       clock("latest-weights-index", scope=self.trainer_scope, replacement=0)
        self.weights               =  byte_block("latest-weights-data", scope=self.my_scope)
        self.weights_index         =       clock("latest-weights-index", scope=self.my_scope, init=init_weights)
        self.data_queue            =       queue("data-queue", scope=self.trainer_scope)
        self.trainer_clock         =       clock("clock", scope=self.trainer_scope)
        self.stats                 =  dictionary("stats", scope=self.my_scope, replacement=0.0, as_type=float, update_op="increment")
        self.stats_str             =  dictionary("stats-str", scope=self.my_scope, update_op="increment")
        self.alive_flag            =        flag("alive", scope=self.my_scope)
        self.validation_artifact   =  byte_block("validation-artifact", scope=self.my_scope)
        self.validation_checksum   =       entry("validation-checksum", scope=self.my_scope)



### Cheese queue (place to go if you are a new worker to get a number that no one uses)
def get_next_worker_id(scope=''):
    for id in range(1000):
        worker_id = f"worker-{id}"
        candidate_scope = keyjoin(scope, worker_id)
        candidate_flag = flag("alive", scope=candidate_scope)
        if (success := candidate_flag.claim()):
            logger.info(f'scope [{scope}] claimed')
            return worker_id
    raise IOError(f"creating too many workers!")
