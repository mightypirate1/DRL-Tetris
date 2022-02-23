import tensorflow.compat.v1 as tf
import time
import os
import logging

from drl_tetris.training_state import training_state
from drl_tetris.utils.timekeeper import timekeeper
from drl_tetris.utils.math import mix
from drl_tetris.utils.logging import logstamp
from drl_tetris.runner import runner

import threads
from tools.tf_hooks import quick_summary
import tools.utils as utils

logger = logging.getLogger(__name__)

class trainer(runner):
    def __init__(self, settings):
        super().__init__(settings, me="trainer")
        self.config = tf.ConfigProto(
            log_device_placement=False,
            device_count={'GPU': 1}
        )
        self.config.gpu_options.allow_growth = True
        self.latest_printed_weights = -1

    def read_settings(self):
        self.trainer_agent_type = self.settings["trainer_type"]
        self.env_type           = self.settings["env_type"]

    def set_runner_state(self, state):
        [self.trainer_agent] = state

    def get_runner_state(self):
        return [self.trainer_agent]

    def create_runner_state(self):
        self.trainer_agent = self.trainer_agent_type(
            id=self.training_state.me,
            settings=self.settings,
            sandbox=self.env_type(settings=self.settings),
            mode=threads.TRAINER,
        )

    @logstamp(logger.info, on_exit=True)
    def graceful_exit(self):
        self.training_state.cache.save()

    def run(self):
        with tf.Session(config=self.config) as session:
            #Initialize!
            self.quick_summary = quick_summary(settings=self.settings, session=session)
            self.trainer_agent.create_models(session)
            # TODO: tidy here
            self.trainer_agent.quick_summary = self.quick_summary
            self.trainer_agent.clock = 0

            saved_weights_exists, saved_weights = self.training_state.weights.get()
            if saved_weights_exists:
                self.trainer_agent.import_weights(saved_weights)

            ### Run!
            try:
                while True:
                    self.load_worker_data()
                    if self.do_training():
                        self.transfer_weights()
                        self.save_weights()
                    self.update_stats()
            except Exception as e:
                self.save_weights(
                    idx=f"CRASH_t={self.training_state.trainer_clock.get()}"
                )
                raise e

    @timekeeper()
    def load_worker_data(self):
        data_from_workers = [*self.training_state.data_queue.pop_iter()]
        n_samples, _ = self.trainer_agent.receive_data(data_from_workers)
        # stats
        self.training_state.stats.update({'runner': {'n_samples_loaded': n_samples}})
        self.update_performance_stats(data_from_workers)

    @timekeeper()
    @logstamp(logger.info, on_entry=True, on_exit=True)
    def do_training(self):
        if self.settings["single_policy"]:
            n = self.trainer_agent.do_training()
        else:
            n  = self.trainer_agent.do_training(policy=0)
            n += self.trainer_agent.do_training(policy=1)
        # stats
        self.training_state.stats.update(
            {'runner': {'n_samples_trained': n}},
        )
        return n

    @timekeeper()
    def transfer_weights(self):
        weights = self.trainer_agent.export_weights()
        self.training_state.weights.set(weights)
        self.training_state.weights_index.tick(1)

    @timekeeper()
    def save_weights(self, idx=None):
        if not idx:
            idx = "LATEST" if not (curr := self.training_state.trainer_weights_index.get()) % 250 else curr
        self.trainer_agent.save_weights(
            *utils.weight_location(
                self.settings,
                idx=idx,
            )
        )

    def update_performance_stats(self, trajectories):
        trajectory_length = self.training_state.stats.get("trajectory_length", replacement=35.)
        for md, trajectory in trajectories:
            trajectory_length = mix(float(trajectory_length), md["length"], alpha=0.05)
        self.training_state.stats.set("trajectory_length", trajectory_length)
        self.training_state.alive_flag.set(expire=120)

    def update_stats(self): # and print :-)
        timestats = timekeeper.stats
        self.training_state.stats.update(timestats['time'], prefix='time')
        if (current_weights := self.training_state.trainer_weights_index.get()) > self.latest_printed_weights:
            statsdict = self.training_state.stats.get_all()
            timekeys = [(k.split('/')[-1],k) for k in statsdict.keys() if 'time' in k]
            currtimes = [timestats['time'][k1] for k1,_ in timekeys]
            tottimes  = [statsdict[k2] for _, k2 in timekeys]
            timestats_totals = [(k1,tt, ct) for (k1,_), tt, ct in zip(timekeys, tottimes, currtimes)]
            k_len = max([len(k) for k,*_ in timestats_totals])
            tformat = lambda t: str(round(float(t),4)).ljust(15) # TODO: This conversion to float should not be needed if the "dict" worked as expected
            time_strs = '\n'.join([f" - {k.ljust(k_len)}: {tformat(tt)} ({tformat(ct)})" for k,tt,ct in timestats_totals])
            message = \
f'''
--------------------------------
trainer metrics:
 - weights: {current_weights}
 - trajectory_len: {self.training_state.stats.get('trajectory_length')}
time:
{time_strs}
--------------------------------
'''
            logger.info(message)
            self.latest_printed_weights = current_weights
        timekeeper.flush()
