import tensorflow.compat.v1 as tf
import time
import os
import logging

from drl_tetris.training_state import training_state
from drl_tetris.utils.timekeeper import timekeeper
from drl_tetris.utils.math import mix
from drl_tetris.utils.logging import logstamp
from drl_tetris.utils.tb_writer import tb_writer
from drl_tetris.runner import runner

import threads
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
        self.run_name           = self.settings["run-id"]

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
        self.transfer_weights()
        self.training_state.cache.save()

    # def validation_artifact(self):
    #     return 0

    # def runner_validation(self, artifact):
    #     return artifact

    def run(self):
        with tf.Session(config=self.config) as session:
            with tb_writer(self.run_name, session) as self.tb_writer:
                # Initialize!
                self.trainer_agent.create_models(session)
                # Restore weights
                saved_weights_exists, saved_weights = self.training_state.weights.get()
                if saved_weights_exists:
                    self.trainer_agent.import_weights(saved_weights)

                with self.tb_writer:
                    ### Run!
                    try:
                        while not self.received_interrupt:
                            self.load_worker_data()
                            if (stats := self.do_training())[0]:
                                self.transfer_weights()
                                self.save_weights()
                                self.update_stats(stats)
                    except Exception as e:
                        self.save_weights(
                            idx=f"CRASH_t={self.training_state.trainer_clock.get()}"
                        )
                        raise e

    @timekeeper()
    def load_worker_data(self):
        data_from_workers = [*self.training_state.data_queue.pop_iter()]
        if self.trainer_agent.receive_data(data_from_workers)[0]:
            self.update_performance_stats(data_from_workers)

    @timekeeper()
    @logstamp(logger.info, on_entry=True, on_exit=True)
    def do_training(self):
        if self.settings["single_policy"]:
            stats = self.trainer_agent.do_training()
        else:
            # Implement this
            raise NotImplementedError('implement')
            n, stats0 = self.trainer_agent.do_training(policy=0)
            m, stats1 = self.trainer_agent.do_training(policy=1)

            # n += m
        # stats
        self.training_state.stats.update(
            {'runner': {'n_samples_trained': stats[0]}},
        )
        return stats

    @timekeeper()
    def transfer_weights(self):
        weights = self.trainer_agent.export_weights()
        self.training_state.weights.set(weights)
        self.training_state.weights_index.tick(1)

    @timekeeper()
    def save_weights(self, idx=None):
        if not idx:
            idx = "LATEST"
            if (curr := self.training_state.trainer_weights_index.get()) % 250 == 0:
                idx = curr
        self.trainer_agent.save_weights(
            *utils.weight_location(
                self.settings,
                idx=idx,
            )
        )

    def update_performance_stats(self, trajectories):
        loaded_samples = self.training_state.stats.get('runner/n_samples_loaded', replacement=0)
        avg_trajectory_length = self.training_state.stats.get("trajectory_length", replacement=35.)
        for md, trajectory in trajectories:
            length =  md["length"]
            loaded_samples += length
            trajectory_length = mix(float(avg_trajectory_length), length, alpha=0.05)
            self.tb_writer.update({'trainer/trajectory_length': length}, time=loaded_samples)
        self.training_state.stats.set("trajectory_length", trajectory_length)
        loaded_samples = self.training_state.stats.set('runner/n_samples_loaded', loaded_samples)
        self.training_state.alive_flag.set(expire=120)

    def update_stats(self, trainingresults): # and print :-)
        timestats = timekeeper.stats
        self.training_state.stats.update(timestats['time'], prefix='time')
        if (current_weights := self.training_state.trainer_weights_index.get()) > self.latest_printed_weights:
            _, trainingstats = trainingresults
            self.tb_writer.update(trainingstats, time=current_weights)
            statsdict = self.training_state.stats.get_all()
            timekeys = [(k.split('/')[-1],k) for k in statsdict.keys() if 'time' in k]
            currtimes = [timestats['time'].get(k1,0.) for k1,_ in timekeys]
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
