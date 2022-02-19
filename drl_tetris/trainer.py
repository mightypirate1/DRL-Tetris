import tensorflow.compat.v1 as tf
import time
import os
import logging

from textwrap import dedent

from drl_tetris.training_state import training_state
from drl_tetris.utils.training_utils import timekeeper
from drl_tetris.utils.math import mix

import threads
from tools.tf_hooks import quick_summary
import tools.utils as utils

logger = logging.getLogger(__name__)

class trainer:
    def __init__(self, settings):
        self.settings = utils.parse_settings(settings)
        self.training_state = training_state(me="trainer")
        self.config = tf.ConfigProto(
            log_device_placement=False,
            device_count={'GPU': 1}
        )
        self.config.gpu_options.allow_growth = True
        self.trainer_agent_type = self.settings["trainer_type"]
        self.latest_printed_weights = -1

    def run(self):
        with tf.Session(config=self.config) as session:
            #Initialize!
            self.quick_summary = quick_summary(
                settings=self.settings, session=session)
            self.trainer_agent = self.trainer_agent_type(
                id=self.training_state.me,
                settings=self.settings,
                session=session,
                sandbox=self.settings["env_type"](settings=self.settings),
                summarizer=self.quick_summary,
                mode=threads.TRAINER,
            )
            saved_weights_exists, _, saved_weights = self.training_state.get_weights()
            if saved_weights_exists:
                self.trainer_agent.import_weights(weights)

            ### Run!
            try:
                while True:
                    self.load_worker_data()
                    if self.do_training():
                        self.transfer_weights()
                        self.save_weights()
                    self.update_stats()
            except Exception as e:
                self.trainer_agent.save_weights(
                    *utils.weight_location(
                        self.settings,
                        idx=f"CRASH_t={self.training_state.get_trainer_clock()}"
                    )
                )
                raise e

    @timekeeper()
    def load_worker_data(self):
        data_from_workers = [*self.training_state.pop_all_worker_data_it()]
        n_samples, _ = self.trainer_agent.receive_data(data_from_workers)
        # stats
        self.training_state.increment_stats({'n_samples_loaded': n_samples})
        self.update_performance_stats(data_from_workers)

    @timekeeper()
    def do_training(self):
        if self.settings["single_policy"]:
            n = self.trainer_agent.do_training()
        else:
            n  = self.trainer_agent.do_training(policy=0)
            n += self.trainer_agent.do_training(policy=1)
        # stats
        self.training_state.increment_stats({'n_samples_trained': n})
        return n

    @timekeeper()
    def transfer_weights(self):
        weights = self.trainer_agent.export_weights()
        self.training_state.publish_weights(weights)

    @timekeeper()
    def save_weights(self):
        self.trainer_agent.save_weights(
            *utils.weight_location(
                self.settings,
                idx=self.training_state.get_current_weight_index()
            ),
            verbose=True,
        )

    def update_performance_stats(self, trajectories):
        trajectory_length = self.training_state.get_stat_by_key("trajectory_length", replacement=0.)
        for trajectory in trajectories:
            trajectory_length = mix(trajectory_length, len(trajectory), alpha=0.01)
        self.training_state.set_stat_by_key("trajectory_length", trajectory_length)

    def update_stats(self):
        timestats = timekeeper.stats
        self.training_state.increment_stats(timestats)
        if (current_weights := self.training_state.get_current_weight_index()) > self.latest_printed_weights:
            logger.info(dedent(
                f'''
                --------------------------------
                trainer metrics:
                    - weights: {current_weights}
                    - trajectory_len: {self.training_state.get_stat_by_key("trajectory_length")}
                time:
                    - training: TO-BE-DONE
                --------------------------------
                '''
            ))
            self.latest_printed_weights = current_weights
            logger.info(f"current_weights: {current_weights}")
        timekeeper.flush()
