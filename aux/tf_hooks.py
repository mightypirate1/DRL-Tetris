import tensorflow as tf
import numpy as np

class quick_summary:
    def __init__(self, settings=None, session=None, init_time=0):
        self.settings = settings
        self.session = session
        self.init_time = init_time
        self.summary_writer = tf.summary.FileWriter(
                                                    "summaries/"+self.settings["run-id"],
                                                    self.session.graph,
                                                    )
    def update(self, stats_dict, time=0):
        t = time-self.init_time
        summary = tf.Summary()
        for x in stats_dict:
            summary.value.add(tag=x, simple_value=np.mean(stats_dict[x]))
        self.summary_writer.add_summary(summary, t)
        self.summary_writer.flush
