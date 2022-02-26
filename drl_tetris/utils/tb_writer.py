import tensorflow.compat.v1 as tf
import numpy as np

class tb_writer:
    def __init__(self, name, session, suffix=""):
        self.summary_writer = tf.summary.FileWriter(
            "summaries/"+name,
            session.graph,
        )
    def update(self, stats_dict, time=0):
        summary = tf.Summary()
        for x in stats_dict:
            summary.value.add(tag=x, simple_value=np.mean(stats_dict[x]))
        self.summary_writer.add_summary(summary, time)
        self.summary_writer.flush()
