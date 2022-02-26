import tensorflow.compat.v1 as tf
import numpy as np

class tb_writer(tf.summary.FileWriter):
    def __init__(self, name, session, *args, **kwargs):
        super().__init__(
            "data/summaries/"+name,
            session=session,
            graph=session.graph,
            flush_secs=60,
        )
    def update(self, stats_dict, time=0):
        summary = tf.Summary()
        for x in stats_dict:
            summary.value.add(tag=x, simple_value=np.mean(stats_dict[x]))
        self.add_summary(summary, time)
        # self.flush()
