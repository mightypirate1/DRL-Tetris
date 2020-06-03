import tensorflow as tf
import numpy as np

class value_estimator:
    def __init__(
                self,
                vec_sizes,
                vis_sizes,
                network,
                rewards,
                dones,
                k,
                gamma,
                _lambda,
                filter=None,
                truncate_aggregation=False,
                time_stamps=None,
                time_stamp_gamma=1.0,
                separate_piece_values=True,
                ):
        self._vec_sizes, self._vis_sizes = vec_sizes, vis_sizes
        self._network = network
        self._gamma = gamma
        self._lambda = _lambda
        self._r = rewards
        self._d = dones
        self._steps = self._create_steps(k, filter)
        self._n_steps = len(self._steps)
        self._truncate_aggregation = truncate_aggregation
        self._time_stamps = time_stamps
        self._time_stamp_gamma = time_stamp_gamma
        self._separate_piece_values = separate_piece_values
        self._vec_inputs = [tf.placeholder(tf.float32, (None, self._n_steps)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self._vec_sizes)]
        self._vis_inputs = [tf.placeholder(tf.float32, (None, self._n_steps)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self._vis_sizes)]
        self._value_estimator_tf = self._create_estimator(
                                                            self._vec_inputs,
                                                            self._vis_inputs,
                                                            rewards,
                                                            dones,
                                                        )

    def feed_dict(self, vec, vis):
        vec_dict = dict(zip(self._vec_inputs, map(self._filter_inputs,vec)))
        vis_dict = dict(zip(self._vis_inputs, map(self._filter_inputs,vis)))
        return { **vec_dict, **vis_dict}

    def _filter_inputs(self, inputs):
        # [None, k+1, *] to [None, len(filtered_k), *]
        return inputs[:,self._steps,:]

    def _create_estimator(self, vec, vis, rewards, dones):
        _d = tf.minimum(1, tf.cumsum(dones, axis=1)) #Minimum ensures there is no stray done from an adjacent trajectory influencing us...
        done_time_tf = tf.reduce_sum( 1-_d, axis=1)
        v_step = [None for _ in range(self._n_steps)]
        idx_dict = dict()
        for idx, t in enumerate(self._steps):
            print("k-step:",idx,t)
            idx_dict[t] = idx # indexes time-steps to location in placeholder
            s_t_vec = [vec_state[:,idx,:] for vec_state in self._vec_inputs]
            s_t_vis = [vis_state[:,idx,:] for vis_state in self._vis_inputs]
            net_output = self._network(s_t_vec, s_t_vis)
            _v_t = net_output[0] if  len(net_output) == 2 else net_output[1] # f(s), g(s,a) or Q,V,A is assumed.
            if self._separate_piece_values:
                _v_t = tf.reduce_sum(_v_t * self._network.used_pieces_mask_tf, axis=3, keepdims=True) / self._network.n_used_pieces
            v_step[idx] = tf.reshape(_v_t, [-1, 1])
        assert self._time_stamps is None, "not yet implemented!"
        # extra_decay = 1.0 ** self.time_stamps_tf
        gamma = self._gamma # * extra_decay
        def k_step_estimate(k):
            e = 0
            for t in range(k):
                e += rewards[:,t,:] * tf.cast((done_time_tf >= t),tf.float32) * (gamma**t)
            v_k = v_step[idx_dict[k]]
            e += v_k * tf.cast((done_time_tf >= k),tf.float32) * (gamma**k)
            # print("K",k,":",e)
            return e
        estimators = [(k,k_step_estimate(k)) for k in self._steps]
        return self._aggregate(estimators, done_time_tf)

    def _aggregate(self,estimators, done_times):
        estimator_sum_tf, weight = 0, 0
        for k,e in estimators:
            lambda_filter = tf.cast((done_times >= k-1),tf.float32) if self._truncate_aggregation else 1.0
            _lambda = self._lambda * lambda_filter
            estimator_sum_tf += e * _lambda**k
            weight += _lambda**k
        value_estimator_tf = estimator_sum_tf / weight
        return value_estimator_tf

    def _create_steps(self,k, filter):
        #0) Figure out what estimates we need:
        estimator_steps = [i for i in range(1,k+1)]
        if filter is not None:
            if len(filter) > 0:
                filter = np.array(filter).reshape((1,-1))
                steps = np.array(estimator_steps).reshape((-1,1))
                filter = np.prod((steps % filter),axis=1)!=0
                estimator_steps = steps[np.where(filter)].ravel().tolist()
        return [int(s) for s in estimator_steps]

    @property
    def output_tf(self):
        return tf.stop_gradient(self._value_estimator_tf)
