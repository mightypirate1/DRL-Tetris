import tensorflow as tf

class value_estimator:
    def __init__(
                self,
                network,
                rewards,
                dones,
                k,
                gamma,
                _lambda,
                filter=None,
                time_stamps=None,
                time_stamp_gamma=1.0,
                ):
        self._vec_sizes, self._vis_sizes = network.state_size_vec, network.state_size_vis
        self._network = network
        self._gamma = gamma
        self._lambda = _lambda
        self._r = rewards
        self._d = dones
        self._steps = self.create_steps(k, filter)
        self._n_steps = len(self._steps)
        self._value_estimator_tf = self.create_estimator()
        # self._time_stamps = None
        # self._time_stamp_gamma = 1.0

    def _np_to_smaller_np(self, np):
        # [None, k+1, *] to [None, len(filtered_k), *]
        return np[:,self._steps,:]

    def feed_dict(self, vec, vis):
         vec_dict = dict(zip(self._vec_inputs, self._np_to_smaller_np(vec)))
         vis_dict = dict(zip(self._vis_inputs, self._np_to_smaller_np(vis)))
         return { **vec_dict, **vis_dict}

    def _create_estimator(self):
        dones_tf = tf.minimum(1, tf.cumsum(self._d, axis=1)) #Minimum ensures there is no stray done from an adjacent trajectory influencing us...
        done_time_tf = tf.reduce_sum( 1-dones_tf, axis=1)
        self._vec_inputs = [tf.placeholder(tf.float32, (None, self._n_steps)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self._vec_sizes)]
        self._vis_inputs = [tf.placeholder(tf.float32, (None, self._n_steps)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self._vis_sizes)]
        v_t_tf = [None for _ in range(len(self._n_steps))]
        idx_dict = dict()
        for idx, t in enumerate(self._steps):
            idx_dict[t] = idx
            s_t_vec = [self._vec_inputs[:,t,:] for vec_state in vector_states_training]
            s_t_vis = [self._vis_inputs[:,t,:] for vis_state in visual_states_training]
            _,v,_,self.ref_scope = self._network(s_t_vec, s_t_vis)
            v_t_tf[idx] = v
        #
        # extra_decay = 1.0 ** self.time_stamps_tf
        gamma = self._gamma # * extra_decay
        def k_step_estimate(k):
            e,k = 0, int(k)
            for t in range(k):
                e += rewards[:,t,:]  * tf.cast((done_time_tf >= t),tf.float32) * (gamma**t)
            e += v_t_tf[idx_dict[k]] * tf.cast((done_time_tf >= k),tf.float32) * (gamma**k)
            return e
        estimators = [(k,k_step_estimate(k)) for k in estimator_steps]
        return self._aggregate(estimators)

    def _aggregate(self,estimators):
        weight = 0
        estimator_sum_tf = 0
        for k,e in estimators:
            estimator_sum_tf += e * self._lambda**k
            weight += self._lambda**k
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
        return steps
