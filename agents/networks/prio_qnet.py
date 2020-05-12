
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import aux
import aux.utils as utils

class prio_qnet:
    def __init__(self, agent_id, name, state_size, output_shape, sess, k_step=1, settings=None, worker_only=False):
        assert len(output_shape) == 3, "expected 3D-actions"
        self.settings = utils.parse_settings(settings)
        self.name = name
        self.session = sess
        self.output_activation = settings["nn_output_activation"]
        self.output_shape = self.n_rotations, self.n_translations, self.n_pieces = output_shape
        self.output_size = self.n_rotations * self.n_translations * self.n_pieces
        self.worker_only = worker_only
        self.stats_tensors = []
        self.scope_name = "agent{}_{}".format(agent_id,name)
        self.state_size_vec, self.state_size_vis = state_size
        self.k_step = k_step
        assert k_step > 0, "k_step AKA n_step_value_estimates has to be greater than 0!"

        ###
        ### TIDY THIS UP ONE DAY :)
        #######
        self.keyboard_range = self.settings["keyboard_range"]
        self.n_used_pieces = len(self.settings["pieces"])
        used_pieces = [0, 0, 0, 0, 0, 0, 0]
        for i in range(7):
            if i in self.settings["pieces"]:
                used_pieces[i] = 1
        self.used_pieces_mask_tf = tf.constant(np.array(used_pieces).reshape((1,1,1,7)).astype(np.float32))
        #######
        #####
        ###

        #Define tensors/placeholders
        with tf.variable_scope(self.scope_name):
            if worker_only:
                self.rewards_tf       = None
                self.dones_tf         = None
                self.actions_tf       =  tf.placeholder(tf.uint8, (None, 1))
                self.pieces_tf        =  tf.placeholder(tf.uint8, (None, 1))
                self.vector_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
                self.visual_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
                self.vector_inputs_training = None
                self.visual_inputs_training = None
                self.actions_training_tf = None
                self.pieces_training_tf = None
            else:
                self.vector_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
                self.visual_inputs          = [tf.placeholder(tf.float32, (None,)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
                self.vector_inputs_training = [tf.placeholder(tf.float32, (None, k_step+1)+s[1:], name='vector_input{}'.format(i)) for i,s in enumerate(self.state_size_vec)]
                self.visual_inputs_training = [tf.placeholder(tf.float32, (None, k_step+1)+s[1:], name='visual_input{}'.format(i)) for i,s in enumerate(self.state_size_vis)]
                self.rewards_tf       = tf.placeholder(tf.float32, (None, k_step+1, 1), name='reward')
                self.dones_tf         = tf.placeholder(tf.int32, (None, k_step+1, 1), name='done')
                self.actions_tf       =  tf.placeholder(tf.uint8, (None, 1))
                self.pieces_tf        =  tf.placeholder(tf.uint8, (None, 1))
                self.actions_training_tf       = tf.placeholder(tf.uint8, (None, k_step+1, 2), name='action')
                self.pieces_training_tf        = tf.placeholder(tf.uint8, (None, k_step+1, 1), name='piece')
                self.learning_rate_tf       = tf.placeholder(tf.float32, shape=())
                self.loss_weights_tf        = tf.placeholder(tf.float32, (None,1), name='loss_weights')
            self.epsilon_tf       = tf.placeholder(tf.float32, shape=())
            self.keyboard_range_tf       = tf.placeholder(tf.float32, shape=())
            self.q_tf, self.v_tf, self.training_values_tf, self.target_values_tf, self.new_prios_tf, self.main_scope, self.ref_scope\
                                    = self.create_duelling_qnet(
                                                            self.vector_inputs,
                                                            self.visual_inputs,
                                                            self.vector_inputs_training,
                                                            self.visual_inputs_training,
                                                            self.actions_tf,
                                                            self.pieces_tf,
                                                            self.actions_training_tf,
                                                            self.pieces_training_tf,
                                                            self.rewards_tf,
                                                            self.dones_tf,
                                                           )
            self.main_net_vars      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.main_scope.name)
            self.main_net_assign_list      = self.create_weight_setting_ops(self.main_net_vars)
            if not worker_only:
                self.reference_net_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.ref_scope.name)
                self.reference_net_assign_list = self.create_weight_setting_ops(self.reference_net_vars)
                self.training_ops = self.create_training_ops()
            self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
            self.init_ops = tf.variables_initializer(self.all_vars)
        #Run init-op
        self.session.run(self.init_ops)

    def evaluate(self,
                inputs,
                epsilon=0.0,
                keyboard_range=None,
                ):
        vector, visual = inputs
        run_list = [self.q_tf, self.v_tf]
        if keyboard_range is None: keyboard_range = self.keyboard_range
        feed_dict = {self.epsilon_tf : epsilon,self.keyboard_range_tf : keyboard_range,}
        for idx, vec in enumerate(vector):
            feed_dict[self.vector_inputs[idx]] = vec
        for idx, vis in enumerate(visual):
            feed_dict[self.visual_inputs[idx]] = vis
        return_values = self.session.run(run_list, feed_dict=feed_dict)
        return return_values

    def train(  self,
                vector_states,
                visual_states,
                actions,
                pieces,
                rewards,
                dones,
                weights=None,
                lr=None,
                epsilon=0.0,
                keyboard_range=None,
              ):

        run_list = [
                    self.training_ops,
                    self.new_prios_tf,
                    self.stats_tensors,
                    ]
        if keyboard_range is None: keyboard_range = self.keyboard_range
        feed_dict = {
                        self.pieces_training_tf : pieces,
                        self.actions_training_tf : actions,
                        self.rewards_tf : rewards,
                        self.dones_tf : dones,
                        self.learning_rate_tf : lr,
                        self.epsilon_tf : epsilon,
                        self.keyboard_range_tf : keyboard_range,
                        self.loss_weights_tf : weights,
                    }
        #Add also the state information to the appropriate placeholders..
        for idx, vec in enumerate(vector_states):
            feed_dict[self.vector_inputs_training[idx]] = vec
        for idx, vis in enumerate(visual_states):
            feed_dict[self.visual_inputs_training[idx]] = vis
        _, new_prios, stats = self.session.run(run_list, feed_dict=feed_dict)
        return new_prios, zip(self.stats_tensors, stats)

    def create_vectorencoder(self, x):
        with tf.variable_scope("vectorencoder", reuse=tf.AUTO_REUSE) as vs:
            for n in range(self.settings['vectorencoder_n_hidden']):
                x = tf.layers.dense(
                                    x,
                                    self.settings['vectorencoder_hidden_size'],
                                    name='vectorencoder_layer{}'.format(n),
                                    activation=tf.nn.elu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                   )
            x = tf.layers.dense(
                                x,
                                self.settings['vectorencoder_output_size'],
                                name='layer{}'.format(self.settings['vectorencoder_n_hidden']+1),
                                activation=self.output_activation,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.contrib.layers.xavier_initializer(),
                               )
        return x

    def create_visualencoder(self, x):
        with tf.variable_scope("visualencoder", reuse=tf.AUTO_REUSE) as vs:
            if self.settings["pad_visuals"]:
                x = self.apply_visual_pad(x)
            for n in range(self.settings['visualencoder_n_convs']):
                y = tf.layers.conv2d(
                                        x,
                                        self.settings["visualencoder_n_filters"][n],
                                        self.settings["visualencoder_filter_sizes"][n],
                                        name='visualencoder_layer{}'.format(n),
                                        padding='same',
                                        activation=tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        bias_initializer=tf.zeros_initializer(),
                                    )
                if n in self.settings["visualencoder_peepholes"] and self.settings["peephole_convs"]:
                    x = tf.concat([y,x], axis=-1)
                else:
                    x = y
                if n in self.settings["visualencoder_poolings"]:
                    y = tf.layers.max_pooling2d(y, 2, 2, padding='same')
        return x

    def apply_visual_pad(self, x):
        #Apply zero-padding on top:
        x = tf.pad(x, [[0,0],[1,0],[0,0],[0,0]], constant_values=0.0)
        #Apply one-padding left, right and bottom:
        x = tf.pad(x, [[0,0],[0,1],[1,1],[0,0]], constant_values=1.0)
        # This makes floor and walls look like it's a piece, and cieling like its free space
        return x

    def create_kbd_visual(self,x):
        x = tf.layers.max_pooling2d(x, 2, 2, padding='valid')
        x = tf.layers.conv2d(
                                x,
                                16,
                                (3,3),
                                name='keyboard_vis_conv',
                                padding='same',
                                activation=tf.nn.elu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                bias_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            )
        x = tf.layers.max_pooling2d(x, 2, 2, padding='valid')
        return x

    def create_kbd(self, x):
        H = self.settings["game_size"][0] + 2 if self.settings["pad_visuals"] else self.settings["game_size"][0]
        x = tf.layers.conv2d(
                                x,
                                32,
                                (5,5),
                                name='keyboard_conv0',
                                padding='same',
                                activation=tf.nn.elu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                bias_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            )
        x = tf.layers.conv2d(
                                x,
                                self.n_rotations * self.n_pieces,
                                (H,3),
                                name='keyboard_conv1',
                                padding='valid',
                                activation=self.advantage_activation_sqrt,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                bias_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            )
        #Interpret with of field as translations for the piece W ~> T, then:
        # [?, 1, T, R*P] -> [?, T, R, P] -> [?, R, T, P]
        x = tf.reshape(x, [-1, 10, 4, 7])
        x = tf.transpose(x, perm=[0,2,1,3])
        return x

    def advantage_activation_sqrt(self,x):
        # ret = tf.sign(x) * tf.minimum( tf.sqrt( tf.abs(x) ), tf.abs(x))
        alpha = 0.01
        ret = tf.sign(x) * (tf.sqrt( tf.abs(x) + alpha**2) - alpha)
        ret = tf.where( tf.is_nan(ret), x, ret)
        return ret

    def create_q_head(self,vectors, visuals, name):
        with tf.variable_scope("q-net-"+name, reuse=tf.AUTO_REUSE) as vs:
            scope = vs
            hidden_vec = [self.create_vectorencoder(vec) for vec in vectors]
            if self.settings["keyboard_conv"]:
                _visuals = [self.create_visualencoder(vis) for vis in visuals]
                hidden_vis = [self.create_kbd_visual(vis) for vis in _visuals]
                kbd = self.create_kbd(_visuals[0]) #"my screen -> my kbd"
            else:
                hidden_vis = [self.create_visualencoder(vis) for vis in visuals]
            flat_vec = [tf.layers.flatten(hv) for hv in hidden_vec]
            flat_vis = [tf.layers.flatten(hv) for hv in hidden_vis]
            x = tf.concat(flat_vec+flat_vis, axis=-1)

            ###
            #####
            for n in range(self.settings['valuenet_n_hidden']):
                x = tf.layers.dense(
                                    x,
                                    self.settings['valuenet_hidden_size'],
                                    name='layer{}'.format(n),
                                    activation=tf.nn.elu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                   )
            V = tf.layers.dense(
                                x,
                                1,
                                name='values',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.contrib.layers.xavier_initializer(),
                                activation=self.settings["nn_output_activation"],
                                # bias_initializer=tf.zeros_initializer(),
                               )
            n_val_outputs = 1 if not self.settings["keyboard_separate_piecevalues"] else self.n_pieces
            if self.settings["keyboard_separate_piecevalues"]:
                _A_p = tf.layers.dense(
                                    x,
                                    self.n_pieces,
                                    name='value_pieceadvantages_unnormalized',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=self.advantage_activation_sqrt,
                                   )
                V = V + 0.3 * (_A_p - tf.reduce_mean(_A_p, axis=1, keepdims=True))
            _V = tf.reshape(V,[-1,1,1,n_val_outputs]) #Shape for Q-calc!
            #####
            ###

            if self.settings["keyboard_conv"]:
                A = self.keyboard_range_tf * kbd
            else:
                _A = tf.layers.dense(
                                    x,
                                    self.n_rotations * self.n_translations * self.n_pieces,
                                    name='advantages_unshaped',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=self.settings["nn_output_activation"],
                                    # bias_initializer=tf.zeros_initializer(),
                                   )
                A = self.keyboard_range_tf * tf.reshape(_A, [-1, self.n_rotations, self.n_translations, self.n_pieces])

            #
            ###
            #####
            a = A
            if self.settings["keyboard_separate_piecevalues"]:
                #Don't blend
                mean_a = tf.reduce_mean(      a, axis=[1,2], keepdims=True )
                max_a = tf.reduce_max(       a, axis=[1,2],   keepdims=True )
                #The value of the state is the average of the values of the state-piece combinations
                V = tf.reshape(tf.reduce_sum(_V * self.used_pieces_mask_tf, axis=3), [-1,1]) / self.n_used_pieces
            else:
                #Blend
                a_maxmasked = self.used_pieces_mask_tf * a + (1-self.used_pieces_mask_tf) * tf.reduce_min(a, axis=[1,2,3], keepdims=True)
                max_a   = tf.reduce_max(a_maxmasked,  axis=[1,2,3],     keepdims=True ) #We max over the actions which WE control (rotation, translation) and average over the ones we dont control (piece)
                #
                _mean_a = tf.reduce_mean(a,      axis=[1,2], keepdims=True )
                mean_a  = tf.reduce_sum(_mean_a, axis=3,     keepdims=True ) / self.n_used_pieces

            A_q    = (a - max_a)  #A_q(s,r,t,p) = advantage of applying rotation r and translation t to piece p in state s; compared to playing according to the argmax-policy
            A_unif = (a - mean_a) #similar but compared to random action
            e = tf.clip_by_value(self.epsilon_tf, 0, 1)
            Q = _V + e * A_unif + (1-e) * A_q
        return Q, V, scope

    def create_duelling_qnet(self, vector_states, visual_states, vector_states_training, visual_states_training, actions, pieces, actions_training, pieces_training, rewards, dones):
        with tf.variable_scope("prio_q-net") as vs:
            q_tf, v_tf, main_scope = self.create_q_head(vector_states, visual_states, "main")

            #Workers are easy!
            if self.worker_only:
                return q_tf, v_tf, None, None, None, main_scope, None

            # # #
            # Trainers do Q-updates:
            # # #

            #1) get all the values and a bool to tell if the round is over
            dones_tf = tf.minimum(1, tf.cumsum(dones, axis=1)) #Maximum ensures there is no stray done from an adjacent trajectory influencing us...
            done_time_tf = tf.reduce_sum( 1-dones_tf, axis=1)

            q_t_tf, v_t_tf = [], []
            for t in range(self.k_step+1):
                scope = "main" if t==0 else "reference"
                s_t_vec = [vec_state[:,t,:] for vec_state in vector_states_training]
                s_t_vis = [vis_state[:,t,:] for vis_state in visual_states_training]
                q,v,ref_scope = self.create_q_head(s_t_vec, s_t_vis, scope)
                q_t_tf.append(q)
                v_t_tf.append(v)
            gamma = -self.settings["gamma"] if self.settings["single_policy"] else self.settings["gamma"]
            def k_step_estimate(k):
                e = 0
                for t in range(k):
                    e += rewards[:,t,:] * tf.cast((done_time_tf >= t),tf.float32) * (gamma**t)
                e += v_t_tf[k] * tf.cast((done_time_tf >= k),tf.float32) * (gamma**k)
                return e
            estimators_tf = [k_step_estimate(k) for k in range(1,self.k_step+1)]
            #3) GAE-style aggregation
            weight = 0
            estimator_sum_tf = 0
            gae_lambda = self.settings["gae_lambda"]
            for e in reversed(estimators_tf):
                estimator_sum_tf *= gae_lambda
                weight *= gae_lambda
                estimator_sum_tf += e
                weight += 1
            target_values_tf = tf.stop_gradient(estimator_sum_tf / weight)

            #Also we need a Q-value to update; Q(s,a,x)!
            Q_s_all = q_t_tf[0]
            r_mask = tf.reshape(tf.one_hot(actions_training[:,0,0], self.n_rotations),    (-1, self.n_rotations,    1,  1), name='r_mask')
            t_mask = tf.reshape(tf.one_hot(actions_training[:,0,1], self.n_translations), (-1,  1, self.n_translations, 1), name='t_mask')
            p_mask = tf.reshape(tf.one_hot(pieces_training[:,0,:],  self.n_pieces),       (-1,  1,  1, self.n_pieces     ), name='p_mask')
            rtp_mask = r_mask*t_mask*p_mask
            _Q_s = Q_s_all * rtp_mask
            Q_s = tf.reduce_sum(_Q_s, axis=[1,2,3])
            training_values_tf = tf.expand_dims(Q_s,1)

            # Prio-Q-net is true to it's name and computes the sample priorities for you!
            if self.settings["optimistic_prios"] == 0.0:
                prios_tf = tf.abs(training_values_tf - target_values_tf) #priority for the experience replay
            else:
                prios_tf = tf.abs(training_values_tf - target_values_tf) + self.settings["optimistic_prios"] * tf.nn.relu(target_values_tf - training_values_tf)

            #As always - we like some stats!
            self.output_as_stats(training_values_tf, name="training-val")
            self.output_as_stats(target_values_tf, name="target-val")
            self.output_as_stats(done_time_tf, name="done_time")
            for i,e in enumerate(estimators_tf):
                self.output_as_stats(e, name='{}_step_val_est'.format(i+1))
                self.output_as_stats(tf.abs(target_values_tf-e), name='{}-step_val_est_diff'.format(i+1))
        return q_tf, v_tf, training_values_tf, target_values_tf, prios_tf, main_scope, ref_scope

    def create_training_ops(self):
        if self.worker_only:
            return None
        self.value_loss_tf = tf.losses.mean_squared_error(self.target_values_tf, self.training_values_tf, weights=self.loss_weights_tf)
        self.regularizer_tf = self.settings["nn_regularizer"] * tf.add_n([tf.nn.l2_loss(v) for v in self.main_net_vars])
        self.loss_tf = self.value_loss_tf + self.regularizer_tf
        training_ops = self.settings["optimizer"](learning_rate=self.learning_rate_tf).minimize(self.loss_tf)
        #Stats: we like stats.
        self.output_as_stats(self.loss_tf, name='tot_loss', only_mean=True)
        self.output_as_stats(self.value_loss_tf, name='value_loss', only_mean=True)
        self.output_as_stats(self.regularizer_tf, name='reg_loss', only_mean=True)
        return training_ops

    def create_weight_setting_ops(self, collection):
        assign_placeholder_list = []
        for var in collection:
            shape, dtype = var.shape, var.dtype
            assign_val_placeholder_tf = tf.placeholder(shape=shape, dtype=dtype)
            assign_op_tf = var.assign(assign_val_placeholder_tf)
            assign_placeholder_list.append(
                                            {
                                                "assign_op" : assign_op_tf,
                                                "assign_val_placeholder" : assign_val_placeholder_tf,
                                            }
                                          )
        return assign_placeholder_list

    def swap_networks(self):
        # Swaps reference- and main-weights
        main_weights = self.get_weights(self.main_net_vars)
        ref_weights  = self.get_weights(self.reference_net_vars)
        self.set_weights(self.reference_net_assign_list, main_weights)
        self.set_weights(self.main_net_assign_list, ref_weights )

    def reference_update(self):
        main_weights = self.get_weights(self.main_net_vars)
        self.set_weights(self.reference_net_assign_list,main_weights)

    def get_weights(self, collection):
        ret = self.session.run(collection)
        return ret

    def set_weights(self, assign_list, weights):
        run_list = []
        feed_dict = {}
        for w,assign in zip(weights,assign_list):
            run_list.append(assign['assign_op'])
            feed_dict[assign['assign_val_placeholder']] = w
        self.session.run(run_list, feed_dict=feed_dict)

    def check_gpu(self):
        for dev in device_lib.list_local_devices():
            if "GPU" in dev.name:
                return True
        return False

    def output_as_stats(self, tensor, name=None, only_mean=False):
        if name is None:
            name = tensor.name
        #Corner case :)
        if len(tensor.shape) == 0:
            self.stats_tensors.append(tf.identity(tensor, name=name))
            return
        self.stats_tensors.append(tf.reduce_mean(tensor, axis=0, name=name+'_mean'))
        if only_mean:
            return
        self.stats_tensors.append(tf.reduce_max(tensor, axis=0, name=name+'_max'))
        self.stats_tensors.append(tf.reduce_min(tensor, axis=0, name=name+'_min'))

    @property
    def output(self):
        return self.q_tf
