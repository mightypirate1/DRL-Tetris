
import tensorflow as tf
import numpy as np
import aux
import aux.utils as utils
from agents.networks import network_utils as N

'''
TODO: Clean up by separating out default chunks of neural-net construction. We don't need a bunch of conv-layer loops all over, do we?
'''

class prio_qnet:
    def evaluate(self,
                inputs,
                only_policy=False,
                ):
        vector, visual = inputs
        run_list = [self.q_tf, self.v_tf] if not only_policy else [self.a_tf]
        feed_dict = {self.training_tf : False}
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
                fetch_visualizations=False,
              ):
        vis_tensors = [] if not fetch_visualizations else self.visualization_tensors
        run_list = [
                    self.training_ops,
                    self.new_prios_tf,
                    self.stats_tensors,
                    self.debug_tensors,
                    vis_tensors,
                    ]
        feed_dict = {
                        self.pieces_training_tf : pieces,
                        self.actions_training_tf : actions,
                        self.rewards_tf : rewards,
                        self.dones_tf : dones,
                        self.learning_rate_tf : lr,
                        self.loss_weights_tf : weights,
                        self.training_tf : True,
                    }
        #Add also the state information to the appropriate placeholders..
        for idx, vec in enumerate(vector_states):
            feed_dict[self.vector_inputs_training[idx]] = vec
        for idx, vis in enumerate(visual_states):
            feed_dict[self.visual_inputs_training[idx]] = vis
        _, new_prios, stats, dbg, vis = self.session.run(run_list, feed_dict=feed_dict)
        N.debug_prints(dbg,self.debug_tensors)
        return new_prios, zip(self.stats_tensors, stats), zip(self.visualization_tensors, vis)

    def create_duelling_qnet(self, vector_states, visual_states, vector_states_training, visual_states_training, actions, pieces, actions_training, pieces_training, rewards, dones):
        with tf.variable_scope("prio_q-net") as vs:
            q_tf, v_tf, a_tf, main_scope = self.create_q_head(vector_states, visual_states, "main")

            #Workers are easy!
            if self.worker_only:
                return q_tf, v_tf, a_tf, None, None, None, main_scope, None

            # # #
            # Trainers do Q-updates:
            # # #

            #1) Evaluate all the states, and create a bool to tell if the round is over
            dones_tf = tf.minimum(1, tf.cumsum(dones, axis=1)) #Minimum ensures there is no stray done from an adjacent trajectory influencing us...
            done_time_tf = tf.reduce_sum( 1-dones_tf, axis=1)
            q_t_tf, v_t_tf = [], []
            for t in range(self.k_step+1):
                scope = "main" if t==0 else "reference"
                s_t_vec = [vec_state[:,t,:] for vec_state in vector_states_training]
                s_t_vis = [vis_state[:,t,:] for vis_state in visual_states_training]
                q,v,_,ref_scope = self.create_q_head(s_t_vec, s_t_vis, scope)
                q_t_tf.append(q)
                v_t_tf.append(v)

            #2) Do all the V-estimators, k-step style
            gamma = -self.settings["gamma"] if self.settings["single_policy"] else self.settings["gamma"]
            def k_step_estimate(k):
                e = 0
                for t in range(k):
                    e += rewards[:,t,:] * tf.cast((done_time_tf >= t),tf.float32) * (gamma**t)
                e += v_t_tf[k] * tf.cast((done_time_tf >= k),tf.float32) * (gamma**k)
                return e
            estimator_steps = [i for i in range(1,self.k_step+1)]
            if "sparse_value_estimate_filter" in self.settings: #filter out all k that are divisible by any of the numbers provided
                filter = np.array(self.settings["sparse_value_estimate_filter"]).reshape((1,-1))
                steps = np.array(estimator_steps).reshape((-1,1))
                filter = np.prod((steps % _filter),axis=1)!=0
                estimator_steps = steps[np.where(filter)]
            estimators = [(k,k_step_estimate(k)) for k in estimator_steps if k <= self.k_step]

            #3) GAE-style aggregation
            weight = 0
            estimator_sum_tf = 0
            gae_lambda = self.settings["gae_lambda"]
            for k,e in estimators:
                estimator_sum_tf += e * gae_lambda**k
                weight += gae_lambda**k
            value_estimator_tf = estimator_sum_tf / weight

            #4) Prepare a training value!
            Q_s_all = q_t_tf[0]
            r_mask = tf.reshape(tf.one_hot(actions_training[:,0,0], self.n_rotations),    (-1, self.n_rotations,    1,  1), name='r_mask')
            t_mask = tf.reshape(tf.one_hot(actions_training[:,0,1], self.n_translations), (-1,  1, self.n_translations, 1), name='t_mask')
            p_mask = tf.reshape(tf.one_hot(pieces_training[:,0,:],  self.n_pieces),       (-1,  1,  1, self.n_pieces     ), name='p_mask')
            rtp_mask = r_mask*t_mask*p_mask
            _Q_s = Q_s_all * rtp_mask
            Q_s = tf.reduce_sum(_Q_s, axis=[1,2,3])

            #5) Target and predicted values. Also prios for the exp-rep.
            target_values_tf = tf.stop_gradient(value_estimator_tf)
            training_values_tf = tf.expand_dims(Q_s,1)
            prios_tf = tf.abs(training_values_tf - target_values_tf)
            if self.settings["optimistic_prios"] != 0.0:
                prios_tf += self.settings["optimistic_prios"] * tf.nn.relu(prios_tf)

            if self.settings["q_target_locked_for_other_actions"]:
                # self.LOSS_FCN_DBG(Q_s_all, rtp_mask, value_estimator_tf)
                target_values_tf = self.create_fixed_targets(Q_s_all, rtp_mask, target_values_tf)
                training_values_tf = Q_s_all

            #As always - we like some stats!
            self.output_as_stats(value_estimator_tf, name="target-val")
            self.output_as_stats(done_time_tf, name="done_time")
        return q_tf, v_tf, a_tf, training_values_tf, target_values_tf, prios_tf, main_scope, ref_scope

    def create_q_head(self,vectors, visuals, name):
        with tf.variable_scope("q-net-"+name, reuse=tf.AUTO_REUSE) as vs:
            scope = vs
            #1) create visual- and vector-encoders for the inputs!
            hidden_vec = [self.create_vectorencoder(vec) for vec in vectors]
            if self.settings["keyboard_conv"]:
                _visuals = [self.create_visualencoder(vis) for vis in visuals]
                hidden_vis = [self.create_kbd_visual(vis) for vis in _visuals]
                A_kbd = self.create_kbd(_visuals[0]) #"my screen -> my kbd"
            else:
                hidden_vis = [self.create_visualencoder(vis) for vis in visuals]
            flat_vec = [tf.layers.flatten(hv) for hv in hidden_vec]
            flat_vis = [tf.layers.flatten(hv) for hv in hidden_vis]

            #2) Take some of the data-stream and compute a value-estimate
            x = tf.concat(flat_vec+flat_vis, axis=-1)
            V = self.create_value_head(x)
            V_qshaped = tf.reshape(V,[-1,1,1,V.shape.as_list()[-1]]) #Shape for Q-calc!

            #3) Compute advantages
            if self.settings["keyboard_conv"]:
                #Or if we just have them... (if I move that line down, I break some backward compatibility)
                A = self.keyboard_range * A_kbd
            else:
                _A = tf.layers.dense(
                                    x,
                                    self.n_rotations * self.n_translations * self.n_pieces,
                                    name='advantages_unshaped',
                                    activation=N.advantage_activation_sqrt,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(),
                                   )
                A = self.keyboard_range * tf.reshape(_A, [-1, self.n_rotations, self.n_translations, self.n_pieces])

            #4) Combine values and advantages to form the Q-fcn (Duelling-Q-style)
            if self.settings["advantage_type"] == "max":
                a_maxmasked = self.used_pieces_mask_tf * A + (1-self.used_pieces_mask_tf) * tf.reduce_min(A, axis=[1,2,3], keepdims=True)
                _max_a   = tf.reduce_max(a_maxmasked,  axis=[1,2],     keepdims=True ) #We max over the actions which WE control (rotation, translation) and average over the ones we dont control (piece)
                if self.settings["separate_piece_values"]:
                    max_a = _max_a
                else:
                    max_a   = tf.reduce_sum(_max_a * self.used_pieces_mask_tf,  axis=3,     keepdims=True ) / self.n_used_pieces #We max over the actions which WE control (rotation, translation) and average over the ones we dont control (piece)
                A  = (A - max_a)  #A_q(s,r,t,p) = advantage of applying rotation r and translation t to piece p in state s; compared to playing according to the argmax-policy
                Q = V_qshaped + A
            elif self.settings["advantage_type"] == "mean":
                _mean_a = tf.reduce_mean(A,      axis=[1,2], keepdims=True )
                # mean_a  = tf.reduce_mean(_mean_a, axis=3,     keepdims=True )
                mean_a  = tf.reduce_sum(_mean_a * self.used_pieces_mask_tf, axis=3,     keepdims=True ) / self.n_used_pieces
                A = (A - mean_a)
                Q = V_qshaped + A
            else:
                Q = A
        V = self.q_to_v(Q)
        return Q, V, A, scope

    ###
    ### Above is the blue-print, below the details
    ###

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
                x = N.apply_visual_pad(x)
                if self.conv_debug.pop("visual_input", False):
                    self.visualization_tensors.append(self.get_random_conv_layers(x,3))
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
                if n in self.conv_debug["visualencoder"]:
                    self.visualization_tensors.append(self.get_random_conv_layers(y,3))
                    self.conv_debug["visualencoder"].remove(n)
                if "visualencoder_dropout" in self.settings:
                    x = tf.keras.layers.SpatialDropout2D(self.settings["visualencoder_dropout"])(x,self.training_tf)
                if n in self.settings["visualencoder_peepholes"] and self.settings["peephole_convs"]:
                    x = N.peephole_join(x,y,mode=self.settings["peephole_join_style"])
                else:
                    x = y
                if n in self.settings["visualencoder_poolings"]:
                    x = tf.layers.max_pooling2d(x, (2,1), (2,1), padding='same')
        return x

    def create_kbd_visual(self,x):
        x = tf.layers.max_pooling2d(x, 2, 2, padding='valid')
        for i in range(self.settings["kbd_vis_n_convs"]):
            x = N.layer_pool(
                            x,
                            n_filters=self.settings["kbd_vis_n_filters"][i],
                            filter_size=(3,3),
                            dropout=(self.settings["visualencoder_dropout"],self.training_tf)
                            )
        x = tf.reduce_mean(x, axis=[1,2])
        return x

    def create_kbd(self, x):
        for i in range(self.settings["keyboard_n_convs"]-1):
            x = tf.layers.conv2d(
                                    x,
                                    self.settings["keyboard_n_filters"][i],
                                    (5,5),
                                    name='keyboard_conv{}'.format(i),
                                    padding='same',
                                    activation=tf.nn.elu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    bias_initializer=tf.zeros_initializer(),
                                )
            if "visualencoder_dropout" in self.settings:
                x = tf.keras.layers.Dropout(self.settings["visualencoder_dropout"])(x,self.training_tf)
        x = tf.layers.conv2d(
                                x,
                                self.n_rotations * self.n_pieces,
                                (x.shape.as_list()[1],3),
                                name='keyboard_conv{}'.format(self.settings["keyboard_n_convs"]-1),
                                padding='valid',
                                activation=N.advantage_activation_sqrt,
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                bias_initializer=tf.zeros_initializer(),
                            )
        X = [ x[:,:,:,p*self.n_pieces:(p+1)*self.n_pieces ] for p in range(self.n_rotations) ]
        x = tf.concat(X, axis=1)
        return x
        # #Interpret with of field as translations for the piece W ~> T, then:
        # # [?, 1, T, R*P] -> [?, T, R, P] -> [?, R, T, P]
        # x = tf.reshape(x, [-1, 10, 4, 7])
        # x = tf.transpose(x, perm=[0,2,1,3])
        # return x

    def create_value_head(self, x):
        for n in range(self.settings['valuenet_n_hidden']):
            x = tf.layers.dense(
                                x,
                                self.settings['valuenet_hidden_size'],
                                name='layer{}'.format(n),
                                activation=tf.nn.elu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer(),
                               )
        v = tf.layers.dense(
                            x,
                            1,
                            name='values',
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            activation=self.settings["nn_output_activation"],
                            bias_initializer=tf.zeros_initializer(),
                           )
        if not self.settings["separate_piece_values"]:
            return v
        v_p = tf.layers.dense(
                              x,
                              7,
                              name='piece_values',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              activation=None,
                              bias_initializer=tf.zeros_initializer(),
                             )
        v_p = self.settings["piece_advantage_range"] * N.advantage_activation_sqrt(v_p - tf.reduce_mean(v_p, axis=1, keepdims=True))
        return v + v_p

    def create_training_ops(self):
        if self.worker_only:
            return None
        if self.settings["q_target_locked_for_other_actions"]:
            sq_error = tf.math.squared_difference(self.target_values_tf, self.training_values_tf)
            ssq_error = tf.expand_dims(tf.reduce_sum(sq_error,axis=[1,2,3]),1)
            self.value_loss_tf = tf.losses.mean_squared_error(ssq_error, [[0.0]], weights=self.loss_weights_tf)
            self.argmax_entropy_tf = N.argmax_entropy_reg(
                                                          self.training_values_tf,
                                                          mask=self.used_pieces_mask_tf,
                                                          n_pieces=self.n_used_pieces,
                                                         )
            self.output_as_stats(self.argmax_entropy_tf, name="argmax_entropy")
        else:
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

    def create_fixed_targets(self, Q, mask, target):
        _target = tf.reshape(target, [-1,1,1,1]) * mask
        return _target + (1-mask) * Q

    def q_to_v(self,q):
        q_p = tf.reduce_max(q, axis=[1,2], keepdims=True)
        v = tf.reduce_sum(q_p * self.used_pieces_mask_tf, axis=3, keepdims=True) / self.n_pieces
        return tf.reshape(v, [-1, 1])

    def get_random_conv_layers(self,tensor, n):
        _max = tensor.shape.as_list()[3]
        if _max - n > 3:
            start = np.random.choice(_max-n)
            idxs = slice(start, start+3)
        else:
            idxs = slice(0,n)
        return tensor[0,:,:,idxs]

    @property
    def output(self):
        return self.q_tf

    ####
    #### Just a bunch of variable initialization and naming...
    ####
    def __init__(self, agent_id, name, state_size, output_shape, sess, k_step=1, settings=None, worker_only=False):
        assert len(output_shape) == 3, "expected 3D-actions"
        assert k_step > 0, "k_step AKA n_step_value_estimates has to be greater than 0!"

        #Basics
        self.settings = utils.parse_settings(settings)
        self.scope_name = "agent{}_{}".format(agent_id,name)
        self.name = name
        self.session = sess
        self.worker_only = worker_only

        #Shapes and params
        self.output_shape = self.n_rotations, self.n_translations, self.n_pieces = output_shape
        self.output_size = self.n_rotations * self.n_translations * self.n_pieces
        self.n_used_pieces = len(self.settings["pieces"])
        self.state_size_vec, self.state_size_vis = state_size
        self.k_step = k_step
        self.output_activation = settings["nn_output_activation"]

        #DBG
        # self.conv_debug = {"visual_input":True, "visualencoder":[0,1,2,3]}
        self.conv_debug = {"visual_input":False, "visualencoder":[]}
        self.stats_tensors, self.debug_tensors, self.visualization_tensors = [], [], []

        #Define tensors/placeholders
        self.keyboard_range = self.settings["keyboard_range"]
        used_pieces = [0, 0, 0, 0, 0, 0, 0]
        for i in range(7):
            if i in self.settings["pieces"]:
                used_pieces[i] = 1
                self.used_pieces_mask_tf = tf.constant(np.array(used_pieces).reshape((1,1,1,7)).astype(np.float32))

        print("REMOVE DBG-prints!!!!")
        print("REMOVE DBG-prints!!!!")
        print("REMOVE DBG-prints!!!!")
        # D = np.zeros((1,4,10,7))
        # for a in range(4):
        #     for b in range(10):
        #         for c in range(7):
        #             D[0,a,b,c] = a*100+b*10+c
        # D = tf.constant(D, dtype=tf.float32)
        # d, dmasked = sess.run([D, D*self.used_pieces_mask_tf])
        # for p in [0,1,5,6,]:
        #     print("piece",p)
        #     print(dmasked[0,:,:,p])
        # exit()

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
            self.training_tf       = tf.placeholder(tf.bool, shape=())
            self.q_tf, self.v_tf, self.a_tf, self.training_values_tf, self.target_values_tf, self.new_prios_tf, self.main_scope, self.ref_scope\
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
