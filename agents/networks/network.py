import tensorflow as tf
import tools.utils as utils
from agents.networks.builders import sventon_architectures as arch

class network:
    def __init__(self, agent_id, name, state_size, output_shape, session, k_step=1, settings=None, worker_only=False, full_network=True):
        #Basics
        self.settings = utils.parse_settings(settings)
        self.name = name
        self.scope = tf.variable_scope("agent{}_{}".format(agent_id,name))
        self.session = session
        self.worker_only = worker_only
        self.full_network = full_network
        self.stats_tensors, self.debug_tensors, self.visualization_tensors = [], [], []
        assert len(output_shape) == 3, "expected 3D-actions"
        assert k_step > 0, "k_step AKA n_step_value_estimates has to be greater than 0!"

        #Shapes and params etc
        self.output_shape = self.n_rotations, self.n_translations, self.n_pieces = output_shape
        self.state_size = self.state_size_vec, self.state_size_vis = state_size
        self.k_step = k_step
        self.estimator_filter = None if not "sparse_value_estimate_filter" in self.settings else self.settings["sparse_value_estimate_filter"]
        self.gamma = -self.settings["gamma"] if self.settings["single_policy"] else self.settings["gamma"]
        self._lambda = self.settings["gae_lambda"]
        if self.settings["architecture"] == "silver":
            self.network_type = arch.resblock_kbd
        elif self.settings["architecture"] == "vanilla":
            self.network_type = arch.convthendense
        elif self.settings["architecture"] == "keyboard":
            self.network_type = arch.convkeyboard
        elif self.settings["architecture"] == "dreamer":
            self.network_type = arch.resblock
    ###
    ### Utilities etc...
    ###
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
        main_weights = self.get_weights(self.main_net.variables)
        ref_weights  = self.get_weights(self.reference_net.variables)
        self.set_weights(self.reference_net_assign_list, main_weights)
        self.set_weights(self.main_net_assign_list, ref_weights )

    def reference_update(self):
        main_weights = self.get_weights(self.main_net.variables)
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

    def get_random_conv_layers(self,tensor, n):
        _max = tensor.shape.as_list()[3]
        if _max - n > 3:
            start = np.random.choice(_max-n)
            idxs = slice(start, start+3)
        else:
            idxs = slice(0,n)
        return tensor[0,:,:,idxs]
