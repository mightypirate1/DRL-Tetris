import time
import logging
import numpy as np
import tools.utils as utils
from agents.sventon_agent.sventon_agent_trainer_base import sventon_agent_trainer_base

logger = logging.getLogger(__name__)

class sventon_agent_ppo_trainer(sventon_agent_trainer_base):
    def do_training(self, time=0, sample=None, policy=None):
        minibatch_size, n_epochs, n_enough_samples_for_training, update_prio_flag = self.settings["minibatch_size"], self.n_train_epochs, self.settings["n_samples_each_update"], False
        #Figure out what policy, model, and experience replay to use...
        if self.settings["single_policy"]:
            policy = "main_net"
        else:
            assert policy is not None, "In multi-policy mode you must specify which model to train!"
            policy = "policy_{}".format(policy)

        exp_rep = self.experience_replay_dict[policy]
        model = self.model_dict[policy]
        #If we dont have enough samples yet we quit early...
        if sample is None and len(exp_rep) < n_enough_samples_for_training:
            return 0, None

        # # #
        #Start!
        # # #

        #1) Get a sample!
        if sample is None: #If no one gave us one, we get one ourselves!
            sample = exp_rep.retrieve_all(compute_stats=True)

        #Unpack a little...
        states, _actions, rewards, dones = sample
        a_env, a_int = _actions
        actions, pieces = a_env[:,:,:2], a_env[:,:,2,np.newaxis]
        probabilities, advantages, target_values = a_int[:,:,0,np.newaxis], a_int[:,:,1,np.newaxis], a_int[:,:,2,np.newaxis]
        vector_states, visual_states = states
        n = len(actions) #n samples

        #TRAIN!
        stats = []
        lr =  utils.evaluate_params(self.settings["value_lr"], time)
        train_params = utils.evaluate_params(self.settings["ppo_parameters"], time)
        for t in range(n_epochs):
            if self.verbose_training: print("[",end='',flush=False); last_print = 0
            last_epoch = t+1 == n_epochs
            perm = np.random.permutation(n)
            for i in range(0,n,minibatch_size):
                batchstats = model.train(
                    [vec_s[perm[i:i+minibatch_size]] for vec_s in vector_states],
                    [vis_s[perm[i:i+minibatch_size]] for vis_s in visual_states],
                    actions[perm[i:i+minibatch_size]],
                    pieces[perm[i:i+minibatch_size]],
                    probabilities[perm[i:i+minibatch_size]],
                    advantages[perm[i:i+minibatch_size]],
                    target_values[perm[i:i+minibatch_size]],
                    rewards[perm[i:i+minibatch_size]],
                    dones[perm[i:i+minibatch_size]],
                    lr=lr,
                    **train_params,
                )
                stats.append(batchstats)
                if self.verbose_training and (i-last_print)/n > 0.02: print("-",end='',flush=False); last_print = i
            if self.verbose_training: print("]",flush=False)
        # Clear AFTER training so that samples don't get lost on SIGINT
        exp_rep.initialize()

        #Sometimes we do a reference update
        if self.time_to_reference_update[policy] == 0:
            self.reference_update(net=policy)
            self.time_to_reference_update[policy] = self.settings["time_to_reference_update"]
        else:
            self.time_to_reference_update[policy] -= 1

        trainingstats = self.merge_training_stats(stats)
        return n, trainingstats
