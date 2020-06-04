import time
import numpy as np
import agents.sventon_agent.sventon_utils as S
from agents.sventon_agent.sventon_agent_trainer_base import sventon_agent_trainer_base

class sventon_agent_dqn_trainer(sventon_agent_trainer_base):
    def do_training(self, sample=None, policy=None):
        minibatch_size, n_epochs, n, update_prio_flag = self.settings["minibatch_size"], self.settings["n_train_epochs_per_update"], self.settings["n_samples_each_update"], False
        self.train_lossstats_raw = list()
        #Figure out what policy, model, and experience replay to use...
        if self.settings["single_policy"]:
            policy = "main_net"
        else:
            assert policy is not None, "In multi-policy mode you must specify which model to train!"
            policy = "policy_{}".format(policy)
            if self.scoreboard[policy] > 0.5 + self.settings["winrate_tolerance"]:
                print("Policy \"{}\" did NOT TRAIN, due to too much winning! ({})".format(policy, self.scoreboard[policy]))
                return 0
        exp_rep = self.experience_replay_dict[policy]
        model = self.model_dict[policy]
        #If we dont have enough samples yet we quit early...
        if sample is None and len(exp_rep) < self.n_samples_to_start_training:
            if not self.settings["run_standalone"]: time.sleep(1) #If we are a separate thread, we can be a little patient here
            return 0

        # # #
        #Start!
        # # #

        #1) Get a sample!
        t_sample = time.time()
        if sample is None: #If no one gave us one, we get one ourselves!
            update_prio_flag = True #If we sampled this ourselves, we take responsibility for updatig the prio of it
            sample, is_weights, filter = \
                            exp_rep.get_random_sample(
                                                        n,
                                                        alpha=self.settings["prioritized_replay_alpha"].get_value(self.clock),
                                                        beta=self.settings["prioritized_replay_beta"].get_value(self.clock),
                                                        compute_stats=True,
                                                      )
        #Unpack a little...
        states_k, _actions_k, rewards_k, dones_k = sample
        _actions_env_k, _actions_int_k = _actions_k
        actions_k, pieces_k = _actions_env_k[:,:,:2], _actions_env_k[:,:,2,np.newaxis]
        vector_states_k, visual_states_k = states_k

        #Create targets for Q-updates:
        print("@",flush=True)
        t_create_targets = time.time()
        vector_states, visual_states, actions, pieces = [S.un_k_step(x, 0) for x in [vector_states_k, visual_states_k, actions_k, pieces_k]]
        targets = np.zeros((n,1))
        for i in range(0,n,minibatch_size):
            start, stop = i, min(n,i+minibatch_size)
            _targets = model.compute_targets(
                                             [vec[start:stop] for vec in vector_states_k],
                                             [vis[start:stop] for vis in visual_states_k],
                                             rewards_k[start:stop],
                                             dones_k[start:stop],
                                             time_stamps=None
                                            )
            targets[start:stop] = _targets

        #TRAIN!
        print("¤",flush=True)
        t_updates = time.time()
        new_prio = np.empty((n,1))
        for t in range(n_epochs):
            if self.verbose_training: print("[",end='',flush=False); last_print = 0
            last_epoch = t+1 == n_epochs
            perm = np.random.permutation(n) if not last_epoch else np.arange(n)
            for i in range(0,n,minibatch_size):
                _new_prio, stats, = model.train(
                                                [vec_s[perm[i:i+minibatch_size]] for vec_s in vector_states],
                                                [vis_s[perm[i:i+minibatch_size]] for vis_s in visual_states],
                                                actions[perm[i:i+minibatch_size]],
                                                pieces[perm[i:i+minibatch_size]],
                                                targets[perm[i:i+minibatch_size]],
                                                weights=is_weights[perm[i:i+minibatch_size]],
                                                lr=self.settings["value_lr"].get_value(self.clock),
                                               )
                self.train_stats_raw.append(stats)
                if last_epoch: new_prio[i:i+minibatch_size] = _new_prio
                if self.verbose_training and (i-last_print)/n > 0.02: print("-",end='',flush=False); last_print = i
            if self.verbose_training: print("]",flush=False)

        #Sometimes we do a reference update
        if self.time_to_reference_update[policy] == 0:
            self.reference_update(net=policy)
            self.time_to_reference_update[policy] = self.settings["time_to_reference_update"]
        else:
            self.time_to_reference_update[policy] -= 1

        #Count training
        self.n_train_steps['total'] += 1
        self.n_train_steps[policy]  += 1
        #Some stats:

        self.generate_training_stats()
        self.update_stats(exp_rep.stats, scope="ExpRep_"+policy)

        #Update prios if we sampled the sample ourselves...
        if update_prio_flag:
            exp_rep.update_prios(new_prio, filter)
        return n
