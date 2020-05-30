import time
import numpy as np
from agents.sventon_agent.sventon_agent_trainer_base import sventon_agent_trainer_base

class sventon_agent_ppo_trainer(sventon_agent_trainer_base):
    def do_training(self, sample=None, policy=None):
        minibatch_size, n_epochs, n, update_prio_flag = self.settings["minibatch_size"], self.settings["n_train_epochs_per_update"], self.settings["n_samples_each_update"], False

        #Figure out what policy, model, and experience replay to use...
        if self.settings["single_policy"]:
            policy = "main_net"
        else:
            assert policy is not None, "In multi-policy mode you must specify which model to train!"
            policy = "policy_{}".format(policy)
            if self.scoreboard[policy] > 0.5 + self.settings["winrate_tolerance"]:
                print("Policy \"{}\" did NOT TRAIN, due to too much winning! ({})".format(policy, self.scoreboard[policy]))
                return False
        exp_rep = self.experience_replay_dict[policy]
        model = self.model_dict[policy]
        #If we dont have enough samples yet we quit early...
        if sample is None and len(exp_rep) < n:
            # if not self.settings["run_standalone"]: time.sleep(1) #If we are a separate thread, we can be a little patient here
            # print("DBG: no training")
            return False

        # # #
        #Start!
        # # #

        #1) Get a sample!
        if sample is None: #If no one gave us one, we get one ourselves!
            sample = \
                exp_rep.retrieve_and_clear(compute_stats=True)
        #Unpack a little...
        states, _actions, rewards, dones = sample
        actions, pieces, probs = _actions[:,:,:2], _actions[:,:,2,np.newaxis], _actions[:,:,3,np.newaxis]
        vector_states, visual_states = states
        self.train_lossstats_raw = list()

        #TRAIN!
        for t in range(n_epochs):
            if self.verbose_training: print("[",end='',flush=False); last_print = 0
            last_epoch = t+1 == n_epochs
            perm = np.random.permutation(n)
            for i in range(0,n,minibatch_size):
                stats, _ = model.train(
                                         [vec_s[perm[i:i+minibatch_size]] for vec_s in vector_states],
                                         [vis_s[perm[i:i+minibatch_size]] for vis_s in visual_states],
                                         actions[perm[i:i+minibatch_size]],
                                         pieces[perm[i:i+minibatch_size]],
                                         probs[perm[i:i+minibatch_size]],
                                         rewards[perm[i:i+minibatch_size]],
                                         dones[perm[i:i+minibatch_size]],
                                         lr=self.settings["value_lr"].get_value(self.clock),
                                        )
                self.train_stats_raw.append(stats)
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

        return True