import numpy as np
import time
import threads
import tools
import tools.utils as utils
from agents.sherlock_agent.sherlock_agent_base import sherlock_agent_base
import agents.agent_utils as agent_utils
import agents.sherlock_agent.sherlock_utils as S
from tools.parameter import *

class sherlock_agent(sherlock_agent_base):
    def __init__(
                 self,
                 n_envs,                    # How many envs in the vector-env?
                 n_workers=1,               # How many workers run in parallel? If you don't know, guess it's just 1
                 id=0,                      # What's this workers name?
                 session=None,              # The session to operate in
                 sandbox=None,              # Sandbox to play in!
                 mode=threads.STANDALONE,   # What's our role?
                 settings=None,             # Settings-dict passed down from the ancestors
                 init_weights=None,
                 init_clock=0,
                ):

        #The base class provides basic functionality, and provides us with types to use! (This is how we are both ppo and q)
        sherlock_agent_base.__init__(self, id=id, name="worker{}".format(id), session=session, sandbox=sandbox, settings=settings, mode=mode)
        self.DBG = {"delta_zero" : 0, "delta_sum_zero" : 0, 'p_sum_zero' : 0}
        #Some general variable initialization etc...

        #Helper variables
        self.env_idxs = [i for i in range(n_envs)]
        self.n_envs = n_envs
        self.n_workers = n_workers
        self.n_experiences,self.send_count, self.send_length = 0, 0, 0

        #In any mode, we need a place to store transitions!
        self.current_trajectory = [self.trajectory_type() for _ in range(self.n_envs if self.settings["single_policy"] else 2*self.n_envs)]
        self.stored_trajectories = list()
        self.avg_trajectory_length = 12 #tau is initialized to something...
        self.action_entropy = 0
        self.theta = 0

        #If we do our own training, we prepare for that
        if self.mode is threads.STANDALONE:
            #Create a trainer, and link their neural-net and experience-replay to us
            self.trainer = self.trainer_type(id="trainer_{}".format(self.id),settings=settings, session=session, sandbox=sandbox, mode=threads.PASSIVE)
            self.model_dict = self.trainer.model_dict
            #STANDALONE agents have to keep track of their own training habits!
            self.time_to_training = max(self.settings['time_to_training'],self.settings['n_samples_each_update'])

        if self.mode is threads.WORKER: #If we are a WORKER, we bring our own equipment
            #Create models
            self.model_dict = {}
            models = ["main_net"] if self.settings["single_policy"] else ["policy_0", "policy_1"]
            for model in models:
                m = self.network_type(
                                        self.id,
                                        model,
                                        self.state_size,
                                        self.model_output_shape, #Output_shape
                                        session,
                                        worker_only=True,
                                        settings=self.settings,
                                     )
                self.model_dict[model] = m

        if init_weights is not None:
            print("Agent{} initialized from weights: {} and clock: {}".format(self.id, init_weights, init_clock))
            self.update_clock(init_clock)
            self.load_weights(init_weights,init_weights)

    # # # # #
    # Agent interface fcns
    # # #

    #
    ###
    #####
    def get_action(self, state_vec, player=None, random_action=False, training=False, verbose=False):
        #Get hypothetical future states and turn them into vectors!
        p_list = utils.parse_arg(player, self.player_idxs)
        #Set up some stuff that depends on what type of training we do...
        if self.settings["single_policy"]:
            model = self.model_dict["main_net"]
        else:
            assert False, "not tested yet. comment out this line if you brave"
            assert p_list[0] == p_list[-1], "{} ::: In dual-policy mode we require queries to be for one policy at a time... (for speed)".format(p_list)
            model = self.model_dict["policy_{}".format(p_list[0])]

        #Run model!
        phi_eval, state_eval, pieces = self.run_model(model, state_vec, player=p_list)

        #Be Sherlock!
        all_actions = [self.sandbox.get_actions(s, player=p) for s,p in zip(state_vec, p_list)]

        all_deltas, delta_sums = S.generate_deltas(state_vec, p_list, self.sandbox)
        # all_deltas = S.pad_and_concat([S.deltas(s, p, self.sandbox)[np.newaxis,:] for s,p in zip(state_vec, p_list)])
        # delta_sums = np.sum(all_deltas, axis=-1, keepdims=True)
        if not self.settings["separate_piece_values"]:
            pieces = np.zeros_like(pieces)
        phi_p = np.concatenate([phi[np.newaxis,:,:,p,np.newaxis] for phi,p in zip(phi_eval,pieces)], axis=0)
        p_unreduced = all_deltas * phi_p
        p_unnormalized = np.sum(p_unreduced, axis=(1,2))
        p_sum = p_unnormalized.sum(axis=-1,keepdims=True)

        if (p_sum == 0).any(): # avoid a corner case :)
            self.DBG["p_sum_zero"] += 1
            p_sum[np.where(p_sum==0)] = 1.0
        probablilities = p_unnormalized / p_unnormalized.sum(axis=1, keepdims=True)

        #Be cautious?
        # print("-------dbg--------")
        # print(all_deltas[0,:,:,3])
        # print(all_actions[0][3])

        #Choose an action . . .
        distribution = self.eval_dist if not training else self.settings["train_distribution"]
        action_idxs = [None for _ in state_vec]
        for i, (state, piece, player, prob, delta, delta_sum, eval) in enumerate(zip(state_vec, pieces,p_list, probablilities, all_deltas, delta_sums, state_eval)):
            if distribution == "argmax": #for eval-runs
                a_idx, entropy = S.action_argmax(prob)
            elif distribution == "pi": #for training
                a_idx, entropy = S.action_distribution(prob)
            elif distribution == "pareto_distribution":
                theta = self.theta = self.settings["action_temperature"](self.clock)
                a_idx, entropy = S.action_pareto(prob, theta)
            elif distribution == "boltzman_distribution":
                assert False, "boltzman_distribution is deprecated"
                theta = self.theta = self.settings["action_temperature"](self.clock)
                a_idx, entropy = S.action_boltzman(prob, theta)
            elif distribution == "adaptive_epsilon":
                epsilon = self.settings["epsilon"](self.clock) * self.avg_trajectory_length**(-1)
                a_idx, entropy = S.action_epsilongreedy(prob, epsilon)
            elif distribution == "epsilon":
                epsilon = self.settings["epsilon"](self.clock)
                a_idx, entropy = S.action_epsilongreedy(prob, epsilon)
            # a = a_idx, piece, pi(a), v_piece(s), v_mean(s), delta(s,a), delta_sum(s)
            a = (
                 a_idx[np.newaxis, np.newaxis],
                 piece[np.newaxis, np.newaxis],
                 prob[a_idx,np.newaxis, np.newaxis],
                 S.value_piece(eval, piece)[np.newaxis, np.newaxis],
                 S.value_mean(eval)[np.newaxis, np.newaxis],
                 delta[np.newaxis,:,:,a_idx,np.newaxis],
                 delta_sum[np.newaxis,:]
                 )

            if delta.sum() == 0:
                self.DBG["delta_zero"] += 1
            if delta_sum.sum() == 0:
                self.DBG["delta_sum_zero"] += 1
            if prob[a_idx].sum() == 0:
                self.DBG["delta_sum_zero"] += 1
            action_idxs[i] = a
        #Nearly done! Just need to create the actions...
        actions = [a[idx[0].squeeze()] for a, idx in zip(all_actions,action_idxs)]
        print(self.DBG)
        #Keep the clock going...
        if training:
            self.clock += self.n_envs * self.n_workers
        return action_idxs, actions

    #
    ###
    #####
    def ready_for_new_round(self, training=False, env=None):
        e_idxs, _ = utils.parse_arg(env, self.env_idxs, indices=True)
        if not self.settings["single_policy"]:
            e_idxs += [e_idx + self.n_envs for e_idx in e_idxs]
        for e in e_idxs:
            a = self.settings["tau_learning_rate"]
            if len(self.current_trajectory[e]) > 0 or training is False:
                self.avg_trajectory_length = (1-a) * self.avg_trajectory_length + a*len(self.current_trajectory[e])

        # Preprocess the trajectories specifiel to prepare them for training
        for e in e_idxs:
            if training and len(self.current_trajectory[e]) > 0:
                t = self.current_trajectory[e]
                if self.settings["workers_do_processing"]:
                    model = self.model_dict["main_net"] if self.settings["single_policy"] else self.model_dict["policy_{}".format(int(e>=self.n_envs))]
                    data = t.process_trajectory(
                                                self.model_runner(model),
                                                self.unpack,
                                                compute_advantages=self.settings["workers_computes_advantages"],
                                                gae_lambda=self.settings["gae_lambda"],
                                                reward_shaper=None,
                                                gamma_discount=self.gamma,
                                                augment=self.settings["augment_data"]
                                               )
                else:
                    data = t
                metadata = {
                            "policy"    : int(e>=self.n_envs),
                            "winner"    : t.winner,
                            "length"    : len(t),
                            "worker"    : self.id,
                            "packet_id" : self.send_count,
                            }
                self.stored_trajectories.append((metadata,data))
                #Increment some counters to guide what we do
                self.send_count += 1
                self.send_length += len(t)
                if self.mode is threads.STANDALONE:
                    self.time_to_training  -= len(self.current_trajectory[e])
            #Clear trajectory
            self.current_trajectory[e] = self.trajectory_type()

        #Standalone agents have to keep track of their training habits!
        if training and self.mode is threads.STANDALONE:
            if self.time_to_training < 1:
                self.trainer.receive_data(self.transfer_data())
                self.trainer.do_training()
                self.time_to_training = self.settings['time_to_training']

    #
    ###
    #####
    def store_experience(self, experience, env=None):
        env_list = utils.parse_arg(env, self.env_idxs)
        #Turn a list of experience ingredients into one list of experiences:
        es = utils.merge_lists(*experience)
        assert len(env_list) == len(es), "WTF!!!! {} != {}".format(len(env_list), len(es))
        for i,e in zip(env_list, es):
            if e[0] is None:
                continue
            if self.settings["single_policy"]:
                self.current_trajectory[i].add(e)
            if not self.settings["single_policy"]:
                #Player1's trajectories strored first (n_envs many) and then player2's:
                self.current_trajectory[i + e[4]*self.n_envs].add(e)
            self.n_experiences += 1
        self.log.debug("agent[{}] appends experience {} to its trajectory-buffer".format(self.id, experience))

    def transfer_data(self, keep_data=False):
        #This function gives away the data gathered
        ret = self.stored_trajectories
        if not keep_data:
            self.stored_trajectories = list()
        return ret
