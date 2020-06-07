import numpy as np

import threads
import aux
import aux.utils as utils
from agents.sventon_agent.sventon_agent_base import sventon_agent_base
import agents.agent_utils as agent_utils
import agents.sventon_agent.sventon_utils as S
from aux.parameter import *

class sventon_agent(sventon_agent_base):
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
        sventon_agent_base.__init__(self, id=id, name="worker{}".format(id), session=session, sandbox=sandbox, settings=settings, mode=mode)

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
        action_eval, state_eval, pieces = self.run_model(model, state_vec, player=p_list)

        #Choose an action . . .
        distribution = self.eval_dist if not training else self.settings["train_distriburion"]
        action_idxs = [None for _ in state_vec]
        for i, (state, _piece, player) in enumerate(zip(state_vec,pieces,p_list)):
            piece, _ = _piece if not self.settings["state_processor_piece_in_statevec"] else (0,None)
            if distribution == "argmax": #for eval-runs
                (r, t), entropy = S.action_argmax(action_eval[i,:,:,piece])
            elif distribution == "pi": #for training
                (r, t), entropy = S.action_distribution(action_eval[i,:,:,piece])
            elif distribution == "pareto_distribution":
                theta = self.theta = self.settings["action_temperature"](self.clock)
                (r, t), entropy = S.action_pareto(action_eval[i,:,:,piece], theta)
            elif distribution == "boltzman_distribution":
                assert False, "boltzman_distribution is deprecated"
                theta = self.theta = self.settings["action_temperature"](self.clock)
                (r, t), entropy = S.action_boltzman(action_eval[i,:,:,piece], theta)
            elif distribution == "adaptive_epsilon":
                epsilon = self.settings["epsilon"](self.clock) * self.avg_trajectory_length**(-1)
                (r, t), entropy = S.action_epsilongreedy(action_eval[i,:,:,piece], epsilon)
            elif distribution == "epsilon":
                epsilon = self.settings["epsilon"](self.clock)
                (r, t), entropy = S.action_epsilongreedy(action_eval[i,:,:,piece], epsilon)
            a_environment = (r,t,piece)
            a_internal = (action_eval[i,r,t,piece] ,S.value_piece(state_eval[i], piece), S.value_mean(state_eval[i]))
            action_idxs[i] = a_environment, a_internal
            # if i == 0 and self.id == 0:
            #     print(a_environment, a_internal)

        #Nearly done! Just need to create the actions...
        actions = [S.make_action(r,t) for (r,t,_), _ in action_idxs]

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
