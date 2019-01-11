import threads
import aux
import aux.utils as utils
import agent.agent_utils as agent_utils
from agents.networks import value_net
from agents.agent_utils.experience_replay import experience_replay
from aux.parameter import *

class vector_agent_trainer:
    def __init__(self, blabla):
        self.experience_replay = agent_utils.experience_replay(self.settings["experience_replay_size"], prioritized=self.settings["prioritized_experience_replay"])
        self.time_to_reference_update = 0#self.settings['time_to_reference_update']
        self.reference_extrinsic_model = value_net(self.id, "reference_extrinsic", self.state_size, session, settings=self.settings, on_cpu=self.settings["worker_net_on_cpu"])

    '''STRAIGHT FROM THE OTHER AGENT  THING,,,'''
    def do_training(self):
        print("I PRETEND TO TRAIN!")
        return
        assert False
        self.log.debug("agent[{}] doing training".format(self.id))
        print("agent{} training...".format(self.id))
        samples, _, time_stamps, is_weights, return_idxs =\
                self.experience_replay.get_random_sample(
                                                         self.settings["n_samples_each_update"],
                                                         alpha=self.settings["prioritized_replay_alpha"].get_value(self.n_train_steps),
                                                         beta=self.settings["prioritized_replay_beta"].get_value(self.n_train_steps),
                                                         )
        #Unpack data (This feels inefficient... think thins through some day)
        states, actions, rewards, next_states, dones, nemeses = [[None]*self.settings["n_samples_each_update"] for x in range(6)]
        for i,s in enumerate(samples):
            states[i], actions[i], rewards[i], next_states[i], dones[i], nemeses[i] = s
        player_list = list(zip(nemeses, [self.id for x in range(len(samples))]))
        #Get data into numpy-arrays...
        state_vec = self.states_to_vectors(states)
        done_vec = np.concatenate(dones)[:,np.newaxis]
        extrinsic_reward_vec = np.concatenate(rewards)[:,np.newaxis]
        next_extrinsic_value_vec_ref, _ = self.run_model(self.reference_extrinsic_model, next_states, player_list=player_list)
        target_extrinsic_value_vec = extrinsic_reward_vec + (1-done_vec.astype(np.int)) * -self.settings["gamma_extrinsic"] * next_extrinsic_value_vec_ref
        if self.settings["use_curiosity"]:
            intrinsic_reward_vec = self.settings["curiosity_reward_multiplier"] * self.run_model(self.curiosity_network, state_vec, player_list=[self.id, self.nemesis])
            next_intrinsic_value_vec_ref, _ = self.run_model(self.reference_intrinsic_model, next_states, player_list=players)
            target_intrinsic_value_vec = intrinsic_reward_vec + (1-done_vec.astype(np.int)) * self.settings["gamma_intrinsic"] * next_intrinsic_value_vec_ref
        is_weights = np.array(is_weights)
        minibatch_size = self.settings["minibatch_size"]
        #TRAIN!
        for t in range(self.settings["n_train_epochs_per_update"]):
            perm = np.random.permutation(self.settings["n_train_epochs_per_update"])
            for i in range(0,self.settings["n_samples_each_update"],minibatch_size):
                self.extrinsic_model.train(state_vec[perm[i:i+minibatch_size]],target_extrinsic_value_vec[perm[i:i+minibatch_size]], weights=is_weights[perm[i:i+minibatch_size]], lr=self.settings["value_lr"].get_value(self.n_train_steps))
                if self.settings["use_curiosity"]:
                    self.intrinsic_model.train(state_vec[perm[i:i+minibatch_size]],target_intrinsic_value_vec[perm[i:i+minibatch_size]], weights=is_weights[perm[i:i+minibatch_size]], lr=self.settings["value_lr"].get_value(self.n_train_steps))
        if self.time_to_reference_update == 0:
            print("Updating agent{} reference model!".format(self.id))
            if self.settings["alternating_models"]:
                #alternate extrinsic
                tmp = self.reference_extrinsic_model
                self.reference_extrinsic_model = self.extrinsic_model
                self.extrinsic_model = tmp
                if self.settings["use_curiosity"]:
                    #alternate intrinsic
                    tmp = self.reference_intrinsic_model
                    self.reference_intrinsic_model = self.intrinsic_model
                    self.intrinsic_model = tmp
            else:
                weights = self.extrinsic_model.get_weights(self.extrinsic_model.trainable_vars)
                self.reference_extrinsic_model.set_weights(self.reference_extrinsic_model.trainable_vars,weights)
                if self.settings["use_curiosity"]:
                    weights = self.intrinsic_model.get_weights(self.intrinsic_model.trainable_vars)
                    self.reference_intrinsic_model.set_weights(self.reference_intrinsic_model.trainable_vars,weights)
            self.time_to_reference_update = self.settings["time_to_reference_update"]
        else:
            self.time_to_reference_update -= 1

        # Get new surprise values for the samples, then store them back into experience replay!
        extrinsic_value_vec, _ = self.run_model(self.extrinsic_model, state_vec, player_list=[self.id, self.nemesis])
        extrinsic_td_errors = (extrinsic_reward_vec - self.settings["gamma_extrinsic"] * next_extrinsic_value_vec_ref - extrinsic_value_vec).reshape(-1).tolist()
        surprise_factors = np.abs(extrinsic_td_errors)
        if self.settings["use_curiosity"]:
            intrinsic_value_vec, _ = self.run_model(self.intrinsic_model, state_vec, player_list=[self.id, self.nemesis])
            intrinsic_td_errors = (intrinsic_reward_vec + self.settings["gamma_intrinsic"] * next_intrinsic_value_vec_ref - intrinsic_value_vec).reshape(-1).tolist()
            surprise_factors += self.settings["curiosity_amount"].get_value(self.n_train_steps) *np.abs(intrinsic_td_errors)
        self.experience_replay.add_samples( samples, surprise_factors, filter=return_idxs, time_stamps=time_stamps)
        '''---'''

        print("---")
        print("TAU:{}".format(self.avg_trajectory_length))
        print("EPSILON:{}".format(self.settings["epsilon"].get_value(self.n_train_steps)/self.avg_trajectory_length))
        print("curiosity_amount:{}".format(self.settings["curiosity_amount"].get_value(self.n_train_steps)))
        print("value_lr:{}".format(self.settings["value_lr"].get_value(self.n_train_steps)))
        print("---")

        #Reset delay
        self.time_to_training = self.settings["time_to_training"]
