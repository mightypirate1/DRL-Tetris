import time
import numpy as np
import threads
import aux
import aux.utils as utils
from agents.pg_vector_agent.pg_vector_agent_base import pg_vector_agent_base
import agents.agent_utils as agent_utils
from agents.networks import pg_net
from agents.agent_utils.pg_experience_replay import pg_experience_replay
from aux.parameter import *
from agents.agent_utils import state_fcns


class pg_vector_agent_trainer(pg_vector_agent_base):
    def __init__(
                 self,
                 id=0,                      # What's this trainers name?
                 session=None,              # The session to operate in
                 sandbox=None,              # Sandbox to play in!
                 mode=threads.ACTIVE,       # What's our role?
                 settings=None,             # Settings-dict passed down from the ancestors
                ):

        #Some general variable initialization etc...
        pg_vector_agent_base.__init__(self, id=id, name="trainer{}".format(id), session=session, sandbox=sandbox, settings=settings, mode=mode)
        self.verbose_training = self.settings["run_standalone"]
        self.train_stats_raw = list()
        self.n_train_steps = 0

        #Bucket of data
        self.experience_replay = pg_experience_replay(
                                                      n_actions=self.settings["n_actions"],
                                                      max_size=self.settings["experience_replay_size"],
                                                      state_size=self.state_size,
                                                     )

        #Models
        self.extrinsic_model = pg_net(
                                      self.id,
                                      "prio_vnet",
                                      self.state_size,
                                      session,
                                      settings=self.settings,
                                     )
        self.model_dict = {
                            "extrinsic_model"           : self.extrinsic_model,
                            "default"                   : self.extrinsic_model,
                          }

    #What if someone just sends us some experiences?! :D
    def receive_data(self, data_list):
        if len(data_list) == 0: return
        if type(data_list[0]) is list:
            data = list()
            for d in data_list:
                data += d
        else: data = data_list

        tot = 0
        n = 0
        for processed_trajectory in data:
                n += 1
                self.experience_replay.add_samples(processed_trajectory)
                tot += processed_trajectory[1].shape[0] #this is the number of actions, which is the length of the trajectory before it was processed
        self.clock += tot
        avg = tot/n if n>0 else 0
        return tot, avg

    def unpack_sample(self, sample):
        #Put it in arrays
        n = self.settings["n_samples_each_update"]
        states        = np.zeros((n,*self.state_size ))
        target_values = np.zeros((n,1 ))
        is_weights    = np.zeros((n,1 ))
        for i,s in enumerate(sample):
            s,p,tv, isw        = s.get_data()
            states[i,:]        = self.state_from_perspective(s, p)
            target_values[i,:] = tv
            is_weights[i,:]    = isw
        return states, target_values, is_weights

    def do_training(self, sample=None):
        if sample is None and len(self.experience_replay) < self.settings["n_samples_each_update"]:
            if not self.settings["run_standalone"]: time.sleep(1) #If we are a separate thread, we can be a little patient here
            return False
        #Start!
        self.train_stats_raw = list()
        self.log.debug("trainer[{}] doing training".format(self.id))

        #Some values:
        minibatch_size = self.settings["minibatch_size"]
        n_epochs       = self.settings["n_train_epochs_per_update"]
        n = self.settings["n_samples_each_update"]

        #Get a sample!
        if sample is None: #If no one gave us one, we get one ourselves!
            sample = self.experience_replay.get_all_samples()
        states, masks, actions, old_probs, target_values, advantages = sample

        #TRAIN!
        for t in range(n_epochs):
            last_epoch = t+1 == n_epochs
            if self.verbose_training: print("[",end='',flush=False); last_print = 0
            perm = np.random.permutation(n) if not last_epoch else np.arange(n)
            for i in range(0,n,minibatch_size):
                _,cl, el, vl, tl = self.extrinsic_model.train(
                                                              states[perm[i:i+minibatch_size]],
                                                              masks[perm[i:i+minibatch_size]],
                                                              actions[perm[i:i+minibatch_size]],
                                                              old_probs[perm[i:i+minibatch_size]],
                                                              target_values[perm[i:i+minibatch_size]],
                                                              advantages[perm[i:i+minibatch_size]],
                                                              self.settings["clipping_parameter"].get_value(self.clock),
                                                              self.settings["value_lr"].get_value(self.clock)
                                                             )
                self.train_stats_raw.append((cl,el,vl,tl))
                if self.verbose_training and (i-last_print)/n > 0.02: print("-",end='',flush=False); last_print = i
            if self.verbose_training: print("]",flush=False)

        #Count training
        self.n_train_steps += 1

        #Throw away samples since this is an on-policy method!
        self.experience_replay.clear()
        return True #In case someone sent a particular sample for us to train on, they might want to know the new prios etc..

    def output_stats(self):
        lossclip, lossentropy, lossval, losstot = 0, 0, 0, 0
        n = len(self.train_stats_raw)
        for ret_vals in self.train_stats_raw:
            _lossclip, _lossentropy, _lossval, _losstot = ret_vals
            lossclip    += _lossclip / n
            lossentropy += _lossentropy / n
            lossval     += _lossval / n
            losstot     += _losstot / n
        return {"Clip loss" : lossclip, "Entropy loss" : lossentropy, "Value loss" : lossval, "Total loss" : losstot}

    def export_weights(self):
        return self.n_train_steps, self.model_dict["default"].get_weights(self.model_dict["default"].all_vars)
        #return n_steps, (m_weight_dict, m_name)
