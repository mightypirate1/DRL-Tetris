import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from game_controller_BOUNCE import game_controller
from aux.parameter import *
from agents.nemesis_agent.nemesis_agent import nemesis_agent

'''
This is a very scetchy way of controlling the game flow.
Don't take it as an official runner: this project is still
to be seen as a collection of parts, which you get to
piece together/modify yourself!

That being said, this code implements a way
'''

#Aliasing for pieces, so they are easier to specify :)
(l,j,s,z,i,t,o) = (0,1,2,3,4,5,6)
np.set_printoptions(linewidth=212)

''' Enter stuff here to customize the experiment... '''
total_steps = 6*10**7#Environment steps
n_save_points = 25

settings = {
            "game_size" : [22,10],
            "pieces" : [l,j,s,z,i,t,o], #It's good practice to preserve the ordering "l,j,s,z,i,t,o", even when using only a subset.
            "time_elapsed_each_action" : 200,
            "n_players" : 2,
            "run-id" : "my_first_experiment",
            "bar_null_moves" : False, #True,
            # "agent" : nemesis_agent,
            "use_curiosity" : False,

            ''' SETTINGS FOR before yolo,swag were:
            "time_elapsed_each_action" : 100,
            "game_size" : [10,5],
            "pieces" : [i,o],
            total_steps = 24*10**5
            "time_to_reference_update" : 5,
            "value_lr" : linear_parameter(5*10**-6, final_val=5*10**-6, time_horizon=(total_steps/2)),
            "prioritized_replay_beta" : linear_parameter(0.7,final_val=1,time_horizon=(total_steps/2)),
            "time_to_training" : 10**3,
            "n_samples_each_update" : 2**10,
            "gamma_extrinsic" : 0.97,
            "gamma_intrinsic" : 0.90,
            etc....
            '''

            "value_head_n_hidden" : 5,
            "time_to_reference_update" : 3,
            "value_head_hidden_size" : 1024,
            "time_to_training" : 10**3,
            "n_samples_each_update" : 2**10,
            "value_lr" : linear_parameter(5*10**-6, final_val=5*10**-6, time_horizon=0.9*(total_steps/2)),
            "prioritized_replay_beta" : linear_parameter(0.7,final_val=1,time_horizon=0.9*(total_steps/2)),
            }

def run_stuff(controller, option=None, argument=None, manual=True):
    def menu():
        option = ""
        print("Menu!")
        print("-----")
        while option not in ["train", "test", "save_net", "save_all", "save_mem", "save", "load_net", "load_all", "load_mem", "load"]:
            option = input("test, train, save or load? ")
        return option

    if True:
    # try:
        if option is None:
            option = menu()
        elif option == "train":
            if argument is None:
                argument = 10000
            controller.train(n_steps=argument)
        elif option == "test":
            if argument is None:
                argument = 10000
            controller.test(n_steps=argument, wait=manual)
        elif option in ["save_net", "save_mem", "save_all", "save"]:
            if argument is None:
                id = int(input("what agent? (int) "))
                name = input("what to call it? ")
            else:
                id, name = argument
            path = "models/"+name
            os.mkdir(path)
            if "mem" in option or option in ["save", "save_all"]:
                controller.agent[id].save(path, option="mem")
            if "net" in option or option in ["save", "save_all"]:
                controller.agent[id].save(path, option="weights")

        elif option in ["load_net", "load_mem", "load_all", "load"]:
            if argument is None:
                id = int(input("what agent? (int) "))
                name = input("what is it called? ")
            else:
                id, name = argument
            path = "models/"+name
            if "net" in option or option in ["load", "load_all"]:
                controller.agent[id].load(path,option="weights")
            if "mem" in option or option in ["load", "load_all"]:
                controller.agent[id].load(path, option="mem")
        else:
            print("UNKNOWN OPTION PASSED:{}".format(option))
    # except FileExistsError as e:
    #     print("Name taken!")
    # except Exception as e:
    #     if not type(e) is KeyboardInterrupt:
    #         print("++++++++<error>++++++++")
    #         print(e)
    #         print("+++++++</error>++++++++")
    # except KeyboardInterrupt:
    #     if not manual:
    #         input("Ctrl-C to abort. Enter to continue.")
    # finally:
    #     if manual:
    #         option = menu()
    #         run_stuff(controller, option=option, argument=None, manual=manual)

## Main code :)
with tf.Session() as session:
    controller = game_controller(settings=settings, session=session)
    '''
    # # #
    # # # #
    # # # # #
    This line of code can be uncommented to go into manual mode. Then you get a
    menu asking what to do. You can perhaps load a couple of models you stored,
    and watch them play!
    '''
    # run_stuff(controller)

    '''
    # # #
    # # # #
    # # # # #
    This is a sample script that you can use so you don't have to go through the
    menus each time...
    '''
    # run_stuff(controller, option="save_net", argument=(0,"lool"), manual=False)
    # run_stuff(controller, option="load_net", argument=(0,"coke_003"), manual=False)
    # run_stuff(controller, option="load_net", argument=(1,"doe_003"), manual=False)
    # # run_stuff(controller, option="train", argument=10, manual=False)
    # run_stuff(controller, option="test", argument=10000, manual=True)
    # exit()

    '''
    # # #
    # # # #
    # # # # #
    Code below trains the agents "ape" and "bacon". Optional showing off of
    learned skills, and saving of their models and memory in versioned folders.
    '''
    name0, name1 = "yolo", "zwag"
    for i in range(n_save_points):
        print("=======train=======")
        run_stuff(controller, option="train", argument=int(total_steps/n_save_points), manual=False)
        # print("=======test========")
        # run_stuff(controller, option="test", argument=1000, manual=False)
        print("=======save========")
        save_mode = "save_net" if t%5==0 else "save_all"
        run_stuff(controller, option=save_mode, argument=(0,"{}_{}".format(name0, str(i).zfill(3))), manual=False)
        run_stuff(controller, option=save_mode, argument=(1,"{}_{}".format(name1, str(i).zfill(3))), manual=False)
