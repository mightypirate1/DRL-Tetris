import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from game_controller_BOUNCE import game_controller

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
settings = {
            "game_size" : [10,5],
            "pieces" : [i,o], #It's good practice to preserve the ordering "l,j,s,z,i,t,o", even when using only a subset.
            "n_players" : 2,
            "run-id" : "my_first_experiment",
            "action_prob_temperature" : 3.0,
            }

def run_stuff(controller, option=None, argument=None, manual=True):
    def menu():
        option = ""
        print("Menu!")
        print("-----")
        while option not in ["train", "test", "save_net", "save_all", "save_mem", "load"]:
            option = input("test, train, save_[mem|net|all] or load? ")
        return option
    try:
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
        elif option in ["save_net", "save_mem", "save_all"]:
            if argument is None:
                id = int(input("what agent? (int) "))
                name = input("what to call it? ")
            else:
                id, name = argument
            path = "models/"+name
            os.mkdir(path)
            if "net" in option or "all" in option:
                controller.agent[id].save(path, option="weights")
            if "mem" in option or "all" in option:
                controller.agent[id].save(path, option="mem")
        elif option == "load":
            if argument is None:
                id = int(input("what agent? (int) "))
                name = input("what is it called? ")
            else:
                id, name = argument
            path = "models/"+name
            controller.agent[id].load_all(path)
        else:
            print("UNKNOWN OPTION PASSED:{}".format(option))
    except FileExistsError as e:
        print("Name taken!")
    except Exception as e:
        if not type(e) is KeyboardInterrupt:
            print("++++++++<error>++++++++")
            print(e)
            print("+++++++</error>++++++++")
    except KeyboardInterrupt:
        if not manual:
            input("Ctrl-C to abort. Enter to continue.")
    finally:
        if manual:
            option = menu()
            run_stuff(controller, option=option, argument=None, manual=manual)

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
    # run_stuff(controller, option="train", argument=10, manual=False)
    # run_stuff(controller, option="load", argument=(0,"ape_010"), manual=False)
    # run_stuff(controller, option="load", argument=(1,"bacon_010"), manual=False)
    # run_stuff(controller, option="test", argument=10000, manual=True)


    '''
    # # #
    # # # #
    # # # # #
    Code below trains the agents "ape" and "bacon". Optional showing off of
    learned skills, and saving of their models and memory in versioned folders.
    '''
    name0, name1 = "ape", "bacon"
    for i in range(1000):
        print("=======train=======")
        run_stuff(controller, option="train", argument=50000, manual=False)

        # print("=======test========")
        # run_stuff(controller, option="test", argument=1000, manual=False)

        print("=======save========")
        save_mode = "save_net" if t%10>0 else "save_all"
        run_stuff(controller, option=save_mode, argument=(0,"{}_{}".format(name0, str(i).zfill(3))), manual=False)
        run_stuff(controller, option=save_mode, argument=(1,"{}_{}".format(name1, str(i).zfill(3))), manual=False)
