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
            "pieces" : [i,o],
            "n_players" : 2,
            "run-id" : "my_first_experiment",
            "action_prob_temperature" : 3.0,
            }

def run_stuff(controller, option=None, argument=None, manual=True):
    def menu():
        option = ""
        print("Menu!")
        print("-----")
        while option not in ["train", "test", "save", "load"]:
            option = input("test, train, save or load? ")
        return option
    try:
        if option is None:
            option = menu()
        if option == "train":
            if argument is None:
                argument = 10000
            controller.train(n_steps=argument)
        if option == "test":
            if argument is None:
                argument = 100
            controller.test(n_steps=argument)
        if option == "save":
            if argument is None:
                id = int(input("what agent? (int) "))
                name = input("what to call it? ")
            else:
                id, name = argument
            path = "models/"+name
            os.mkdir(path)
            controller.agent[id].save_all(path)
        if option == "load":
            if argument is None:
                id = int(input("what agent? (int) "))
                name = input("what is it called? ")
            else:
                id, name = argument
            path = "models/"+name
            controller.agent[id].load_all(path)
    except FileExistsError as e:
        print("Name taken!")
    except Exception as e:
        print("++++++++<error>++++++++")
        print(e)
        print("+++++++</error>++++++++")
    finally:
        if manual:
            run_stuff(controller)

## Main code :)
with tf.Session() as session:
    controller = game_controller(settings=settings, session=session)
    '''
    This line of code can be uncommented to go into manual mode.
    Then you get a menu asking what to do. You can perhaps load a couple of
    models you stored, and watch them play!
    '''
    # run_stuff(controller)

    '''
    Code below does some training of agents "ape" and "bacon".
    They train some, they train their models, then they test their models, and
    then they save their models (and memory) in versioned folders.
    '''
    for i in range(1000):
        print("=======train=======")
        run_stuff(controller, option="train", argument=10000, manual=False)
        print("=======test========")
        run_stuff(controller, option="test", argument=100, manual=False)
        run_stuff(controller, option="save", argument=(0,"ape_{}".format(str(i).zfill(3))), manual=False)
        run_stuff(controller, option="save", argument=(1,"bacon_{}".format(str(i).zfill(3))), manual=False)
