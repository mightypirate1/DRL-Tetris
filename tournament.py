import pickle
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
            "lr" : 5*10**(-4),
            "bar_null_moves" : False, #True,
            }

def run_stuff(controller, option=None, argument=None, manual=True):
    def menu():
        option = ""
        print("Menu!")
        print("-----")
        while option not in ["train", "test", "save_net", "save_all", "save_mem", "save", "load_net", "load_all", "load_mem", "load"]:
            option = input("test, train, save or load? ")
        return option

    # try:
    if True:
        if option is None:
            option = menu()
        elif option == "train":
            if argument is None:
                argument = 10000
            controller.train(n_steps=argument)
        elif option == "test":
            if argument is None:
                argument = 10000
            return controller.test(n_steps=argument, wait=manual)
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
    Code below plays all agents against each other and store the scores!
    '''
    folders = [f.path.split("/")[1] for f in os.scandir("models") if f.is_dir() ]
    _models = [f.split("_")[0] for f in folders]
    _models = ["obi", "poo", "q", "rusty", "ike", "joe"]
    only_last_version = True
    models = {}
    results = {}
    for m in _models:
        models[m] = [f.split("_")[1] for f in folders if m in f]
        models[m].sort(key=lambda y:int(y),reverse=True)
        if only_last_version:
            models[m] = [models[m][0]]

    print("Agents entering competition:")
    for m in models:
        print("{} : {}".format(m,models[m]))

    #Test everyone vs everyone!
    for m1 in models:
        for n1 in models[m1]:
            #Load the 1st agent...
            a1 = m1+"_"+n1
            run_stuff(controller, option="load_net", argument=(0,a1), manual=False)
            for m2 in [m for m in models if m > m1]:
                for n2 in models[m2]:
                    #Load the 2nd agent...
                    a2 = m2+"_"+n2
                    run_stuff(controller, option="load_net", argument=(1,a2), manual=False)
                    #
                    print("{} vs {}!".format(a1,a2))
                    stats = run_stuff(controller, option="test", argument=3000, manual=False)
                    results[a1+"/"+a2] = {a1:sum(stats.scores[controller.agent[0]]),a2:sum(stats.scores[controller.agent[1]])}
                    print("results: {}".format(results[a1+"/"+a2]))
                    with open("results/tournament_results.pkl", 'wb') as f:
                        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
