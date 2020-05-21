# DRL-Tetris
This repository is three things:

1. It is the open-source multiplayer tetris game [SpeedBlocks] turned into a reinforcement learning (RL) environment with a complete python front-end. The Environment is highly customizable: game-field size, block types used, action type etc. are all easily changed. It is written to function well with RL at large scale by running arbitrary numbers of tetris-games in parallel, in a simple-to-use manner.

2. A multi-processing framework for running multiple workers gathering trajectories for a trainer thread. The framework is flexible enough to facilitate many different RL algorithms. If you match the format of the template-agent provided, your algorithm should work right away with the framework.

3. A small but growing family of RL-algorithms that learns to play two-player tetris through self-play:

    **SIXten** learns the value-function thru a k-step estimation scheme, utilizing the world-model of the environment and a prioritized distributed experience replay (modelled on Schaul et al.). Toghether with the multiprocessing framework described above it's similar to Ape-X (https://arxiv.org/abs/1803.00933) but the RL algorithm itself is different.

    **SVENton** is a double[1] duelling[2] k-step DQN-agent with a novel Convolutional Neuro-Keyboard interfacing it with the environment (1: https://arxiv.org/abs/1509.06461 2: https://arxiv.org/abs/1511.06581). It too utilizes the distributed prioritized experience replay, and the multi-processing framework. It is highly experimental, but it's included so SIXten doesn't get lonely.

These components may be separated into different repositories at some time in the future, but for now they are one.

> NOTE: The master-branch contains some features that are experiental. If there are issues, revert to the stable branch (https://github.com/mightypirate1/DRL-Tetris/tree/stable)

> NOTE: The quality of code has changed as I learned. Please be lenient when looking at the remnants of old code. There will come a day when it's fixed, let's hope!

## Installation:
* Pull the repository.
* Install dependencies (see "Dependencies").
* Build the backend module (see "Build backend").

> NOTE: It has come to my attention that this code-base does not easily run on mac or windows. My apologies, I might fix that one day. Feel free to contribute :)

Once these steps are done, you should be able to use the environment, and to train your own agents (see "Usage").

#### Dependencies:
The versions specified are the version used on the test system.

- Python3 (3.6.3)
- CMake (3.9.1)

Python modules:
- NumPy (1.16)
- Tensorflow (1.12.0)
- SciPy (1.2.0)
- Docopt (0.6.2)

On Ubuntu, apt and pip3 solves the dependencies easily:
```
apt install cmake python3-dev python3-pip
pip3 install docopt scipy numpy tensorflow
```
> Replace tensorflow with tensorflow-gpu for GPU support. This might require some work, but the official documentation should help: [tensorflow].

If the installation of any dependency fails, we refer to their documentation.

If you are not on Ubuntu, install the dependencies as you would on you system and proceed to the next step.

#### Build backend:
To build the package, we used CMake and make:
```
cd path/to/DRL-tetris/environment/game_backend/source
cmake .
make
```

## Usage:
To start training, we recommend starting off from the example in experiments/sixten_base.py

To run the example project using 32 environments per worker thread, and 3 worker threads (+1 trainer thread), for 10M steps, run
```
python3 thread_train.py experiments/sixten_base.py --n 32 --m 3 --steps 10000000
```
periodically during training, weights are saved to models/project_name/weightsNNN.w. Additionally, backups are made to models/project_name/weightsLATEST.w, and the final version is saved to models/project_name/weightsFINAL.w.

To test these weights out against themselves
```
python3 eval.py path/to/weightfile.w
```
or against other weights
```
python3 eval.py path/to/weightfile1.w path/to/weightfile2.w (...)
```
Settings are saved along with the weights so that it is normally possible to make bots made with different settings, neural-nets etc. play each other. As long as the game_size setting is the same across projects, they should be compatible! See "Customization" for more details.

#### Demo weights:
The project ships with some pre-trained weights as a demo. When in the DRL-Tetris folder, try for instance
```
python3 eval.py models/demo_weights/SIXten/weightsDEMO1.w
```
to watch SIXten play.

## Customization:
The entire repository uses a settings-dictionary (the default values of which are found in aux/settings.py). To customize the environment, the agent, or the training procedure, create dictionary with settings that you pass to the relevant objects on creation. For examples of how to create such a dictionary, see "thread_train.py", and for how to pass it to the environment constructor, see "threads/worker_thread.py".

For minor customizations, you can just edit the settings-dictionary in thread_train.py.
To change the size of the field used, just find the game_field entry and put a new value there. Any option that is in aux/settings.py can be overridden this way.

#### Pieces:
What pieces are being used is specified in the settngs-dictionary's field "pieces". It contains a list of any subset of {0,1,2,3,4,5,6}. [0,1,2,3,4,5,6] means the full set is used. The numbers correspond to the different pieces via the aliasing (L,J,S,Z,I,T,O) <~> (0,1,2,3,4,5,6). If those letters confuse you, you might want to check out https://tetris.fandom.com/wiki/Tetromino

The pre-defined settings on the master branch plays with only the O- and the L-piece to speed up training (pieces set to [0,6]).

> Quickest way to enable all pieces is to comment out the line in "thread_train.py" that reduces it to O and L:
> ```
> # "pieces" : [0,6],
> ```
> "pieces" will get the default value instead, which means all pieces are used.

#### Advanced customization:

If you wish to customizations that are not obvious how to do, just contact me and I will produce the documentation needed asap. To write your own agent and/or customize the training procedure, you will have to write code. Probably the best way to get into the code is to look at the function "thread_code" in threads/worker_thread.py where the main training loop is located.

## On the horizon:

#### Network play:
So far no official client exists for playing against the agents you train. Coming soon is a closer integration of the environmnet backend and the game itself. This will allow for an evaluation mode where an agent plays versus a human player online in the same way that two human play against each other. Stay tuned!

#### API-documentation:
The environment documentation is next on the todo-list. For now I will say that the functionality is similar conceptually to the OpenAI gym environments, and should be quite understandable from reading the code (check out the function "thread_code" in threads/worker_thread.py). Having said that, if you would like to see documentation happen faster, or if you have any question regarding this, contact me and I will happily answer.

#### Standardized environment configurations:
A few standard configurations will be decided on and made official so that they are easily and reliable recreated. Basically replacing
```
settings = {...}
env = tetris_environment(...,...,...,...,settings=settings)
```
with
```
env = environment.make("FullSize-v0")
```

## Contribute!
If you want to get involved in this project and want to know what needs to be done, feel free to contact me and I will be happy to discuss!

If you find a bug, have concrete ideas for improvement, think something is lacking, or have any other suggestions, I will be glad to hear about it :-)

## Contact:
yfflan at gmail dot com

[SpeedBlocks]: <https://github.com/kroyee/SpeedBlocks>
[tensorflow]: <https://www.tensorflow.org/install/>
