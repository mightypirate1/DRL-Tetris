# DRL-Tetris
This repository is two things:
1. It is the open-source multiplayer tetris game [SpeedBlocks] turned into a reinforcement learning (RL) environment with a simple python front-end. The Environment is highly customizable (game-field size, block types used, action type etc. are all easily changed) and is written to function well with RL at large scale.

2. A multi-process RL algorithm that learns to play two-player tetris through self-play. The multi-processing framework is similar to that of Ape-X (https://arxiv.org/abs/1803.00933) but the RL algorithm itself is different.

These two components may be separated into different repositories at some time in the future, but for now they are one.

## Installation:
* Pull the repository.
* Install dependencies (see "Dependencies").
* Build the backend module (see "Build backend").

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

#### Build backend:
To build the package, we used CMake and make:
```
cd path/to/DRL-tetris/environment/game_backend/source
cmake .
make
```

## Usage:
To start training, we recommend starting off from the example in thread_train.py

To run the example project using 32 environments per worker thread, and 3 worker threads (+1 trainer thread), for 10M steps, run
```
python3 thread_train.py --n 32 --m 3 --steps 10000000
```

periodically during training, weights are saved to models/project_name/weightsNNN.w

To test these weights out against itself, run
```
python3 eval.py project_name NNN project_name NNN
```
As you might guess, you could make any weight version of any project play against any other, given that they both use the same environment settings.

## Customization:
The entire repository uses a settings-dictionary (the default values of which are found in aux/settings.py). To customize the environment, the agent, or the training procedure, create dictionary with settings that you pass to the relevant objects on creation.

For minor customizations, you can just edit the settings-dictionary in thread_train.py.
To change the size of the field used, just find the game_field entry and put a new value there. Any option that is in aux/settings.py can be overridden this way.

If you wish to customizations that are not obvious how to do, just contact me and I will produce the documentation needed asap. To write your own agent and/or customize the training procedure, you will have to write code. Probably the best way to get into the code is to look at the function "thread_code" in threads/worker_thread.py where the main training loop is located.

## API-documentation:
The environment documentation is next on the todo-list. For now I will say that the functionality is similar conceptually to the OpenAI gym environments, and should be quite understandable from reading the code (check out the function "thread_code" in threads/worker_thread.py). Having said that, if you would like to see documentation happen faster, or if you have any question regarding this, contact me and I will happily answer.

## Contribute!
If you want to get involved in this project and want to know what needs to be done, feel free to contact me and I will be happy to discuss this!

If you have concrete ideas for improvement, or think something is lacking, or have any other suggestion, I will be glad to hear about it, but won't guarantee that it happens :-)

[SpeedBlocks]: <https://github.com/kroyee/SpeedBlocks>
[tensorflow]: <https://www.tensorflow.org/install/>
