


# dqn_navigation

An implementation of DQN to solve the Unity Machine Learning Agents Toolkit Navigation environment. 

## Introduction

For this project we will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent](banana.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Installation

To set up your python environment to run the code in this repository, follow the instructions below.

Create (and activate) a new environment with Python 3.6.
Linux or Mac:

    conda create --name drlnd python=3.6
    source activate drlnd

Windows:

    conda create --name drlnd python=3.6 
    activate drlnd

Follow the instructions in this repository https://github.com/openai/gym to perform a minimal install of OpenAI gym.

Clone this repository  (if you haven't already!) then, install several dependencies.

    pip install -r requirements.txt

Create an IPython kernel for the drlnd environment.

    python -m ipykernel install --user --name drlnd --display-name "drlnd"

Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.

![Jupyter Kernel](jupyter_kernel.png)

You will also need to download the environment file and unzip it in the project root directory. The environments are (by platform)

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [Windows 32bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* [Windows 64bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


## Approach

The implementation approach is based on DQN. The code provides a number of DQN extensions: Double DQN, N-step DQN and Priorized Experience Replay. We compare the convergence of these different approaches.

## Running 

In order to use the project open the Jupyter notebook `Report.ipynb`. This notebook contains further details of the environment, agents and training.


