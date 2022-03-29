[//]: # (Image References)

[image1]: imgs/agent.gif "Trained Agent"


# Continuous Control Project - Using DDPG Algorithm to follow trajectories!

### Introduction

This repository explain how to train your own intelligent agent to follows trajectories in a environment. It's a project part of [Udacity Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) in Deep Reinforcement Learning.

The task selected to train our agents is Continuous Control a double-jointed arm. Basically, we have a 3d environment with a double-jointed arm and we need to this arm follows a target location, like the figure below...

<center>

![Trained Agent][image1]

</center>

The environment gives a reward of +0.1 for each step that the agent's hand is in the goal location and the goal of your agent is to maintain its position at the target location for as many time steps as possible.

Here, the state space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

To remember, the task is episodic and in order to solve the environment, your agent **must get an average score of +30 over 100 consecutive episodes**!

### Getting Start

For first, let's clone this repository... \
(Let's assume that you are executing this on Linux OS)

1. Create a path to clone the project

```bash
mkdir NAME_OF_PROJECT & cd NAME_OF_PROJECT
```

2. Clone the project

```bash
git clone https://github.com/leocneves/drl_continuous_control & cd drl_continuous_control
```

3. Follow instructions in **Dependencies** from [THIS](https://github.com/udacity/deep-reinforcement-learning#dependencies) repository

4. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

5. Place the file in the DRLND GitHub repository, in the `drl_continuous_control/` folder, and unzip (or decompress) the file.

6. Done! In this repository we've 4 main files:

 - Continuous_Control.ipynb: Notebook with train/test code;
 - ddpg_agent.py: Code with all structure of our agent (Parameters, functions to step, select actions, update rules, etc...);
 - model.py: Contains the architecture of Neural Net applied to our agent;
 - Report.mb: Contains the description of code and results.


7. To train the agent just open the notebook **Continuous_Control.ipynb** and execute all cells! At final of training step *(mean of last 100 rewards are more than +30 or episode are greater than 2000)* we can see *'checkpoint_actor.pth'* / *'checkpoint_critic.pth'* created where contains the weights of neural nets from training step and in *'results/'* we can see graph generated to illustrate convergence in learning by plotting scores for each 100 episodes.

### What's next?

* Go to the [main notebook](https://github.com/leocneves/drl_continuous_control/blob/master/Continuous_Control.ipynb) and try to execute by yourself!

* Visit Report.md file for more details about the code and results.

---
