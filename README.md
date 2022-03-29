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
git clone https://github.com/leocneves/drl_continuous_control & cd dqn_navigation
```

3. Follow instructions in **Dependencies** from [THIS](https://github.com/udacity/deep-reinforcement-learning#dependencies) repository

4. Save file from [THIS LINK](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) in root of repository

5. Unzip it! (**Remember** this file has no vis, for see agent in environment download [THIS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip) version...)

```bash
sudo apt install unzip & unzip Reacher_Linux_NoVis.zip
```

6. Done! In this repository we've 4 main files:

 - Continuous_Control.ipynb: Notebook with train/test code;
 - ddpg_agent.py: Code with all structure of our agent (Parameters, functions to step, select actions, update rules, etc...);
 - model.py: Contains the architecture of Neural Net applied to our agent;
 - Report.mb: Contains the description of code and results.


5. To train the agent just open the notebook **Continuous_Control.ipynb** and execute all cells! At final of training step *(mean of last 100 rewards are more than +30 or episode are greater than 2000)* we can see *'checkpoint_actor.pth'* / *'checkpoint_critic.pth'* created where contains the weights of neural nets from training step and in *'results/'* we can see graph generated to illustrate convergence in learning by plotting scores for each 100 episodes.

### What's next?

* Go to the [main notebook](https://github.com/leocneves/drl_continuous_control/blob/master/Continuous_Control.ipynb) and try to execute by yourself!

* Visit Report.md file for more details about the code and results.

---
