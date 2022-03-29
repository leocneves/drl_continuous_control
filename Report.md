[//]: # (Image References)

[image1]: results.png  "Results"

### Learning Algorithm

For first, this repository shows how to solve "Unity ML-Agents Reacher Environment" (more details can be found [HERE](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#reacher)) with a DDPG strategy from th  paper in [THIS LINK](https://arxiv.org/abs/1509.02971). For this task we are using the base model available in Lessons of Udacity Nanodegree Deep Reinforcement Learning. Let's check it!

#### The 'model.py' file

For the architecture we choose to implement, in model we need to define two main classes to help with neural nets architecture. Actor-Critic methods needs two Neural Nets and it's what we can see in this file.

Actor Class, containing the Actor Neural Nets achitecture can be defined as:

```python
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=0, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

```

As we can see, two hidden layers are used and default number of neurons are 128 for each layer.

A forward function is defined too and is responsible for foward propagation of this neural net. As we can see below, ReLu activations are responsible for process outputs.

```python
def forward(self, state):
    """Build an actor (policy) network that maps states -> actions."""
    if state.dim() == 1:
        state = torch.unsqueeze(state,0)
    x = F.relu(self.fc1(state))
    x = self.bn1(x)
    x = F.relu(self.fc2(x))
    return F.tanh(self.fc3(x))
```
**Critic class has the same structure of Actor and this two neural networks are equal represented in model.py file.**

#### The 'ddpg_agent.py' file

This file has the class **Agent**, that describes our agent with interact functions, class **OUNoise** that help us with action selections and class **ReplayBuffer** that is responsible to manipulate the interaction experiences of the agent.

For first, for we initiate the agent we need to entry some initial params:

```python
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

```

Here we need to specify the state space size (33 on this continuous control task), the action size (we've 4 actions on this env.) and the seed.

Now, we have 3 main functions...

1. Step Function: This function is responsible for receives state, action, reward, next_state, done *from unity environment*, ADD the new observation experience from the env. in memory buffer and call the function *learn* to update weights with the new values. We can see the peace of code below:

```python
def step(self, state, action, reward, next_state, done):
    """Save experience in replay memory, and use random sample from buffer to learn."""
    # Save experience / reward
    self.memory.add(state, action, reward, next_state, done)

```

2. Act Function: This function is responsible for choose actions from observed state (input param). Here we have actions selected from actor neural net and here we can add some noise in actions selected (continuous) to add bias at learning step.
We can see the peace of code below:

```python
def act(self, state, add_noise=True):
    """Returns actions for given state as per current policy."""
    state = torch.from_numpy(state).float().to(device)
    self.actor_local.eval()
    with torch.no_grad():
        action = self.actor_local(state).cpu().data.numpy()
    self.actor_local.train()
    if add_noise:
        action += self.noise.sample()
    return np.clip(action, -1, 1)
```

3. Learn Function: This function is responsible for update value parameters using given batch of experience tuples. In this step we basically update the neural net weights, both Actor and Critic nets from experiences.
We can see the peace of code below:

```python
def learn(self, experiences, gamma):
    """Update policy and value parameters using given batch of experience tuples.
    Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
    where:
        actor_target(state) -> action
        critic_target(state, action) -> Q-value
    Params
    ======
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor
    """
    states, actions, rewards, next_states, dones = experiences

    # ---------------------------- update critic ---------------------------- #
    # Get predicted next-state actions and Q values from target models
    actions_next = self.actor_target(next_states)
    Q_targets_next = self.critic_target(next_states, actions_next)
    # Compute Q targets for current states (y_i)
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    # Compute critic loss
    Q_expected = self.critic_local(states, actions)
    critic_loss = F.mse_loss(Q_expected, Q_targets)
    # Minimize the loss
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # ---------------------------- update actor ---------------------------- #
    # Compute actor loss
    actions_pred = self.actor_local(states)
    actor_loss = -self.critic_local(states, actions_pred).mean()
    # Minimize the loss
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # ----------------------- update target networks ----------------------- #
    self.soft_update(self.critic_local, self.critic_target, TAU)
    self.soft_update(self.actor_local, self.actor_target, TAU)

```

---

The class **ReplayBuffer** just manipulate the experience buffer at each step by the agent. The code below shows the initialize parameters for this object.

```python
"""Params
    action_size (int): dimension of each action
    buffer_size (int): maximum size of buffer
    batch_size (int): size of each training batch
    seed (int): random seed
"""
self.action_size = action_size
self.memory = deque(maxlen=buffer_size)
self.batch_size = batch_size
self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
self.seed = random.seed(seed)    
```

Here we have to define the number of actions, memory (length of experiences stored), batch size to store in buffer and this class has two simple methods: add and sample.

The **add function** is responsible to append new experiences in the buffer as we can see in next code part (simple like that!):

```python
def add(self, state, action, reward, next_state, done):
    """Add a new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)   
```

The **sample function** is responsible to randomly sample a batch of experiences from memory and return this experiences to train the neural net.

```python
def sample(self):
    """Randomly sample a batch of experiences from memory."""
    experiences = random.sample(self.memory, k=self.batch_size)

    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

    return (states, actions, rewards, next_states, dones)  
```

### Hyperparameters used

Here, in 'ddpg_agent.py' file we have some hyperparameters to ours neural nets and buffer memory. We can use a lot of strategies to define this parameters like grid search to find best ones, but I used the default numbers suggested in Nanodegree Lessons about deep reinforcement learning and can be found here:

```python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
```


### Plot of Rewards

In the next figure we can see the plot of Score x Episodes til the end of episodes or the agent solve the task (Mean of last 100 scores equal or greater than +30.0).

![image1]

We can see that our agent was able to growth the reward received, showing the ability to learn with this env. and getting the score suggested by Udacity to declare the environment solved (more than +30.0) at episode **259**!

### Ideas for Future Work

For future work we can implement algorithms like REINFORCE, TNPG, RWR, REPS, TRPO, CEM, CMA-ES to compare the results. Not just it, we can change neural nets architectures and RL parameters with grid search strategies!

---
