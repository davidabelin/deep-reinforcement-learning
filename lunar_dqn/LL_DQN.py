##############  DQN_utils.py

import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(2.5e5)  # replay buffer size int(1e5)
BATCH_SIZE = 128          # minibatch size was 64
GAMMA = 0.99              # discount factor was 0.99
TAU = 7.5e-4              # for soft update of target parameters was 1e-3
LR = 3e-4                 # learning rate was 5e-4
LEARN_EVERY = 8           # how often to update the network was 4
SEED = 1234

OFF=0
MAIN=2
RIGHT=3
LEFT=1
ACTIONS=[0,2,3,1]
num2act = {num:act for num, act in zip(ACTIONS, ["OFF  ", "MAIN ", "RIGHT", "LEFT "])}
norm = lambda x: (x - x.mean())/x.std() if x.std()!=0. else 0.
scale = lambda x: (x - x.min())/(x.max() - x.min()) if x.max()!=x.min() else x
pix_norm = lambda x: x/255.

LOWS=np.array([-1.5, -1.5, -5., -5., -3.1415927, -5., False, False])
HIGHS=np.array([1.5, 1.5, 5., 5., 3.1415927, 5., True, True])
def scale_input(state):
    st = np.zeros_like(state)
    st[:-2] = (state[:-2] - LOWS[:-2]) / (HIGHS[:-2] - LOWS[:-2])
    st[-2:] = state[-2:] 
    return st
def scale_input_batch(states):
    sts =  np.array([scale_input(state) for state in states])     
    return sts

def disentangle(experiences):
    '''Separates SARS'D Experiences into its components
        params:
        experiences (deque): list of Experiences, 
                             named tuples of (state, action, reward, next_state, done)
                       
        returns:
        a list for each named component in the tuple 
    '''
    states = [e.state for e in experiences if e is not None]
    actions = [e.action for e in experiences if e is not None]
    rewards = [e.reward for e in experiences if e is not None]
    next_states = [e.next_state for e in experiences if e is not None]
    dones = [e.done for e in experiences if e is not None]
    return states, actions, rewards, next_states, dones

##############  Q_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Assign action_values to states according to current policy
           param: state (list-like) of values from environent of shape (batch_size, state_size)
           Returns a Q-value estimate for each possible action: "OFF", "MAIN", "RIGHT", "LEFT"
           These values can be converted into relative probabilities by passing the values through a
           softmax function. An epsilon-greedy choice can be made on either to determine the action taken.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

##############  DQN_agent.py

import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from Q_network import QNetwork
from DQN_utils import *

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, 
                       fc1_units=32, fc2_units=16, learn_every=LEARN_EVERY):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learn_every = learn_every
        self.step_tracker = 0
        #self.seed = random.seed(seed)

        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_units, fc2_units).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1_units, fc2_units).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
    def step(self, state, action, reward, next_state, done):
        """Save a complete episode in replay memory
           Periodically take random samples from buffer to train networks."""
        
	  # Save experience (*maybe*...)
        add_memory = True
        if reward==-100:
            if random.random()<0.5: add_memory = False   
        elif reward<0:
            if random.random()<0.667: add_memory = False  
            else: reward*=2          
        if add_memory:
            self.memory.add(state, action, reward, next_state, done)

        # Begin to train when enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            self.step_tracker = self.step_tracker%self.learn_every
            if self.step_tracker==0:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state or "double state" of shape (N, 2, state_size)
            eps (float): epsilon, for epsilon-greedy action selection
                         eps = 0: always greedy
                         eps = 1: always random
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            # Action values are raw outputs of network (ie. no activation function for last layer)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
      
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of N (s, a, r, s', done) tuples 
            For "extended" states, state_shape will be (N, trail_size, state_size)
            gamma (float): discount factor
        """
        # Expects data to be pre-torched, normed or scaled, and well-shaped
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_target = rewards + (gamma * Q_target_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_target)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                      
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from; local_theta
            target_model (PyTorch model): weights will be copied to; target_theta
            tau (float): interpolation parameter << 1.
        """
        for target_theta, local_theta in zip(target_model.parameters(), local_model.parameters()):
            target_theta.data.copy_(tau*local_theta.data + (1.0-tau)*target_theta.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)  
        self.rewards = deque(maxlen=buffer_size)
        self.reward_means = []
        self.reward_stds = []
        # Each "experience" is the outcome (as "sars'd") from one step taken by one agent in env
        # state --> action --> reward & next_state (done=True if next state is S+)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = torch.manual_seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.rewards.append(reward)
          
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = disentangle(experiences)
        
        # Scale state values to (0,1) using env-given max's and min's
        states = scale_input_batch(states)
        next_states = scale_input_batch(next_states)
        
        # Norm rewards to current mean and std of all rewards in memory
        reward_mean = np.mean(self.rewards)
        reward_std = np.std(self.rewards)
        rewards = (rewards - reward_mean)/reward_std if reward_std!=0. else rewards
        self.reward_means.append(reward_mean)
        self.reward_stds.append(reward_std)
        
        # Cast prepped input to tensors
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        # Return a batch of network-ready tensors
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# render ai gym environment
#!pip install gymnasium[box2d]
import gymnasium as gym

#!pip install progressbar
import progressbar as pb

from collections import deque
import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
%matplotlib inline

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display
else:  
    #!python -m pip install pyvirtualdisplay
    from pyvirtualdisplay import Display
    display = Display(visible=True, size=(1400, 900))
    display.start()

# install package for displaying animation
#!pip install JSAnimation
from JSAnimation.IPython_display import display_animation

import torch
import torch.nn as nn
import torch.nn.functional as F

import DQN_agent
from DQN_agent import Agent
from DQN_agent import SEED, LOWS, HIGHS, ACTIONS, num2act
from DQN_agent import disentangle, scale_input_batch, scale_input, scale, norm, pix_norm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ",device) 
plt.ion()

##### NEW GYM = GYMNASIUM
#!pip install gymnasium[box2d]
#import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="rgb_array",    #"human",       #
                                 continuous= False,
                                 gravity= -10.,
                                 enable_wind= False,
                                 wind_power= 0.,
                                 turbulence_power= 0.)
state, info = env.reset(seed = 1234)
obs = env.render()
    
done = False
while not done:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    state, reward, done, trun, info = env.step(action)
    done = done or trun
    #print(action, reward)
    obs = env.render()
    plt.imshow(obs)

state_shape = env.observation_space.shape
state_size = state_shape[0]
action_size = env.action_space.n
print('State shape: ', state_size)
print('Number of actions: ', action_size)
plt.imshow(obs)

seed = 1234
agent = Agent(state_size=state_size, action_size=action_size, seed=seed, 
              fc1_units=128, fc2_units=64, learn_every=4)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    window_size = 100                  # scores to rolling-remember
    scores_window = deque(maxlen=window_size)
    eps = eps_start                    # initialize epsilon
    episteps = max_t
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset(seed=SEED)
        score = 0
        for t in range(episteps):
            action = agent.act(state, eps)
            next_state, reward, done, trun, _ = env.step(action)
            agent.step(state, action, reward, next_state, done or trun)
            state = next_state
            score += reward
            if done or trun:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        print("\rEpisode {:4d} |  Average Score: {:-4.2f} |  Epsilon: {:1.3f} |  Max Steps: {:3d} |  Buffer: {:6d}".format(i_episode, 
                                                                                                                np.mean(scores_window), 
                                                                                                                eps,
                                                                                                                episteps,
                                                                                                                len(agent.memory.memory)), 
                                                                                                                end="")
        if i_episode % 100 == 0:
            chkpntname = "data/chkpnt{}.pth".format(i_episode)
            torch.save(agent.qnetwork_local.state_dict(), chkpntname)   
            print("\rEpisode {:4d} |  Average Score: {:-4.2f} |  Epsilon: {:1.3f} |  Max Steps: {:3d} |  Buffer: {:6d}".format(i_episode, 
                                                                                                            np.mean(scores_window), 
                                                                                                            eps,
                                                                                                            episteps,
                                                                                                            len(agent.memory.memory)))
        #episteps = (episteps - 1) if episteps>=100 else max_t
        
        if np.mean(scores_window)>=100.:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:-4.2f}'.format(i_episode-100, 
                                                                                           np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'data/highpoint.pth')
            break
             
    return scores

# Expects data to be pre-torched, normed or scaled, and well-shaped
#states, actions, rewards, next_states, dones = disentangle(agent.memory.memory)
print("states:", np.asarray(states).shape)


agent.qnetwork_local.eval()
agent.qnetwork_target.eval()
with torch.no_grad():    
    # To tensors
    Tstates = torch.from_numpy(np.vstack(states)).float().to(device)
    Tactions = torch.from_numpy(np.vstack(actions)).long().to(device)
    Trewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
    Tnext_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
    Tdones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

    # Get max predicted Q values (for next states) from target model
    Q_target_next = agent.qnetwork_target(Tnext_states).detach().max(1)[0].unsqueeze(1)
    print("Q_target_next:", Q_target_next.shape)

    # Compute Q targets for current states 
    Q_target = Trewards + (DQN_agent.GAMMA * Q_target_next * (1 - Tdones))
    print("Q_target:", Q_target.shape)

    # Get expected Q values from local model
    Q_expected = agent.qnetwork_local(Tstates).gather(1, Tactions)
    print("Q_expected:", Q_expected.shape)

    # Compute loss
    loss = F.mse_loss(Q_expected, Q_target)
    print("loss:", loss)  
agent.qnetwork_target.train()
agent.qnetwork_local.train()



scores = dqn(n_episodes=1200, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995)


## Output:
Episode  100 |  Average Score: -141.90 |  Epsilon: 0.606 |  Max Steps: 901 |  Buffer:   5187
Episode  200 |  Average Score: -130.18 |  Epsilon: 0.367 |  Max Steps: 801 |  Buffer:  11893
Episode  300 |  Average Score: -226.68 |  Epsilon: 0.222 |  Max Steps: 701 |  Buffer:  19348
Episode  400 |  Average Score: -224.73 |  Epsilon: 0.135 |  Max Steps: 601 |  Buffer:  25798
Episode  500 |  Average Score: -191.90 |  Epsilon: 0.082 |  Max Steps: 501 |  Buffer:  34182
Episode  600 |  Average Score: -151.25 |  Epsilon: 0.049 |  Max Steps: 401 |  Buffer:  46820
Episode  700 |  Average Score: -164.79 |  Epsilon: 0.030 |  Max Steps: 301 |  Buffer:  61318
Episode  800 |  Average Score: -212.62 |  Epsilon: 0.018 |  Max Steps: 201 |  Buffer:  70138
Episode  900 |  Average Score: -171.39 |  Epsilon: 0.011 |  Max Steps: 101 |  Buffer:  76742
Episode 1000 |  Average Score: -279.67 |  Epsilon: 0.010 |  Max Steps: 902 |  Buffer:  82316
Episode 1100 |  Average Score: -346.50 |  Epsilon: 0.010 |  Max Steps: 802 |  Buffer:  88092
Episode 1200 |  Average Score: -280.30 |  Epsilon: 0.010 |  Max Steps: 702 |  Buffer:  92782

