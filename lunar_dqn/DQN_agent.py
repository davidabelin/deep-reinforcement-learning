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

    def __init__(self, state_size, action_size, seed=SEED, 
                       fc1_units=64, fc2_units=64, learn_every=LEARN_EVERY):
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
        self.fc1_units=fc1_units
        self.fc2_units=fc2_units
        #self.seed = random.seed(seed)

        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, action_size, seed, self.fc1_units, self.fc2_units).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, self.fc1_units, self.fc2_units).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, SEED)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.steps = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        self.steps +=1
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Sample size in memory, get random batch of torch tensors to train on
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.5):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state (N, state_size) or "double state" of shape (N, 2, state_size)
            eps (float): epsilon, for epsilon-greedy action selection
                         eps = 0: always policy-greedy
                         eps = 1: always random
        """
        # Epsilon-greedy action selection
        if random.random() > eps:
            # On-Policy, action=argmax(Q) (the 'greedy' option)
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                # Action values are raw outputs of network (ie. no activation function for last layer)
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            action_values = action_values.cpu().data.numpy()
            return np.argmax(action_values)
        else:
            # Stochastic Off-policy option
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

        # --------- soft update target network weights ---------- #
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

    def clear_memory(self):
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, SEED)
        self.t_step = 0
        self.steps = 0

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed=SEED):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)  
        #self.rewards = deque(maxlen=buffer_size)
        #self.reward_means = []
        #self.reward_stds = []
        # Each "experience" is the outcome (as "sars'd") from one step taken by one agent in env
        # state --> action --> reward & next_state (done=True if next state is S+)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = torch.manual_seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        #self.rewards.append(reward)
          
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = disentangle(experiences)
        
        # Scale state values to (0,1) using env-given max's and min's
        #states = scale_input_batch(states)
        #next_states = scale_input_batch(next_states)
        
        # Norm rewards to current mean and std of all rewards in memory
        #reward_mean = np.mean(self.rewards)
        #reward_std = np.std(self.rewards)
        #rewards = log_norm(rewards)
        #self.reward_means.append(reward_mean)
        #self.reward_stds.append(reward_std)
        
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

       
def evaluate(state, use_local=True):    
    """Using a network offline (no_grad) to estimate state -> action values.
       Input: 
       State (np.array) shaped as (batch_size, state_size)
       Use-local (bool) whether to use local or remote network; default local 
       Returns: 
       Action-values (tensor), shaped as (batch_size, action_size)
    """
    state = np.asarray(state)
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    network = agent.qnetwork_local if use_local else agent.qnetwork_remote
    network.eval()
    with torch.no_grad():
        action_values = network(state)
    network.train()  
    return action_values.detach().cpu().numpy()
     