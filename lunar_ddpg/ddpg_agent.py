import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(5.12e4)   # replay buffer size
BATCH_SIZE = 128            # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 1e-9       # L2 weight decay
SEED = 1234

actor_units_in=256
actor_units_hid=128
critic_units_in=512
critic_units_hid=128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scalenp = lambda x: 0. if np.max(x)==np.min(x) else (x - np.min(x))/(np.max(x) - np.min(x))
normnp = lambda x: 0. if np.std(x)==0 else (x - np.mean(x))/np.std(x) 

class Agent():
    """Coordinates training for the neural networks."""
    def __init__(self, state_size, action_size, num_agents=1, random_seed=SEED,
                       actor_units_in=actor_units_in, actor_units_hid=actor_units_hid,
                       critic_units_in=critic_units_in, critic_units_hid=critic_units_hid,
                       actor_filename=None, critic_filename=None):
        """Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): not used here
            random_seed (int): random seed, widely shared but not really used...
            actor_units (int): number of units for layers in the actor model
            critic_units (int): number of units for layers in the critic model
            actor_filename=None, critic_filename=None
        """
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.num_agents = num_agents
        self.seed = random.seed(random_seed) # what happens here?
        self.steps = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed,
                                 actor_units_in, actor_units_hid).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed,
                                  actor_units_in, actor_units_hid).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed,
                                    critic_units_in, critic_units_hid).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed,
                                    critic_units_in, critic_units_hid).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        if num_agents > 1: noisize = (num_agents, action_size)
        else: noisize = action_size
        self.noise = OUNoise(noisize, random_seed, mu=0., theta=0.15, sigma=0.2)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        location='cpu'  # device 
        if actor_filename:
            actor_weights = torch.load(actor_filename, map_location=location)
            self.actor_local.load_state_dict(actor_weights)
            self.actor_target.load_state_dict(actor_weights)

        if critic_filename:
            critic_weights = torch.load(critic_filename, map_location=location)
            self.critic_local.load_state_dict(critic_weights)
            self.critic_target.load_state_dict(critic_weights)
        

    def act(self, state, add_noise=True, return_noise=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().squeeze().numpy()
        self.actor_local.train()
        
        if return_noise:
            return action, self.noise.sample()
        elif add_noise:
            action += self.noise.sample()
            return np.clip(action, -1, 1)
        else:
            return action

    def reset(self):
        self.noise.reset()

    def step(self, state, action, reward, next_state, done, return_loss=False):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.steps+=1
        # Save step data as an experience tuple in memory
        self.memory.add(state, action, reward, next_state, done)
        # Begin batching from memory when buffer has enough data
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            actor_loss, critic_loss = self.learn(experiences, GAMMA)
            if return_loss: return actor_loss, critic_loss
            else: return None, None
        else: return None, None

    def learn(self, experiences, gamma):
        """
        Update policy and value parameters using batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state)) where:
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
        critic_loss = F.mse_loss(Q_expected, Q_targets) #torch.clamp(, -1e6, 1e6)
        
        # Backprop critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean() #torch.clamp(, -1e6, 1e6)
        
        # Backprop actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

        return actor_loss.detach().cpu().squeeze().numpy(), critic_loss.detach().cpu().squeeze().numpy()        
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed) # why?
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).float().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)