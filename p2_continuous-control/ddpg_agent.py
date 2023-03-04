import numpy as np
import random
import copy
from collections import namedtuple, deque

try:
    from ddpg_model import ActorX as Actor
except:
    from ddpg_model import Actor
from ddpg_model import CriticX as Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 1e-6        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units=64, fc2_units=32).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units=64, fc2_units=32).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

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

    def reset(self):
        self.noise.reset()

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
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

    
class ReplayBuffer:
    """Vanilla Fixed-size buffer to store experience tuples. (ie. "trajectories"?)"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
            Params
            ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, step, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)
  

class PriorityReplay:
    """Fixed-size buffer to store experience tuples and batch them by priority.
       Collection of Trajectories? 
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a PriorityReplay object.
            Params
            ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)  # internal memory (deque)
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["step", "state", "action", "reward", "next_state", "done",
                                                                "priority"])
    
    def add(self, step, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        ### Priority calculated here:
        #priority = ((1000 - step)/1000 + int(done)) * abs(reward)
        #TO DO just use rewards for priorities ??
        priority = (1000 - step) * abs(reward)
        e = self.experience(step, state, action, reward, next_state, done, priority)
        self.buffer.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory according to their relative priority."""
        #TO DO just use rewards for priorities ??
        priorities = [m.priority for m in self.buffer]
        sum_priority = sum(priorities)          
        
        probs = [p / sum_priority for p in priorities]
        experiences = random.choices(population=self.buffer, k=BATCH_SIZE, weights=probs)
                    
        normed_rewards = norm_rewards(experiences)             
        normed_rewards = torch.from_numpy(np.vstack(normed_rewards)).float().to(device)
                    
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
                    
        return (states, actions, normed_rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)
                 
    def norm_rewards(self, experiences):
        rewards = [e.reward for e in experiences]
        ravg = np.mean(rewards)
        rstd = np.std(rewards)
        if rstd != 0:
            normed_rewards = [(r-ravg)/rstd for r in rewards]
        else:
            normed_rewards = rewards 
        return normed_rewards

norm = lambda x: (x - x.mean()) / (x.std()+10e-6)
scale = lambda x: (x - x.min()) / (x.max() - x.min())

#######
class AgentX():
    """Interacts with and learns from the environment. 
       Agent made of four networks: remote and local Critics; remote and local Actors
    """
    
    def __init__(self, state_size, action_size, random_seed, 
                 add_noise=True, priority_replay=False, lr=23):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            add_noise (bool): if True, add noise to action vectors
            priority_replay (bool): if working, else vanilla replay
            lr (int) : "learning rate" (between 13 and 23)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.add_noise = True #add_noise  
        self.priority_replay = False #priority_replay
        self.lr = lr
        
        # Memory: a replay buffer, or a priority replay buffer
        # Collections of Trajectory-like "episodes"
        if False: #priority_replay:
            self.memory =  PriorityReplay(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        else: 
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        ## Four networks per Agent:
        # Two Actor Networks: learns (states --> actions)
        self.actor_local = Actor(state_size, action_size, random_seed,
                                 fc1_units=64, fc2_units=32).to(device)     
        self.actor_target = Actor(state_size, action_size, random_seed,
                                 fc1_units=64, fc2_units=32).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Two Critic Networks: learns (states --> actions)
        self.critic_local = CriticX(state_size, action_size, random_seed).to(device)
        self.critic_target = CriticX(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, 
                                                               weight_decay=WEIGHT_DECAY)

        # Noise to add to actions
        #if self.add_noise: 
        self.noise = OUNoise(action_size, random_seed)
        #else: self.noise = None

    def reset(self):
        #if self.add_noise: 
        self.noise.reset()

    def step(self, step, state, action, reward, next_state, done):
        """ Save experience in replay memory; then
            Use a random sample from the buffer to train on.
        """
        if len(self.memory) > BATCH_SIZE and reward > 0.:
            ### Save experience ONLY if reward? <<< TO DO (probably)
                                ### B/c what use are non-scoring steps?
            self.memory.add(step, state, action, reward, next_state, done)
            
            # Learn from memory batches as soon as n_experiences > batch_size
            if step % self.lr == 0: 
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if self.add_noise:
            action += self.noise.sample()
        return np.clip(action, -1., 1.)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (step, s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        ####### TO DO What to do about the dones? 
        ####### Log of all rewards? Pre-scale?
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_mse_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize critic loss
        self.critic_optimizer.zero_grad()
        critic_mse_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------- soft update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

        #### TO DO return critic_mse_loss.tolist() for priority adjustment?

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), 
                                             local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def train_one_agent(n_episodes=1000, print_every=100, max_score=-np.inf):
    scores_deque = deque(maxlen=100)
    all_scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        steps = np.zeros(num_agents)
        while True:
            steps += 1
            actions = agency.act(states)                       # list-like of actions (onex4 per agent)
            env_info = env.step(actions)[brain_name]           # send all 20 actions to tne environment
            next_states = env_info.vector_observations         # get next states (one per agent)
            rewards = env_info.rewards                         # rewards returned (for each agent)
            dones = env_info.local_done                        # see if any episodes are finished
            agency.step(steps, states, actions, rewards, next_states, dones)
            states = next_states                               # roll over states to next time step
            scores += rewards                                  # update the score (for each agent)
            if np.any(dones):                                  # exit loop if any episode finished
                break    
        scores_deque.append(scores)
        all_scores.append(scores) 
        #if max(all_scores).any() > max_score: max_score = max(all_scores).any()
        epi_max = np.max([[np.max(s) for s in scores] for scores in all_scores])
        if max_score < epi_max: max_score = epi_max
        avg_score = np.mean([[np.mean(s) for s in scores] for scores in scores_deque])
        print('\rEpisode {}\tAverage Score: {:.2f}\tHigh Score: {:.2f}'.format(i_episode, avg_score, max_score), end="")
        if i_episode % print_every == 0:
            agent = random.choice(agency.agents) #[a for a in agency.agents])
            torch.save(agent.actor_local.state_dict(), './data/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), './data/checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}\tHigh Score: {:.2f}\n'.format(i_episode, avg_score, max_score))   
    return all_scores, max_score     