import numpy as np
import random
import copy
from collections import namedtuple, deque, defaultdict
import torch
import torch.nn.functional as F
import torch.optim as optim

from models import Actor, Critic, QNetwork
from utils import *

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed=SEED, learn_every=LEARN_EVERY):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed
        self.learn_every = learn_every # how often to train the network (in terms of steps taken)
        self.step_counter = 0 # modulo learn_every  
        self.steps = 0 # the full count, can be many times the buffer size

        # Noise process
        self.action_noise = OUNoise(action_size, random_seed, mu=0.0, theta=1.0, sigma=-1.0)
        #self.side_action_noise = OUNoise(1, random_seed, mu=0.0, theta=2.0, sigma=-1.0)
        # see also self.EZNoiz()

        ###### TO DO: different random_seeds in locals and targts? Critic and actors?
        ###### MOST RECENTLY: local is Uniform init, False is Normal init
        # Actor Networks (w/ Local and Target Networks)
        local = False
        self.actor_local = Actor(state_size, action_size, seed=random_seed,
                                 fc1_units=32, fc2_units=16, local=local).to(device)
        self.actor_target = Actor(state_size, action_size, seed=random_seed,
                                 fc1_units=32, fc2_units=16, local=local).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR) #local=0.67*random_seed+17

        # Critic Networks (w/ Local and Target Networks)
        self.critic_local = Critic(state_size, action_size, seed=random_seed, # local=0.67*random_seed+17
                                   fc1_units=32, fc2_units=16,
                                   local=local).to(device)
        self.critic_target = Critic(state_size, action_size, seed=random_seed, # local=0.67*random_seed+17
                                   fc1_units=32, fc2_units=16,
                                   local=local).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.steps+=1
        # Initially prioritize experience by reward and by good engine use
        priority=10.
        self.memory.add(state, action, reward, next_state, done, priority)
             
        # Sample a batch when enough samples are available in memory
        self.step_counter = int(self.step_counter%self.learn_every)
        if self.step_counter==0 and len(self.memory)>=4*BATCH_SIZE:
            experiences = self.memory.sample(scale_input=True, prioritize=True)
            self.learn(experiences, gamma=GAMMA)
                
    def act(self, state, epsilon=0., nu=0.):
        """Deterministically returns specific action for given state per current policy and the current epsilon-greediness factor, epsilon, which should decay as training proceeds.
           Only called from training loop.
           PARAMS 
           state (np.array): the state values from which to predict best action values
           epsilon (float): epsilon-greedy factor, should decay over episodes
           nu (float): width of noise (max=1, min=0), should decay over episodes
        """
        nu = np.clip(nu, -0.002, 2.002)
        if random.random() < epsilon:
            ## Markhov(ian) decision process
### TO DO: more efficient way?
            # results in choosing one number from a uniform distribution (0.45,1)            ## { (ten (0.45,1)s', one (0)) }
            action0 = (random.random()*1.05 + 0.95) / 2.                                       ## *10 + [0.0] ) 
            # choose +/- the number so taken from a distribution ((-1,-0.45), (0.45,1))      ##'{ (five s, one (0)) }
            action1 = ((random.random()*1.05 + 0.95) / 2.)*random.choice([-1,1])
                                                                                             ### []*5 + [0.0]) 
            # return the two choices as a [main, side] action
            action = np.array([action0, action1])
            action = (1-nu)*action + nu*self.action_noise.sample()
        else:
            ## Policy decision process
            s = torch.from_numpy(state).float().to(device)
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(s)
                action[0] = (action[0]*1.05 + 0.95) / 2.
                action[1] = (action[1]*1.05 + torch.sign(action[1])*0.95)/2.
            self.actor_local.train()
            #nu = np.max()
            action = (1-nu)*action.detach().cpu().data.numpy() + nu*self.action_noise.sample()
            del s
#noiz = nu*(random.random()-0.5)         # ez_noise(width=nu) # nu is distance away from zero towards +/-1
            #if nu>2.0 or nu<0: nu = 1.0  #np.clip(nu, 0, 0.25)
            #action = [a + nu*(random.random()-0.5) for a in action]
### TO DO: for noise use mean and sigma !!!!!
##action[1] = action[1]*random.choice([-1,1])
#[a+n for a, n in zip(action, noiz)]
        return action

    def reset(self):
        self.action_noise.reset()

    def learn(self, experiences, gamma, add_noise=False):
        """Update policy and value parameters using given batch of experience tuples.
           Batches are sampled according to priorities, which are updated proportional to 
        Q_values = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                                               state and reward values should be pre-normed
            gamma (float): discount factor
            add_noise (bool): whether or not to inject noise into action values here, default "not here"
        """
        states, actions, rewards, next_states, dones = experiences  
        
        # ------------------ priority ---------------- #
        # TO DO update priorities
            #priorities = priorities*0.999  #DID? priorities decay over time
            ### update priority according to which values...?
            
        # ------------------ critic ------------------ #
        # Target Actor newtork predicts next-state actions from 
        # next_states (provided by env.) using θ_target weights
        next_actions = self.actor_target(next_states)   #torch.clamp(, -1, 1)
        # Critic makes Q-value predictions from actor's action predictions
        # and their next_states, also with "slow update" θ_target weights
        Q_targets_next = self.critic_target(next_states, next_actions)
        # Use these values to TD-bootstrap one step up from current states, actions, rewards
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Local critic (with up-to-date θ_local weights) also predicts Q-values 
        # from actual current state and the values of the actions already taken
        # by Local Actor with θ_locals
        Q_expected = self.critic_local(states, actions)
        # Mean sqaure error of difference between expected values using
        # the fast-update local weights and slow-update θ_target weights
        # (two networks for... stability?)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimizing this difference brings both estimates closer to... agreement?
        self.critic_optimizer.zero_grad()
        # Backward pass through both local and target critics...?
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------- actor ------------------ #
        # Actor with θ_local weights predicts next action_values
        predicted_actions = self.actor_local(states)    #torch.clamp(, -1, 1)
        # Critic with θ_local weights estimates state values with these predicions 
        actor_loss = -self.critic_local(states, predicted_actions).mean()
        # Minimizing the negative of the error --> gradient ascent toward (local) max Q-values..?
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Clip the weights? Where? To avoid "reward cliff" during gradient ASCent?
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # -------- soft-update target weights -------- #
        # *Very Slowly* b/c Otherwise...?
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)     
        
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter << 1.0
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples and their corresponding priority weight. 
       Initialized by Agent as a memory buffer"""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.seed = torch.manual_seed(seed)
        self.memory = deque(maxlen=buffer_size)  # "window" size of buffer (as a deque)
        self.priorities = {} #defaultdict() 
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.idx = 0
        
    def add(self, state, action, reward, next_state, done, priority=10.):
        """Add a new experience/priority pair to their respective memory buffers.
           Experiences and priorities should share indices in both buffers, and should
           be discarded at the same rate.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.update({self.idx:priority})
        self.idx = (self.idx+1)%self.buffer_size
    
    def sample(self, scale_input=False, prioritize=False):
        """ Agent.learn() requests a randomly sampled batch of experiences from buffer.
        """
        if prioritize:
            states=[]; actions=[]; rewards=[]; next_states=[]; dones=[];
            randex = random.choices(list(self.priorities.keys()),
                                    weights=list(self.priorities.values()), 
                                    k=self.batch_size)
            for idx in randex:
                e=self.memory[idx]
                states.append(e.state) 
                actions.append(e.action)
                rewards.append(e.reward)
                next_states.append(e.next_state)
                dones.append(e.done)   
                self.priorities[idx] = self.priorities[idx] + e.reward/200 - 1e-3 # linear decay depending on reward
        else:
            experiences = random.sample(self.memory, k=self.batch_size)
            states, actions, rewards, next_states, dones = zip(*experiences)                    
        
        if scale_input:
            states = scale_input_batch(states)
            next_states = scale_input_batch(next_states)
            #rewards = [(r+100)/200 for r in rewards]

        # Cast prepped input to tensors and return to network
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).float().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        #priorities = torch.from_numpy(np.vstack(priorities)).float().to(device)
        
        # Return a batch of network-ready tensors
        return (states, actions, rewards, next_states, dones)#, priorities)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class OUNoise:
    """Ornstein-Uhlenbeck process.
       Provides the variance necessary for the gradient ascent step.
    """
# 2. try changing params of calling func NO
    def __init__(self, size, seed, mu=0., theta=1.0, sigma=1.0):
        """Initialize parameters and noise process."""
# 1. try to change back from np.ones  NO 
        self.mu = mu * np.zeros(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = torch.manual_seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random()-0.5 for i in range(len(x))])
        self.state = x + dx
# 3. try to subtract it from action upon return to act() NO
        return self.state


###############################################################################
######################   All BELOW FROM DQN   #################################
###############################################################################

class Q_Agent():
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

        # "Soft-update" target network weights
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

###############################################################################
######################  Q
###############################################################################
class QReplayBuffer:
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
     

###############################################################################
######################  Q
###############################################################################