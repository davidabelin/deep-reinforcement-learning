import numpy as np
import random
import copy
from collections import namedtuple, deque, defaultdict

from ddpg_model_lunar import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size = int(1e6)
BATCH_SIZE = 64        # minibatch size = 128
GAMMA = 0.99 #1-1/12    # discount factor = 0.99
TAU = 1e-3              # for soft update of target parameters = 1e-3
LR_ACTOR = 1e-4         # learning rate of the actor  = 1e-4
LR_CRITIC = 3e-4        # learning rate of the critic = 3e-4
WEIGHT_DECAY = 1e-5     # L2 weight decay = 0.0001
LEARN_EVERY = 3
SEED = 1234

norm = lambda x: (x - np.mean(x))/np.std(x) if np.std(x)!=0. else x
pix_norm = lambda x: x/255.
scale = lambda x: (x - np.min(x))/(np.max(x) - np.min(x)) if np.max(x)!=np.min(x) else x

LOWS=np.array([-1.5, -1.5, -5., -5., -3.1415927, -5., False, False])
HIGHS=np.array([1.5, 1.5, 5., 5., 3.1415927, 5., True, True])

###################### UTILS #######################
def scale_input(state):
    ''' Scales state values according to their LOWS and HIGHS given by the env.
    '''
    scaled = np.zeros_like(state)
    scaled[:-2] = (state[:-2] - LOWS[:-2]) / (HIGHS[:-2] - LOWS[:-2])
    scaled[-2:] = state[-2:] 
    return scaled
def scale_input_batch(states):
    ''' Scales batches of state values according to their LOWS and HIGHS
        Returns the batch as an np.array
    '''
    sib =  np.asarray([scale_input(state) for state in states])     
    return sib

def disentangle(experiences):
    '''Separates SARS'D Experience data into its components
       experiences (deque): constant maximum sized list of Experiences with named tuples of 
                            (state, action, reward, next_state, done) #, priority)           
       Returns: a separate list for each named component of the tuple 
    '''
    #states = [e.state for e in experiences if e is not None]
    #actions = [e.action for e in experiences if e is not None]
    #rewards = [e.reward for e in experiences if e is not None]
    #next_states = [e.next_state for e in experiences if e is not None]
    #dones = [e.done for e in experiences if e is not None]
    #priorities = [e.priority for e in experiences if e is not None]
    states, actions, rewards, next_states, dones = zip(*experiences)
    return states, actions, rewards, next_states, dones#, priorities

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################################

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
        self.action_noise = OUNoise(action_size, random_seed)#, mu=0., theta=0.015, sigma=0.02)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed,
                                   fcs1_units=64, fc2_units=128, fc3_units=64).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed,
                                    fcs1_units=64, fc2_units=128, fc3_units=64).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.steps+=1
        # Initially prioritize experience by reward and by good engine use
        # [env source]: m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  ### 0.5..1.0 ### !!!
        #main_check = action[0]>-0.5 and action[0]<1.001
        #lateral_check = (action[1]> -1.001 and action[1]< -0.499) or (action[1]>0.499 and action[1]<1.001)
        #if main_check and lateral_check:
        if reward>=0.: priority=10.
        #elif reward==0.: priority=10.
        #elif random.random()>0.5:
        #    priority=9.9
        else: priority=9.999
        self.memory.add(state, action, reward, next_state, done, priority)
             
        # Sample a batch when enough samples are available in memory
        self.step_counter = int(self.step_counter%self.learn_every)
        if self.step_counter==0 and len(self.memory)>=2*BATCH_SIZE:
            experiences = self.memory.sample(scale_input=True)
            self.learn(experiences, gamma=GAMMA)
                
    def act(self, state, add_noise=True, return_noise=False):
        """Returns specific action for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        #if add_noise:
        noise = self.action_noise.sample()
        #action += noise
        #action[0] = np.clip(action[0], 0., 1.)
        #action[1] = np.clip(action[1], -1., 1.)
        if return_noise: return action, noise
        elif add_noise: return action - noise #### CAUTION: subtracting not adding!!
        else: return action

    def reset(self):
        self.action_noise.reset()

    def learn(self, experiences, gamma, add_noise=True):
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
        # Clipping the weights? results to avoid "reward cliff"
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
        #self.memories = defaultdict()
        #self.idx2mem = defaultdict()
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
    
    def sample(self, scale_input=False):
        """ Agent requests a randomly sampled batch of experiences from buffer.
            Then sent to Agent.learn()
        """
        #experiences = random.choices(self.memory, weights=priority_weights, k=self.batch_size)
        randex = random.choices(list(self.priorities.keys()), k=self.batch_size)
                                #weights=list(self.priorities.values()), 
                                #k=self.batch_size)
        # Disentange experiences
        # TO DO def disentangle(experiences):
        states=[]; actions=[]; rewards=[]; next_states=[]; dones=[];  # priorities=[];
        for idx in randex:
            e=self.memory[idx]
            #if e is not None:
            states.append(e.state) 
            actions.append(e.action)
            rewards.append(e.reward)
            next_states.append(e.next_state)
            dones.append(e.done)   
            #self.priorities[idx] = self.priorities[idx] * 0.999 # decay priority over time   
                
        if scale_input:
            #pass
            #def prep_data(data):
                # Scale state values to (0,1) using env-given max and min
            states = scale_input_batch(states)
            next_states = scale_input_batch(next_states)
            rewards = [(r+100)/200 for r in rewards]
        
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
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
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
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
# 3. try to subtract it from action upon return to act()
        return self.state