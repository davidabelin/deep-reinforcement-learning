import numpy as np
import random
import copy
from collections import namedtuple, deque, defaultdict
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#####################################################################
############## SETTINGS 
##############

### Project hyperparameter settings
BUFFER_SIZE = int(6.4e4)  # replay buffer size = int(1e6)
BATCH_SIZE = 64         # minibatch size = 128
GAMMA = 0.99 #1-1/12    # discount factor = 0.99
TAU = 1e-3              # for soft update of target parameters = 1e-3
LR_ACTOR = 1e-4         # learning rate of the actor  = 1e-4
LR_CRITIC = 3e-4        # learning rate of the critic = 3e-4
WEIGHT_DECAY = 1e-4     # L2 weight decay = 0.0001
LR = 4e-4               # learning rate of Q-networks 5e-4
LEARN_EVERY = 4
SEED = 1234

EPSILON = 1.0
EPS_DECAY = 0.995
EPS_END = 0.005

### From the _discrete_ lunar environment
OFF=0
MAIN=2
RIGHT=1
LEFT=3
ACTIONS=[0,2,1,3]
num2act = {num:act for num, act in zip(ACTIONS, ["OFF  ", "MAIN ", "RIGHT", "LEFT "])}

### Lunar env states
HIGHS=np.array([1.5, 1.5, 5., 5., 3.1415927, 5., True, True])
LOWS=np.array([-1.5, -1.5, -5., -5., -3.1415927, -5., False, False])

### Quick-norms
norm_np = lambda x: (x - np.mean(x))/np.std(x) if np.std(x)!=0. else x.mean()/10
scale_np = lambda x: (x - np.min(x))/(np.max(x) - np.min(x)) if np.max(x)!=np.min(x) else 0.5
norm = lambda x: (x - x.mean())/x.std() if x.std()!=0. else x.mean()/10
scale = lambda x: (x - x.min())/(x.max() - x.min()) if x.max()!=x.min() else 0.5
pix_norm = lambda x: x/255.

action_size=2
####################################################################
###################### UTILS 
######################

def ez_noise(size=action_size, width=0.1):
    '''Width (float (0, 1)): the distance away from zero a noise-value can be chosen,
                             ie. the range within which to choose a random number '''
    width = np.clip(width, 0.0, 0.1)
    noiz0 = [ width*random.random() for _ in range(size)]
    noiz1 = [ width*2*(random.random() - 0.5) for _ in range(size)]
    noiz = (noiz0, noiz1)
    return noiz
    
def greedy_epsilon_action(epsilon, action_size=2, continuous=True):
    if continuous:
        #action-range is (0,1) for a0, (-1,1) for a1
        #return (a0,a1)
        pass
    else:
        pass
        

def hidden_layer_init(layer):
    ''' Taken from Models, called by Agent
        PARAMS
        layer (nn.Layer): the torch layer to be initialized        
    '''
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def scale_input(state):
    ''' Scales state values according to their LOWS and HIGHS given by the env.
    '''
    scaled = np.zeros_like(state)
    scaled[:-2] = (state[:-2] - LOWS[:-2]) / (HIGHS[:-2] - LOWS[:-2])
    scaled[-2:] = state[-2:] 
    return scaled

def scale_input_batch(states):
    ''' Scales batches of state values according to their LOWS and HIGHS
        Returns the batch as an np.array of shape (batch_size, state_size)
    '''
    sib =  np.asarray([scale_input(state) for state in states])     
    return sib

def disentangle(experiences):
    '''Separates SARS'D Experience data into its components
       experiences (deque): constant maximum sized list of 
       Experiences, named tuples of (state, action, reward, next_state, done) #, priority)           
       Returns: a separate list for each named component of the tuple 
    '''
    states, actions, rewards, next_states, dones = zip(*experiences)
    return states, actions, rewards, next_states, dones   #, priorities

#####################################################################
############## TO DO 
##############

##torch.nn.utils.clip_grad_norm_ where?

class Timer():
    '''TO DO'''
    def __init__(self, seed=1234):
        #super(Timer, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.start = time.time() 
    def startt():
        pass
    def stopt():
        pass
    def clockt():
        pass
    
def log_norm(rewards):
    '''
       rewards (list): list-like of floats between (-100, 100)
       returns a list of scaled, then logged values
    '''
    rewards = [(r+100.) for r in rewards] # values --> (0, 1), no negatives!
    rewards = [np.log10(r) if r>0 else 0 for r in rewards] # smooth out distribution
    return rewards

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
