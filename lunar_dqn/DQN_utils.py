import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size int(1e5)
BATCH_SIZE = 64         # minibatch size 64
GAMMA = 0.99            # discount factor 0.99
TAU = 1e-3              # for soft update of target parameters 1e-3
LR = 5e-4               # learning rate  5e-4
UPDATE_EVERY = 5        # how often to update the network 4
LEARN_EVERY = UPDATE_EVERY

OFF=0
MAIN=2
RIGHT=1
LEFT=3

ACTIONS=[0,2,1,3]
num2act = {num:act for num, act in zip(ACTIONS, ["OFF  ", "MAIN ", "RIGHT", "LEFT "])}
HIGHS=np.array([1.5, 1.5, 5., 5., 3.1415927, 5., True, True])
LOWS=np.array([-1.5, -1.5, -5., -5., -3.1415927, -5., False, False])
norm = lambda x: (x - x.mean())/x.std() if x.std()!=0. else 0.
pix_norm = lambda x: x/255.
scale = lambda x: (x - x.min())/(x.max() - x.min()) if x.max()!=x.min() else x
SEED = 1234

def scale_input(state):
    st = np.zeros_like(state)
    st[:-2] = (state[:-2] - LOWS[:-2]) / (HIGHS[:-2] - LOWS[:-2])
    st[-2:] = state[-2:]  #don't norm the binary True or False values
    return st

def scale_input_batch(states):
    states =  [scale_input(state) for state in states]
    return states

def log_norm(rewards):
    '''
       rewards (list): list-like of floats between (-100, 100)
       returns a list of scaled, then logged values
    '''
    rewards = [(r+100.) for r in rewards] # values --> (0, 1), no negatives!
    rewards = [np.log10(r) if r>0 else 0 for r in rewards] # smooth out distribution
    return rewards

def disentangle(experiences):
    '''Separates SARS'D Experiences into its components
        params:
        experiences (deque): list of Experiences, 
                             named tuples of (state, action, reward, next_state, done)
                       
        returns:
        a list for each named component in the tuple 
    '''
    #states = [e.state for e in experiences if e is not None]
    #actions = [e.action for e in experiences if e is not None]
    #rewards = [e.reward for e in experiences if e is not None]
    #next_states = [e.next_state for e in experiences if e is not None]
    #dones = [e.done for e in experiences if e is not None]
    states, actions, rewards, next_states, dones = zip(*experiences)
    return states, actions, rewards, next_states, dones
