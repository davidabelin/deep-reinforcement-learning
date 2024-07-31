import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
import random

def hidden_init(layer):
    ''' From Models, called by Agent'''
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model.
           An actor (policy) neural network for approximating $\mu=\max_a\left(Q(s,a)\right)$ for continuous action values
    """

    def __init__(self, state_size, action_size, seed=SEED,
                       fc1_units=64, fc2_units=64, local=True):
        """Actor network approximates function "argmax of action-values" to estimate directly
           the deterministic, off-policy, continuous action-values themselves
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in first hidden layer
            output (float): 2-tuple of continuous action values in (-1,1) 
                            
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_out =  nn.Linear(fc2_units, action_size)
        self.reset_parameters(local)

    def reset_parameters(self, local):
        if local:
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc_out.weight.data.uniform_(-3e-3, 3e-3)
        else:
            print("Target Actor is Normal")
            self.fc1.weight.data.normal_(*hidden_init(self.fc1))
            self.fc2.weight.data.normal_(*hidden_init(self.fc2))
            self.fc_out.weight.data.normal_(-3e-3, 3e-3)   
            
    def forward(self, states):
        """An actor (policy) network for approximating function mu = maxQ(s,a)) over continuous values for actions
           Returns: mu (float):  mu(s) = max_a(Q(a,s)), 
           the "max" action values to apply, passed through a final tanh filter to guarantee
           values in (-1,1) as required. Return shape is (batch_size, 2), one each for main and lateral engines
        """
        x = F.leaky_relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)  # sigfxn guarantees all values are in (0,1)
        x[0] = torch.sigmoid(x[0])  ###   (1.1*+0.9)/2
        x[1] = torch.tanh(x[1])
        #x[1] = (1.1*x[1] + torch.sign(x[1])*0.9)/2
        return x

class Critic(nn.Module):
    """Critic (Value) Model: (states,actions)-->V()"""
    def __init__(self, state_size, action_size, seed=SEED, 
                       fc1_units=64, fc2_units=64, local=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            output (float): estimated value of each state given action mu(a|s) from actor
            
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1_s = nn.Linear(state_size, fc1_units)
        self.fc1_a = nn.Linear(action_size, fc2_units)
        self.fc2 = nn.Linear(fc1_units+fc2_units, state_size+action_size)
        self.fc3 = nn.Linear(state_size+action_size, 1)
        self.reset_parameters(local)

    def reset_parameters(self, local):
        if local:
            self.fc1_s.weight.data.uniform_(*hidden_init(self.fc1_s))
            self.fc1_a.weight.data.uniform_(*hidden_init(self.fc1_a))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        else:
            print("Target Critic is Normal")
            self.fc1_s.weight.data.normal_(*hidden_init(self.fc1_s))
            self.fc1_a.weight.data.normal_(*hidden_init(self.fc1_a))
            self.fc2.weight.data.normal_(*hidden_init(self.fc2))
            self.fc3.weight.data.normal_(-3e-3, 3e-3)
            
    def forward(self, state, action):
        """Critic or "value" network to map (state, action) pairs -> estimated ideal Q-values of the actions taken by actor.
            state (array of floats): state-vector values
            action (tuple of floats): continuous actions in the range: (0, -1) < (main, lateral) < (1, 1)
            returns a single state/action value to each agent at every step, shape: (n_agents, batch_size, 1)
        """
        xs = F.leaky_relu(self.fc1_s(state))
        xa = F.leaky_relu(self.fc1_a(action))
        x = torch.cat((xs, xa), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)  
        #output shape:(batch_size, n_agents, 1) eg. (64, 12, 1)

    
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
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
           softmax function. An epsilon-greedy choice can be applied to either to determine the action taken.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class QQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state0 = nn.Linear(state_size, fc1_units)
        self.state1 = nn.Linear(state_size, fc2_units)
        self.hidden = nn.Linear(fc1_units+fc2_units, state_size)
        self.out = nn.Linear(state_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values.
           Shape of input state should be: (batch_size, 2, state_size)
        """
        s0 = state[:,0,:].squeeze()
        s0 = F.relu(self.state0(s0))
        s1 = state[:,1,:].squeeze()
        s1 = F.relu(self.state1(s1))                       
        x = torch.cat((s0,s1), dim=-1)
        x = F.relu(self.hidden(x))
        return self.out(x)