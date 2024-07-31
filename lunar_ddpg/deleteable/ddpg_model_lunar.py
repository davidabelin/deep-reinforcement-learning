import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Timer():
    def __init__(self, seed=1234):
    
        #super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.start = time.time()
        
    def startt():
        pass
    def stopt():
        pass
    def clockt():
        pass
    
    
    
class Actor(nn.Module):
    """Actor (Policy) Model.
           An actor (policy) neural network for approximating $\mu=\max_a\left(Q(s,a)\right)$ for continuous action values
    """

    def __init__(self, state_size, action_size, seed=1234, fc1_units=128, fc2_units=64):
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
        self.fc1 = nn.Linear(state_size, fc1_units, bias=False)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.act =  nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.act.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, states):
        """An actor (policy) network for approximating function mu = maxQ(s,a)) over continuous values for actions
           Returns: mu (float):  mu(s) = max_a(Q(a,s)), 
           the "max" action values to apply, passed through a final tanh filter to guarantee
           values in (-1,1) as required. Return shape is (batch_size, 2), one each for main and lateral engines
        """
        x = F.leaky_relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = self.act(x)
        x[0] = torch.sigmoid(x[0])  # main engine range is 0, 1
        x[1] = torch.tanh(x[1])     # lateral engine range = -1, 1
        return x


class Critic(nn.Module):
    """Critic (Value) Model: (states,actions)-->V()"""
    def __init__(self, state_size, action_size, seed=1234, fcs1_units=64, fc2_units=128, fc3_units=32):
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
        self.fcs1 = nn.Linear(state_size, fcs1_units, bias=False)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units, bias=False)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Critic (value) network mapping (state, action) pairs -> the Q-values of the actions taken by actor
            state (array of floats): 
            action (tuple of floats): continuous actions in the range (0, -1) < (main, lateral) < (1, 1)
            returns a single value for each agent/state/action, shape: (n_agents, batch_size, 1)
        """
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)  #output shape:(batch_size, n_agents, 1) eg. (64, 12, 1)
