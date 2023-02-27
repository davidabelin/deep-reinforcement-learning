import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorX(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fci_units=32, fch_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ActorX, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fci = nn.Linear(state_size, fci_units)
        self.fch = nn.Linear(fci_units, fch_units)
        self.fco = nn.Linear(fch_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fci.weight.data.uniform_(*hidden_init(self.fci))
        self.fch.weight.data.uniform_(*hidden_init(self.fch))
        self.fco.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.leaky_relu(self.fci(state))
        x = F.leaky_relu(self.fch(x))
        return torch.tanh(self.fco(x))


class CriticX(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, 
                 in_units=32, h1_units=64, h2_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CriticX, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs = nn.Linear(state_size, in_units)
        self.fca = nn.Linear(action_size, in_units//2)
        self.fch1 = nn.Linear(in_units+in_units//2, h1_units)
        self.fch2 = nn.Linear(h1_units, h2_units)
        self.fco = nn.Linear(h2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs.weight.data.uniform_(*hidden_init(self.fcs))
        self.fca.weight.data.uniform_(*hidden_init(self.fca))
        self.fch1.weight.data.uniform_(*hidden_init(self.fch1))
        self.fch2.weight.data.uniform_(*hidden_init(self.fch2))
        self.fco.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs(state))
        xa = F.leaky_relu(self.fca(action))
        x = torch.cat((xs, xa), dim=1)
        x = F.leaky_relu(self.fch1(x))
        x = F.leaky_relu(self.fch2(x))
        return self.fco(x)
