import torch
import torch.nn as nn
import torch.nn.functional as F

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
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Pixel_QN(nn.Module):
    """Like a QNetwork but trains on image frames as states"""
    
    def __init__(self, state_size, action_size, seed=0):
        super(Pixel_QN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.convolve1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.convolve2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.convolve3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.think = nn.Linear(64 * 8 * 12, 64 * state_size)
        self.guess = nn.Linear(64 * state_size, action_size)

    def forward(self, frame):
        # Preprocess by grayscaling image
        x = self.grayscale(frame)
        x = x.unsqueeze(1)
        # Pass through convolutional layers
        x = F.relu(self.convolve1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.convolve2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.convolve3(x))
        x = F.max_pool2d(x, 2)
        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.think(x))
        x = F.relu(self.guess(x))
        return x
    
    def grayscale(self, image, human_eye=True):
        if human_eye:
           # Use scale factors ideal for human eyes:
           return np.dot(image[...,:3],[0.2989, 0.5870, 0.1140])
        else:
            return np.mean(image[...,:3])
