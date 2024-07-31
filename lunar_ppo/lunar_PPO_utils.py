from parallelEnv_PPO import parallelEnv 
from surrogate_fxns import *
#import random as rand
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from JSAnimation.IPython_display import display_animation
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

OFF=0
MAIN=2
RIGHT=3
LEFT=1
ACTIONS=[OFF,MAIN,RIGHT,LEFT]

LOWS=np.array([-1.5, -1.5, -5., -5., -3.1415927, -5., False, False])
HIGHS=np.array([1.5, 1.5, 5., 5., 3.1415927, 5., True, True])

norm = lambda x: (x - x.mean()) / x.std() if x.std()!=0. else 0.
scale = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max()!=x.min() else x
pix_norm = lambda x: x/255.
pols2probs = nn.Softmax(dim=1)

SEED=1234

def scale_input(state):
    scaled_state = np.zeros_like(state)
    scaled_state[:-2] = (state[:-2] - LOWS[:-2]) / (HIGHS[:-2] - LOWS[:-2])
    scaled_state[-2:] = state[-2:] 
    return scaled_state

def scale_input_batch(states):
    scaled_batch =  np.array([scale_input(state) for state in states])     
    return scaled_batch

#Because the full states of many of the Atari games are not completely observable from the image frames, 
#Mnih et al. “stacked” the four most recent frames so that the inputs to the network had dimension
#84⇥84⇥4. This did not eliminate partial observability for all of the games, but it was
#helpful in making many of them more Markovian.
# Text pg. 438

def prep_input(state0, state1):
    '''Takes two sequential states and returns input to the network   
        Params
        states (np.array) shape (8,)
        Returns (np.array) shape (2,8) prepped for input 
    '''
    state0 = scale_input(state0)
    state1 = scale_input(state1)
    state = np.asarray([state0, state1])
    return state

# preprocess a single frame
# crop image and downsample to 80x80
# stack two frames together as input
def prep_frame(image, bkg_color = np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2, ::2]-bkg_color, axis=-1)/255.
    return img

# this is useful for batch processing especially on the GPU
def prep_frames(images, bkg_color = np.array([144, 72, 17])):
    '''
        prepare the rendered frames from all parallel agents for 
        convolutional neural net
    '''
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images[:, :, 34:-16:2, ::2]-bkg_color, axis=-1)/255.
    batch_input = np.swapaxes(list_of_images_prepro,0,1)
    return torch.from_numpy(batch_input).float().to(device)

# function to animate a list of frames
def animate_frames(frames):
    plt.axis('off')

    # color option for plotting
    # use Greys for greyscale
    cmap = None if len(frames[0].shape)==3 else 'Greys'
    patch = plt.imshow(frames[0], cmap=cmap)  

    fanim = animation.FuncAnimation(plt.gcf(),
                                    lambda x: patch.set_data(frames[x]), 
                                    frames = len(frames), 
                                    interval=30)
    
    display(display_animation(fanim, default_mode='once'))
    
# play a game and display the animation
# nrand = number of random steps before using the policy
def play(env, policy, time=200, preprocess=None, nrand=5):
    env.reset(seed=SEED)

    # start game
    action = env.action_space.sample()
    env.step((0,0))
    
    # perform nrand random steps in the beginning
    for _ in range(nrand):
        state1, reward1, is_done, is_trunc, info = env.step(env.action_space.sample())
        state2, reward2, is_done, is_trunc, info = env.step((0,0))     
        
    state1, reward1, is_done, is_trunc, info = env.step(env.action_space.sample())
    frame1 = env.render()
    state2, reward2, is_done, is_trunc, info = env.step((0,0))
    frame2 = env.render()   
    
    anim_frames = []
    for _ in range(time):
        policy_input = prep_input(state1, state2)  
        action = policy(policy_input)
        state1, _, is_done, is_trunc, _ = env.step(action)
        frame1 = env.render()
        if preprocess is None:
            anim_frames.append(frame1)       
        else:
            state2, _, is_done, is_trunc, _ = env.step(0)
            frame2 = env.render()
            anim_frames.append(prep_frames([frame1 , frame2]))
        if is_done or is_trunc:
            break
    #env.close()
    animate_frames(anim_frames)
    return 

# collect trajectories for a parallelized parallelEnv object
def collect_trajectories(envs, policy, tmax=1000, skip=0):
    
    # number of parallel instances
    n=len(envs.ps)

    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    pol_list=[]
    prob_list=[]
    action_list=[]
    
    # To convert raw values to between (0, 1)
    pols2probs = nn.Softmax(dim=1).to(device)
 
    # initiate all parallel environments
    envs.reset()
    _, _, _, _ = envs.step([2]*n) #fire main engine once
    
    # perform same # random steps for each agent
    for _ in range(skip):
        states, _, _, _ = envs.step([envs.action_space.sample()]*n)
     
    for t in range(tmax):
        # prepare the input
        # preprocess_states returns a shape (n, state.shape) as input for the policy
        #states = prep_input(states)  #scale_input_batch(states)
        
        # pols --> probs --> pi_old
        #pols = states_to_pols(policy, states).squeeze().cpu().detach().numpy()
       
        # convert to literal probabilities for greedy action choices
        ########## TO DO "no further gradient propagation needed so back to cpu"
        #probs = pols2probs(pols).squeeze().cpu().detach().numpy()
        probs = states_to_probs(policy, states).squeeze().cpu().detach().numpy()
        # choose actions with highest value/prob
        actions = np.argmax(probs, axis=1)
        
        # take action and skip game one step forward
        states, rewards, is_done, _ = envs.step(actions)
        #st2, re2, is_done, is_trun, _ = envs.step([0]*n)
        #is_done = is_done or is_trun
        #rewards = re1 + re2
        
        # store the results of this step
        state_list.append(states)
        action_list.append(actions)
        reward_list.append(rewards)
        #pol_list.append(pols)
        prob_list.append(probs)
        
        # stop if any of the trajectories is done
        if is_done.any():
            break
    # pol_list, 
    return prob_list, state_list, action_list, reward_list

# convert states to probability by passing through the policy
#def states_to_probs(policy, states):
    # shape of input is
#    states = torch.stack(states)
#    policy_input = states.view(-1,states.shape[-2])
#    pols = policy(policy_input)
#    probs = pols2probs(pols).view(states.shape[:-2])
    # shape of output is
#    return probs

# convert states to policy output states in shape ()
#def states_to_pols(policy, states):
    # shape of input is
#    states = torch.stack(states)
#    policy_input = states.view(-1,*states.shape[-2:])
#    pols = policy(policy_input).view(states.shape[:-2])
    # shape of output is
#    return pols

def states_to_sigs(policy, states):
    ''' Sends state values through policy
        params: 
            policy (pytorch NN)
            states (np.array-like)
       returns: 
            sigs (tensor): passes raw output values of network through
                           a sigmoid filter; values between 0, 1
    '''
    sigmo = nn.Sigmoid().to(device)
    policy.eval()
    with torch.no_grad():
        if torch.is_tensor(states):
            sigs = policy(states.to(device))
        else:
            states = np.asarray(states) 
            sigs = policy(torch.from_numpy(states).to(device))
        sigs = sigmo(sigs)          #.detach().cpu().numpy()
    policy.train()      
    return sigs

def states_to_pols(policy, states):
    '''
       params: 
            policy (pytorch NN)
            states (np.array)
       returns: 
           pols (tensor): policy values, "raw" output of NN w/out activation on last layer
    '''
    policy.eval()
    with torch.no_grad():
        if torch.is_tensor(states):
            pols = policy(states.to(device))
        else:
            pols = policy( torch.from_numpy(states).to(device) )
    policy.train()
    return pols
    
def states_to_probs(policy, states):
    ''' Sends state values through policy
        params: 
            policy (pytorch NN)
            states (np.array-like)
       returns: 
            probs (tensor): converts output values of network to
                            normal distribution of probs : sum(probs)=1.
    '''
    softprobs = nn.Softmax(dim=1).to(device)
    policy.eval()
    with torch.no_grad():
        if torch.is_tensor(states):
            pols = policy(states.to(device))
        else:
            #states = np.asarray(states) 
            pols = policy(torch.from_numpy(states).to(device))
        probs = softprobs(pols)          #.detach().cpu().numpy()
    policy.train()      
    return probs

# the output is the probability of moving right
# P(left) = 1-P(right)
class SolvedImagePolicy(nn.Module):

    def __init__(self):
        super(SolvedImagePolicy, self).__init__()
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size=9*9*16
        
        # two fully connected layers
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)

        # Sigmoid to 
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))
    
class VectorPolicy(nn.Module):
    ''' Estimate action probabilities from states
        state_size: number of vector observations
        action_size: number of discrete actions available
    '''
    def __init__(self, state_size=8, action_size=4, seed=1234):
        super(VectorPolicy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        # fully connected layers
        self.state_in = nn.Linear(state_size, 32, dtype=torch.float32)
        #self.action_in = nn.Linear(action_size, 32) 
        self.hidden = nn.Linear(32, 16, dtype=torch.float32)
        self.action_qs = nn.Linear(16, action_size, dtype=torch.float32)
        # Sigmoid to action-values each in 0,1 
        #self.sigmoid = nn.Sigmoid()
        # Tanh activation to action values in -1,1
        #self.tanh = nn.Tanh()
        self.leaky = nn.LeakyReLU()
    
    def forward(self, state):
        # expects pre-normed values around mean +/- std
        x = self.leaky(self.state_in(state))
        x = self.leaky(self.hidden(x))
        #return self.sigmoid(self.action_qs(x)) # returns q-values for n_actions in 0,1
        return self.action_qs(x)  # raw network output