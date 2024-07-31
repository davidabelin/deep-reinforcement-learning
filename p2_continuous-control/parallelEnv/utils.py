from parallelEnv import parallelEnv 
import random
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


################# TO DO convert to use with Unity envs 
# collect trajectories for a parallelized parallelEnv object
def collect_trajectories(envs, policy, tmax=200, nrand=5):
    
    # number of parallel instances
    n=len(envs.ps)

    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]

    envs.reset()
    
    # start all parallel agents
    envs.step([1]*n)
    
    # perform nrand random steps
    for _ in range(nrand):
######## TO DO discontinue taking two steps for everything ###############
        fr1, re1, _, _ = envs.step(np.random.choice([RIGHT, LEFT],n))
        fr2, re2, _, _ = envs.step([0]*n)
    
    for t in range(tmax):
        # preprocess_batch converts two "frames" into shape (n, 2, 80, 80)
        batch_input = preprocess_batch([fr1,fr2])
        
        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        probs = policy(batch_input).squeeze().cpu().detach().numpy()
        
        action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
        probs = np.where(action==RIGHT, probs, 1.0-probs)
        
        # advance the game (0=no action)
        # we take one action and skip game forward
        fr1, re1, is_done, _ = envs.step(action)
        fr2, re2, is_done, _ = envs.step([0]*n)

        reward = re1 + re2
        
        # store the result
        state_list.append(batch_input)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)
        
        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if is_done.any():
            break

    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, action_list, reward_list

# clipped surrogate function
# similar as -policy_loss for REINFORCE, but for PPO
def clipped_surrogate(policy, old_probs, states, actions, rewards, discount=0.995, epsilon=0.1, beta=0.01):

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)  ## TO DO what is this?
    
    # ratio for clipping
    ratio = new_probs/old_probs

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

    # include a regularization term to steer new_policy towards 0.5 
    # + 1.e-10 to avoid log(0) = nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+(1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    # returns an average of all the entries of the tensor, effectively computing
    # L_sur^clip / T averaged over time-steps, number of trajectories, and agents
    # >> "this is desirable because we have normalized our rewards" << TO DO
    return torch.mean(clipped_surrogate + beta*entropy)
        
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
        self.state_in = nn.Linear(state_size, 64)
        #self.action_in = nn.Linear(action_size, 32) 
        self.hidden = nn.Linear(64, 16)
        self.action_qs = nn.Linear(16, action_size)
        # Sigmoid to action-values each in 0,1 
        self.sigmoid = nn.Sigmoid()
        # Tanh activation to action values in -1,1
        self.tanh = nn.Tanh()
        self.leaky = nn.LeakyReLU()
    
    def forward(self, state):
        # expects pre-normed values around mean +/- std
        x = self.leaky(self.state_in(state))
        x = self.leaky(self.hidden(x))
        return self.sigmoid(self.action_qs(x)) # returns q-values for n_actions in 0,1

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
        pols = policy( torch.from_numpy( states ).to(device) )
    policy.train()
    return pols
    
# convert batch of states to batch of probabilities by passing through the policy
def states_to_prob(policy, states):
    '''Deprecated, use states_to_probs()'''
    states = torch.stack(states)
    policy_input = states.view(-1,*states.shape[-3:])
    return policy(policy_input).view(states.shape[:-3])    

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
        pols = policy(torch.from_numpy(states).to(device))
        probs = softprobs(pols)          #.detach().cpu().numpy()
    policy.train()      
    return probs

#################                TO DO                 #############
################# Adjust everything to work with Unity #############
#################                                      #############


# preprocess a single frame
# crop image and downsample to 80x80
# stack two frames together as input
def preprocess_single(image, bkg_color = np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2,::2]-bkg_color, axis=-1)/255.
    return img

# convert outputs of parallelEnv to inputs to pytorch neural net
# this is useful for batch processing especially on the GPU
def preprocess_batch(images, bkg_color = np.array([144, 72, 17])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5: list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images[:,:,34:-16:2,::2]-bkg_color, axis=-1)/255.
    batch_input = np.swapaxes(list_of_images_prepro,0,1)
    return torch.from_numpy(batch_input).float().to(device)

# function to animate a list of frames
def animate_frames(frames):
    plt.axis('off')

    # color option for plotting
    # use Greys for greyscale
    cmap = None if len(frames[0].shape)==3 else 'Greys'
    patch = plt.imshow(frames[0], cmap=cmap)  

    fanim = animation.FuncAnimation(plt.gcf(), \
        lambda x: patch.set_data(frames[x]), frames = len(frames), interval=30)
    
    display(display_animation(fanim, default_mode='once'))
   


# play a game and display the animation
# nrand = number of random steps before using the policy
def play(env, policy, time=2000, preprocess=None, nrand=5):
    env.reset()

    # star game
    env.step(1)
    
    # perform nrand random steps in the beginning
    for _ in range(nrand):
        frame1, reward1, is_done, _ = env.step(np.random.choice([RIGHT,LEFT]))
        frame2, reward2, is_done, _ = env.step(0)
    
    anim_frames = []
    
    for _ in range(time):
        
        frame_input = preprocess_batch([frame1, frame2])
        prob = policy(frame_input)
        
        # RIGHT = 4, LEFT = 5
        action = RIGHT if rand.random() < prob else LEFT
        frame1, _, is_done, _ = env.step(action)
        frame2, _, is_done, _ = env.step(0)

        if preprocess is None:
            anim_frames.append(frame1)
        else:
            anim_frames.append(preprocess(frame1))

        if is_done:
            break
    
    env.close()
    
    animate_frames(anim_frames)
    return 



def timer_train_policy(policy, envs, episodes, tmax=1000, SGD_epoch=4, gamma=0.5, epsilon=0.995, beta=0.999):

    # keep track of progress
    mean_rewards = []

    # keep track of how long training takes
    widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA() ]
    timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()

    for e in range(episodes):

        # collect trajectories
        old_probs, states, actions, rewards = collect_trajectories(envs, policy, tmax=tmax)

        total_rewards = np.sum(rewards, axis=0)

        # gradient ascent step
        for _ in range(SGD_epoch):

            # Loss is surrogate function ratio, R
            # made negative because ascent delta is in the _opposite direction_ of loss
            # clipped to given limits to avoid "reward cliff" run-away estimation
            L = -clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)

            #L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards, epsilon=epsilon, beta=beta)
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L

        # the clipping parameter reduces as time goes on
        epsilon = max(epsilon*epsilon, 0.005)  #.999

        # the regulation term also reduces to decrease exploration after many runs
        beta = max(beta*beta, 0.005) #.995

        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))

        # display some progress every 20 iterations
        if (e+1)%20 ==0 :
            print("Episode: {0:d}, Avg. score: {1:f}".format(e+1,np.mean(total_rewards)))
            print(total_rewards)

        # update progress widget bar
        timer.update(e+1)

    timer.finish()
    return mean_rewards