import time
from ddpg_agent import *

def train(env, agency, n_episodes=2000, max_t=1000, hiscore=30):
    """Deep Q-Learning for a Continuous Action Space
    
    Params
    ======
        env (environment): Here, a Unity Env. Originally, a Gym(nasium) env
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        hiscore (int): the metric score to achieve to solve the env 
    """
    FIRST = True
    start = time.time()
    episode_times = []; episode_lengths = []; scores = []; agent_scores = []
    action_steps = []; noise_steps= []; actor_loss = []; critic_loss = []
    scores_window = deque(maxlen=100)
    
    for i_episode in range(1, n_episodes+1):
        epistart = time.time()
        score = 0. ; episteps = 0.; episcores = np.zeros(agency.num_agents)
        a_loss = 0. ; c_loss = 0.
        env_info = env.reset(train_mode=True)['ReacherBrain']
        states = env_info.vector_observations
        agency.reset()
        for t in range(max_t):
            actions, noises = agency.act(states, return_noise=True)
            action_steps.append(actions)
            noise_steps.append(noises)
            actions = np.clip(actions+noises, -1, 1)
            env_info = env.step(actions)['ReacherBrain']
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            al, cl = agency.step(states, actions, rewards, next_states, dones,
                                 return_loss=True)
            if al != None: 
                a_loss += al; c_loss += cl; 
            states = next_states; episteps+=1
            score += np.mean(rewards); episcores += rewards
            if np.any(dones):
                break  
        # Data appendage
        scores_window.append(scores)       
        scores.append(scores)
        episode_lengths.append(episteps)
        actor_loss.append(a_loss/episteps)          
        critic_loss.append(c_loss/episteps)
        #if np.abs(actor_loss[-1]) > 1e9 or np.abs(critic_loss[-1]) > 1e9: break
        episode_times.append(time.time()-epistart)
        cycle_steps = agency.steps%BUFFER_SIZE
        buffer_cycle = agency.steps//BUFFER_SIZE

        print("\rEpisode {:4d} | Score:{:8.2f} | Actor Loss:{:8.2f} | Critic Loss:{:8.2f} | Episode: {:4d} Steps in {:5.3f} sec | Memory Buffer:{:7d} into cycle {:3d}".format(i_episode,
                                                                                                                                                                                score, actor_loss[-1], critic_loss[-1],
                                                                                                                                                                                episteps, time.time()-epistart,
                                                                                                                                                                                cycle_steps, buffer_cycle))
        #print("\r{:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f}".format(*individual_scores), end="")
                                                                                                                                        
        if i_episode % 10 == 0:
            chkpntname = "checkpoints/actor_chkpnt{}.pth".format(i_episode)
            torch.save(agent.actor_target.state_dict(), chkpntname)  
            chkpntname = "checkpoints/critic_chkpnt{}.pth".format(i_episode)
            torch.save(agent.critic_target.state_dict(), chkpntname) 
            print("\rEpisode {:4d} | Score: {:8.2f} | Actor Loss: {:8.2f} | Critic Loss: {:8.2f} | Average: {:5.1f} Steps in {:5.3f} sec | Memory Buffer:{:7d} into cycle {:3d}".format(i_episode, 
                                                                                                                                                                                    np.mean(scores_window), np.mean(actor_loss), np.mean(critic_loss),
                                                                                                                                                                                    np.mean(episode_lengths), np.mean(episode_times),
                                                                                                                                                                                    cycle_steps, buffer_cycle))
        #print("\r{:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f} | {:4.2f}".format(*individual_scores))
        
        if np.mean(scores_window)>=hiscore: # and FIRST:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:5.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.fast_actor.state_dict(), 'checkpoints/actor_slvdpnt.pth')
            torch.save(agent.fast_critic.state_dict(), 'checkpoints/critic_slvdpnt.pth')
        #    FIRST = False
        #elif np.mean(scores_window)>=1.25*hiscore:
        #    print("\n***** High Score! *****")
        #    print("\tGame over.")
        #    torch.save(agent.fast_actor.state_dict(), 'data/actor_hipnt.pth')
        #    torch.save(agent.fast_critic.state_dict(), 'data/critic_hipnt.pth')
            break
            
    t_time = time.time() - start
    print("Total time: {:3d} minutes {:4.2f} seconds \tAvg. Episode time: {:5.3f} seconds \tAvg. Episode steps: {:5.3f}".format(int(t_time//60),
                                                                                                                                   t_time%60,
                                                                                                                                   t_time/i_episode,
                                                                                                                                   agent.steps/i_episode))
    return scores, episode_lengths, episode_times, action_steps, noise_steps, actor_loss, critic_loss