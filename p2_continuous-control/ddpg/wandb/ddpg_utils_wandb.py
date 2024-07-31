

scalenp = lambda x: (x - np.min(x))/(np.max(x) - np.min(x))
normnp = lambda x: 0. if np.std(x)==0 else (x - np.mean(x))/np.std(x) 
        
def 



def wb_log(score, individual_scores):
       wandb.log( {"episode_scores":score,
        "agent_scores":individual_scores,
        "cycle_steps":agency.steps%ddpg_agent.BUFFER_SIZE,
        "buffer_cycle":agency.steps//ddpg_agent.BUFFER_SIZE,}
                )