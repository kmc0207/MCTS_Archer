import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, RobertaModel
import torch.nn as nn
import numpy as np


def add_trajectory_reward(trajectory):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory

def add_mc_return(trajectory, gamma = 0.95):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1]))*gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1 )/ gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards*gamma_matrix, axis = 1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory


# def take_action(agent, tokenizer, observation, decode_f=lambda x: x,
#                 noise_std = 0, temperature = 2.0, do_sample=True):
#     raw_action = decode_f(agent.get_action(observation))
#     raw_action = [a[1:] if a.startswith('\n') else a for a in raw_action]
#     raw_action = [a.split('\n')[0] for a in raw_action]
#     return raw_action


def batch_interact_environment(agent, tokenizer, env, num_trajectories,\
        post_f = lambda x: x, use_tqdm = True, decode_f = lambda x: x,
        env_idx = None,debug=False,eval_flag=False):
    """
    in a bacthed way, interact with the environments  to get a list of trajectories
    [[{"observation":, "next_observation":, "reward":, "done":},...],...]
    post_f: function to add additional attributes to the trajectory
    """
    bsize = env.bsize
    all_trajectories = []
    for num_t in tqdm(range(num_trajectories//bsize), disable = not use_tqdm):
        done = False
        trajectories = [[] for _ in range(bsize)]
        # obs = reset_to(env, 69)
        if eval_flag:
            batch_obs = env.eval_reset()
        else:
            batch_obs = env.reset(idx=env_idx)
        batch_done = [False,]*bsize
        steps = 0
        while not all(batch_done):
            steps += 1
            # print(f"Environment stpes {str(steps)}")
            action = agent.get_action(batch_obs)
            batch_return = env.step(decode_f(action))
            for i,result in zip(range(bsize), batch_return):
                if result is None:
                    continue
                next_obs, r, done = result
                trajectories[i].append({"observation": batch_obs[i], \
                                "next_observation": next_obs, \
                                "reward": r, \
                                "done": done, \
                                "action": action[i]})
                batch_obs[i] = next_obs
                batch_done[i] = done
            # obs = next_obs
        if debug:
            print(trajectories[0][-1]["next_observation"])
        all_trajectories += [post_f(add_mc_return(add_trajectory_reward(trajectory)))\
                              for trajectory in trajectories]
        # breakpoint()
        # trajectories.append(post_f(add_trajectory_reward(trajectory)))
    return all_trajectories


def batch_interact_environment_dpo(agent, tokenizer, env, num_trajectories,\
        post_f = lambda x: x, use_tqdm = True, decode_f = lambda x: x,
        env_idx = None,debug=False,one_split=False):
    """
    in a bacthed way, interact with the environments  to get a list of trajectories
    [[{"observation":, "next_observation":, "reward":, "done":},...],...]
    post_f: function to add additional attributes to the trajectory
    """
    bsize = env.bsize
    word_list_len = len(env.env_list[0].word_list)
    all_trajectories = []
    if one_split:
        split_nums = np.random.randint(1,20)
        split_list = [split_nums for _ in range(bsize)]
    else:
        split_list = [np.random.randint(1,20) for _ in range(bsize)]
        split_list[0] = 20
    batch_traj = []
    for num_t in tqdm(range(num_trajectories//bsize), disable = not use_tqdm):
        done = False
        trajectories = [[] for _ in range(bsize)]
        # obs = reset_to(env, 69)
        if env_idx is None:
            env_idx = np.random.randint(0, word_list_len)
        batch_obs = env.reset(idx=env_idx)
        batch_done = [False,]*bsize
        steps = 0
        while not all(batch_done):
            steps += 1
            # print(f"Environment stpes {str(steps)}")
            action_before = agent.get_action(batch_obs)
            new_action = []
            #print('Action : ',action)
            for i in range(len(action_before)):
                if split_list[i] > steps:
                    if batch_done[i] or batch_done[0]:
                        continue
                    else:
                        #print(f'Action {i} : {action[i]}')
                        #print(f'Is changed to {action[0]}')
                        new_action.append(action_before[0])
                else:
                    new_action.append(action_before[i])
            action = new_action
            #print('After Action : ',action)
            batch_return = env.step(decode_f(action))
            for i,result in zip(range(bsize), batch_return):
                if result is None:
                    continue
                next_obs, r, done = result
                trajectories[i].append({"observation": batch_obs[i], \
                                "next_observation": next_obs, \
                                "reward": r, \
                                "done": done, \
                                "action": action[i],
                                'split': split_list[i],
                                'batch_num': num_t})
                batch_obs[i] = next_obs
                batch_done[i] = done
            # obs = next_obs
            batch_traj.append(trajectories)
        if debug:
            print('N1\n', trajectories[0][-1]["next_observation"], '\n', trajectories[0][-1]["reward"], '\n', trajectories[0][-1]["split"])
            print('N2\n', trajectories[1][-1]["next_observation"], '\n', trajectories[1][-1]["reward"], '\n', trajectories[1][-1]["split"])
        all_trajectories += [post_f(add_mc_return(add_trajectory_reward(trajectory)))\
                              for trajectory in trajectories]
        # breakpoint()
        # trajectories.append(post_f(add_trajectory_reward(trajectory)))
    return all_trajectories,batch_traj
