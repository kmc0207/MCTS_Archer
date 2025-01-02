import torch
import transformers
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, RobertaModel



def rollout_from_state(env, agent, start_obs, decode_f, max_steps=1000):
    """
    주어진 환경 상태 및 observation에서 시작해서 done 될 때까지 rollout하며 reward의 합을 반환하는 함수.
    """
    obs = start_obs
    total_reward = 0.0
    steps = 0
    done = False
    while (not done) and steps < max_steps:
        action = agent.get_action([obs])  # single obs as list
        action = decode_f(action)[0]
        next_obs, r, done = env.step(action)[0] # env.step(action) -> [(next_obs, reward, done), ...] 라 가정
        total_reward += r
        obs = next_obs
        steps += 1
    return total_reward


def generate_trajectories(env, agent, num_trajectories, N=None, 
                          decode_f=lambda x: x,
                          use_tqdm=False,
                          debug=False):
    """
    N 스텝까지는 동일하게 따라간 다음, N+1 스텝에서 두 가지 action을 시도해보고,
    각각 끝까지 돌려본 후 더 나은 action을 chosen, 다른 action을 rejected로 결정하는 형태.
    
    return: 리스트 형태로 [(observation_at_step_N, (chosen_action, rejected_action)), ...]
    """
    results = []
    iterator = range(num_trajectories)
    if use_tqdm:
        from tqdm import tqdm
        iterator = tqdm(iterator)

    for _ in iterator:
        # 초기화
        obs = env.reset()
        done = False
        steps = 0

        # N 스텝 전까지 단일 action으로 진행
        saved_states = []
        if N is None:
            N = np.random.randint(1, 15)
        for _n in range(N):
            if done:
                # 만약 N 이전에 done이면 그냥 종료
                break
            action = agent.get_action(obs)  # batch_size=1
            action = decode_f(action)[0]
            next_obs, r, done = env.step(action)[0]
            obs = next_obs
            if debug:
                print(f"Action: {action}, Step: {_n}, Reward: {r}")
        
        
        # 여기서 env의 현재 상태 저장 (branching 위해)
        state_at_N = env.get_state()
        obs_at_N = obs
        if debug:
            print(f'Branching at step {N}')

        # N+1 스텝에서 두 개의 서로 다른 action 시도
        # 만약 agent가 stochastic하게 행동한다면, 단순히 두 번 호출해서 다른 action을 얻을 수도 있음.
        # action이 같게 나오면 다시 시도하는 로직을 넣을 수도 있음.

        candidate_actions = []
        while len(candidate_actions) < 2:
            candidate_action = agent.get_action([obs_at_N])
            candidate_action = decode_f(candidate_action)[0]
            if candidate_action not in candidate_actions:
                candidate_actions.append(candidate_action)
        if debug:
            print(f'Candidate actions: {candidate_actions}')
        action1, action2 = candidate_actions

        
        if debug:
            print(f'Start N1')
        # 첫 번째 action으로 롤아웃
        env.set_state(state_at_N)  # state 복원
        env_out = env.step(action1)[0] 
        if env_out is not None:
            obs1, r1, done1 = env_out
            if debug:
                print(f'Action1: {action1}, Reward: {r1}')
        else:
            # 혹은 obs_at_N 그대로 유지
            obs1, r1, done1 = obs_at_N, 0, True
        total_reward_1 = r1
        if not done1:
            total_reward_1 += rollout_from_state(env, agent, obs1, decode_f)
        if debug:
            print(f'Start N2')
            
        # 두 번째 action으로 롤아웃
        env.set_state(state_at_N)  # state 복원
        env_out = env.step(action2)[0]
        if env_out is not None:
            obs2, r2, done2 = env_out
            if debug:
                print(f'Action2: {action2}, Reward: {r2}')
        else:
            obs2, r2, done2 = obs_at_N, 0, True
        total_reward_2 = r2
        if not done2:
            total_reward_2 += rollout_from_state(env, agent, obs2, decode_f)

        print(f'Action1: {action1}, Reward: {total_reward_1}, Action2: {action2}, Reward: {total_reward_2}')
        # 비교 후 chosen / rejected 결정
        if total_reward_1 > total_reward_2:
            chosen_action = action1
            rejected_action = action2
        elif total_reward_1 < total_reward_2:
            chosen_action = action2
            rejected_action = action1
        else:
            continue
        final_dict ={
            'observation_at_N': obs_at_N,
            'chosen_action': chosen_action,
            'rejected_action': rejected_action,
            'chosen_reward': total_reward_1,
            'rejected_reward': total_reward_2,
        }
        results.append(final_dict)

    return results