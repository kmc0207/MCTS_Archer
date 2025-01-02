from archer.environment import batch_interact_environment,batch_interact_environment_dpo
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from archer.algorithms.archer import ArcherTrainer
from archer.algorithms.online_filteredbc import BCTrainer
import wandb
import threading
import os
import torch
import time
from archer.environment import batch_interact_environment
from archer.data import DummyDataset,  ReplayBuffer, DPO_ReplayBuffer, filter_top_10_percent
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from archer.algorithms.archer import ArcherTrainer
from archer.algorithms.online_filteredbc import BCTrainer
from archer.algorithms.dpo import DPOTrainer
import wandb
import threading
import os
import torch
import time
from archer.environment import generate_trajectories
def dpo_train_loop(env,\
                eval_env,\
                agent,\
                tokenizer,\
                accelerator,\
                warmup_iter: int = 20,
                rollout_size: int = 50,\
                eval_size: int = 1,
                batch_size: int = 2,
                capacity: int = 500000,
                iterations: int = 10,\
                epochs:int = 3, \
                grad_accum_steps: int = 1,\
                env_idx:int = None,\
                do_sample: bool = False,\
                temperature: float = 2.0,\
                critic_lr: float= 1e-3,\
                lm_lr: float = 1e-5,\
                gamma: float = 0.9,
                tau: float = 0.1,
                use_wandb: bool = False,
                env_load_path: str = '',
                actor_epochs: int = 3,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                save_freq: int = 25,
                eval_freq: int = 2,
                agent_type: str = "archer",
                decode_f: callable = lambda x: x,
                dpo : bool = False,
                clear : bool = False,
                use_ref : bool = False,
                use_mix : bool = False,
                bc_weight: float = 1.0,
                dpo_weight: float = 1.0,
                update_ref : bool = False,
                stable_update : bool = False,
                reward_filtering : bool = False,
                update_term : int = 1,
                beta: float = 0.5,
                bc_explore: bool = False,
                clear_freq: int = 10,
                bc_rollout_size: int = 64,
                one_split: bool = False,
                bc_epochs: int = 3,
                bc_from_dpo: bool = False,
                get_dict: bool = False,
                ppo_update: bool = False,
                use_bc: bool = False,
                **kwargs):
    trainer = DPOTrainer(agent=agent,\
                            tokenizer=tokenizer,\
                            accelerator=accelerator,
                            lm_lr = lm_lr,\
                            epochs = actor_epochs,\
                            grad_accum_steps=grad_accum_steps,
                            max_grad_norm=max_grad_norm,
                            use_ref=use_ref,
                            beta=beta,
                            bc_epochs=bc_epochs)
    replay_buffer= DPO_ReplayBuffer(batch_size= batch_size, capacity=capacity)
    bc_buffer = ReplayBuffer(batch_size= batch_size, capacity=capacity)

    all_trajectories = []
    bc_trajectories = []
    
    
    if get_dict:
            all_trajectories = torch.load(os.path.join(save_path, 'trajectories_dpo.pt'))
            replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer_dpo.pt'))
            bc_trajectories = torch.load(os.path.join(save_path, 'bc_trajectories.pt'))
            bc_buffer = torch.load(os.path.join(save_path, 'bc_buffer.pt'))    

    last_eval=-200
    new_eval=-200
    stable_update_time =0
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(save_path, 'trainer.pt')):
            # print("Not using existing checkpoint")
            print("Loading from checkpoint")
            trainer.load(os.path.join(save_path, 'trainer.pt'))
            all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
            replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
        else:
            print("Creating new checkpoint directory")
            os.makedirs(save_path, exist_ok=True)
    agent.prepare()
    training_step = 0

    last_training_buffer_size=0
    print(f'Env batch size: {env.bsize}')
    print(f'Gradient accumulation steps: {trainer.grad_accum_steps}')
    print(f'Replay buffer batch size: {replay_buffer.batch_size}')
    #main training loop
    print(">>>start iterations")
    for i in tqdm(range(iterations)):
        # print(">>>Interacting with Environment")
        start_time = time.time()
        if accelerator.is_main_process:
            #여기 딴거쓸때는 수정
            #agent.temperature = 1.5
            trajectories,batch_traj = batch_interact_environment_dpo(agent = agent,\
                                        tokenizer= tokenizer,\
                                        env = env,\
                                        num_trajectories= rollout_size,\
                                        env_idx = env_idx,
                                        use_tqdm=False,
                                        decode_f = decode_f,
                                        debug=False,
                                        one_split=one_split)
            rewards = [d[0]["trajectory_reward"] for d in trajectories]
            #print(rewards)
            info = {"rollout.dpo.mean": np.mean([d[0]["trajectory_reward"] for d in trajectories]),\
                    "rollout.dpo.max": np.max([d[0]["trajectory_reward"] for d in trajectories]),\
                    "rollout.dpo.min": np.min([d[0]["trajectory_reward"] for d in trajectories])}
            
            if bc_explore:
                bc_batch_trajectories = batch_interact_environment(agent = agent,\
                                        tokenizer= tokenizer,\
                                        env = env,\
                                        num_trajectories= bc_rollout_size,\
                                        env_idx = env_idx,
                                        use_tqdm=False,
                                        decode_f = decode_f,
                                        debug=False)
                bc_rewards = [d[0]["trajectory_reward"] for d in bc_batch_trajectories]
                info.update({"rollout.mean": np.mean([d[0]["trajectory_reward"] for d in bc_batch_trajectories]),\
                    "rollout.max": np.max([d[0]["trajectory_reward"] for d in bc_batch_trajectories]),\
                    "rollout.min": np.min([d[0]["trajectory_reward"] for d in bc_batch_trajectories])})
            
            
            gather_time = time.time()
            print(f'Gather time: {gather_time - start_time}')
            
            
            #eval
            if (i+1) % eval_freq == 0:
                print('Start Evaluation')
                #agent.temperature = 1.0
                old_sample = agent.do_sample
                #agent.do_sample = False
                eval_env_idx = None
                eval_trajectories =  batch_interact_environment(agent = agent,\
                                                    tokenizer= tokenizer,\
                                                    env = eval_env,\
                                                    num_trajectories=  max(eval_size, eval_env.bsize),\
                                                    env_idx = eval_env_idx,
                                                    use_tqdm=False,
                                                    decode_f = decode_f,
                                                    debug=False,
                                                    eval_flag=True)
                
                for num_env,tr in enumerate(eval_trajectories):
                    print('Reward:', tr[0]['trajectory_reward'])
                    print('Next Observation:', tr[-1]['next_observation'])
                    print('Answer:', eval_env.env_list[num_env].curr_word)
                eval_mean = np.mean([d[0]["trajectory_reward"] for d in eval_trajectories])
                print(f'Evaluation mean: {eval_mean}')
                if stable_update:
                    if eval_mean > last_eval:
                        last_eval=eval_mean
                        update_ref=True
                        stable_update_time +=1
                        info.update({"stable update time":stable_update_time})
                agent.do_sample = old_sample
                info.update({"eval_rollout.mean": np.mean([d[0]["trajectory_reward"] for d in eval_trajectories]),\
                        "eval_rollout.max": np.max([d[0]["trajectory_reward"] for d in eval_trajectories]),\
                        "eval_rollout.min": np.min([d[0]["trajectory_reward"] for d in eval_trajectories]),})
            #eval_end
            



            all_trajectories += trajectories
            data = sum(trajectories, [])
            #bc_trajectories += trajectories
            #print(trajectories)
            batch_reward_mean= 0
            batch_max_reward = -200
            batch_max_data = []
            batch_max_length =0 
            if not bc_explore and not bc_from_dpo:
                #bc_trajectories += trajectories
                #print(len(batch_traj))
                for tr in batch_traj:
                    #print(len(tr))
                    for t in tr:
                        #print(len(t))
                        if t[0]['trajectory_reward'] > batch_max_reward:
                            batch_max_reward = t[0]['trajectory_reward']
                            batch_max_data = [t]
                        elif t[0]['trajectory_reward'] == batch_max_reward:
                            batch_max_data.append(t)
                        bc_trajectories += batch_max_data
                        batch_reward_mean += t[0]['trajectory_reward']
                        batch_max_length +=1
                        batch_max_data = []
                        batch_max_reward = -200
                batch_reward_mean = batch_reward_mean/batch_max_length
                        
                print('batch_reward_mean:', batch_reward_mean)
                info.update({"rollout.mean": batch_reward_mean})
            
            if bc_explore:
                bc_trajectories += bc_batch_trajectories
            
            
            
            cur_batch = []  # 현재 배치를 저장할 리스트
            lenghts = []
            rewards = []
            print(f'Length of  trajectories: {len(trajectories)}')
            print(f'Length of bc_trajectories: {len(bc_trajectories)}')
            update_num=0
            for tr in trajectories:
                lenghts.append(len(tr))
                rewards.append(tr[0]['trajectory_reward'])
            
            if one_split:
                for num_t in range(rollout_size//batch_size):
                    root = trajectories[num_t*batch_size]
                    batch_max_reward = root[0]['trajectory_reward']
                    split_number = root[0]['split']
                    if split_number >= len(root):
                        print('split_number > len(root)')
                        continue
                    batch_max_data = root[split_number]
                    batch_max_traj = root
                    batch_min_reward = root[0]['trajectory_reward']
                    batch_min_data = root[split_number]
                    #root_action = root[split_number]['action']
                    
                    

                    
                    for num_b in range(1,batch_size):
                        if num_t*batch_size + num_b >= rollout_size:
                            break
                        split_number = trajectories[num_t*batch_size + num_b][0]['split']
                        if split_number >= len(trajectories[num_t*batch_size + num_b]):
                            print('split_number > len(now)')
                            continue
                        cur_now = trajectories[num_t*batch_size + num_b][split_number]
                        cur_observation = cur_now['observation']
                        cur_reward = cur_now['trajectory_reward']
                        cur_action = cur_now['action']
                        if cur_reward > batch_max_reward:
                            if cur_action == batch_max_data['action']:
                                continue
                            batch_max_reward = cur_reward
                            batch_max_data = cur_now
                            batch_max_traj = trajectories[num_t*batch_size + num_b]
                        if cur_reward < batch_min_reward:
                            if cur_action == batch_min_data['action']:
                                continue
                            batch_min_reward = cur_reward
                            batch_min_data = cur_now
                    if batch_max_data['action'] == batch_min_data['action']:
                        continue
                    print('observation:', cur_observation)
                    print('max_reward:', batch_max_reward)
                    print('min_reward:', batch_min_reward)
                    print('max_action:', batch_max_data['action'])
                    print('min_action:', batch_min_data['action'])
                    replay_buffer.insert(observation = cur_observation,\
                                        chosen_action = batch_max_data['action'],\
                                        rejected_action = batch_min_data['action'],\
                                        chosen_reward = batch_max_data['trajectory_reward'],\
                                        rejected_reward = batch_min_data['trajectory_reward'])

                    update_num+=1
                        

                        
                
            else:
                rollout_mean = 0
                for num_t in range(rollout_size//batch_size):
                    root = trajectories[num_t*batch_size]
                    #print(f'Length of root : {len(root)}')
                    for num_b in range(1,batch_size):
                        
                        if num_t*batch_size + num_b >= rollout_size:
                            break
                        split_number = trajectories[num_t*batch_size + num_b][0]['split']
                        if split_number >= len(root):
                            print('split_number > len(root)')
                            continue
                        if split_number >= len(trajectories[num_t*batch_size + num_b]):
                            print('split_number > len(now)')
                            continue
                        root_now = root[split_number]
                        
                        now_not_number = trajectories[num_t*batch_size + num_b]
                        now = now_not_number[split_number]
                        now_observation = now['observation']
                        if now['action'] == root_now['action']:
                            continue
                        if now['trajectory_reward'] > root_now['trajectory_reward']:
                            chosen_action = now['action']
                            chosen_reward = now['trajectory_reward']
                            rejected_action = root_now['action']
                            rejected_reward = root_now['trajectory_reward']
                            choosed_traj = now_not_number
                        elif now['trajectory_reward'] < root_now['trajectory_reward']:
                            chosen_action = root_now['action']
                            chosen_reward = root_now['trajectory_reward']
                            rejected_action = now['action']
                            rejected_reward = now['trajectory_reward']
                            choosed_traj = root
                        else:
                            continue
                        print('observation:', now_observation)
                        print('chosen_action:', chosen_action)
                        print('rejected_action:', rejected_action)
                        print('chosen_reward:', chosen_reward)
                        print('rejected_reward:', rejected_reward)
                        print('length of choosed trajectory:', len(choosed_traj))
                        replay_buffer.insert(observation = now_observation,\
                                            chosen_action = chosen_action,\
                                            rejected_action = rejected_action,\
                                            chosen_reward = chosen_reward,\
                                            rejected_reward = rejected_reward)
                        if bc_from_dpo:
                            bc_trajectories.append(choosed_traj)
                            rollout_mean += choosed_traj[0]['trajectory_reward']
                        update_num+=1
            
            
            
                        
            info.update({"rollout.reward.mean": np.mean([d["reward"] for d in data]),\
                    "rollout.reward.max": np.max([d["reward"] for d in data]),\
                    "rollout.reward.min": np.min([d["reward"] for d in data])})
            if update_num !=0:
                info.update({"rollout.mean": rollout_mean/update_num})
            print('Length of replay_buffer:', len(replay_buffer))
            
            
            
            if len(replay_buffer) > trainer.grad_accum_steps*replay_buffer.batch_size*warmup_iter:
                print(">>> Saving Replay Buffer")
                if not get_dict:
                    torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer_dpo.pt'))
                    torch.save(all_trajectories, os.path.join(save_path, 'trajectories_dpo.pt'))
                    torch.save(bc_trajectories, os.path.join(save_path, 'bc_trajectories.pt'))
                    torch.save(bc_buffer, os.path.join(save_path, 'bc_buffer.pt'))
                    print(">>> Saved Replay Buffer")
                    time.sleep(5)
                    print('batch_reward_mean:', batch_reward_mean)
                
        else:
            info = {}
        
        if update_num < 2:
            print('No update')
        
        if len(replay_buffer) > trainer.grad_accum_steps*replay_buffer.batch_size*warmup_iter :
            print("Training")
        
            last_training_buffer_size = len(replay_buffer)
            accelerator.wait_for_everyone()
            #all_trajectories = torch.load(os.path.join(save_path, 'trajectories_dpo.pt'))
            #replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer_dpo.pt'))
            #bc_trajectories = torch.load(os.path.join(save_path, 'bc_trajectories.pt'))
            #bc_buffer = torch.load(os.path.join(save_path, 'bc_buffer.pt'))
            
            training_step+=1
            if reward_filtering:
                filtered_replay_buffer = filter_top_10_percent(replay_buffer)    
            else:
                filtered_replay_buffer = replay_buffer
            if use_mix :
                
                filtered_buffer = ReplayBuffer(batch_size= batch_size, capacity=capacity)
                episode_rewards = [d[0]["trajectory_reward"] for d in bc_trajectories]
                cutoff = np.quantile(episode_rewards, 1 - 0.1)
                
                filtered_trajectories = list(filter(lambda x: x[0]["trajectory_reward"] >= cutoff, bc_trajectories))
                print("Episode Reward Cutoff: ", cutoff)
                info.update({"Episode Reward Cutoff": cutoff})
                print('before:', len(bc_trajectories))
                #print('filtered_trajectories:', len(filtered_trajectories))
                data = sum(filtered_trajectories, [])
                for d in data:
                    filtered_buffer.insert(**d)
                
                
                print('filtered_buffer:', len(filtered_buffer))
                print('filtered_dpo_buffer:', len(filtered_replay_buffer))
                
                if training_step % update_term and update_ref == 0:
                    update_ref_=True
                else:
                    update_ref_=False
                
                info_bc = trainer.combined_update(filtered_buffer, 
                                                  filtered_replay_buffer,
                                                  bc_weight=bc_weight, 
                                                  dpo_weight=dpo_weight,
                                                  update_ref=update_ref_)
                if stable_update:
                    update_ref=False
                    
                info.update(info_bc)
            elif use_bc:
                filtered_buffer = ReplayBuffer(batch_size= batch_size, capacity=capacity)
                episode_rewards = [d[0]["trajectory_reward"] for d in bc_trajectories]
                cutoff = np.quantile(episode_rewards, 1 - 0.1)
                
                filtered_trajectories = list(filter(lambda x: x[0]["trajectory_reward"] >= cutoff, bc_trajectories))
                print("Episode Reward Cutoff: ", cutoff)
                info.update({"Episode Reward Cutoff": cutoff})
                print('before:', len(bc_trajectories))
                #print('filtered_trajectories:', len(filtered_trajectories))
                data = sum(filtered_trajectories, [])
                for d in data:
                    filtered_buffer.insert(**d)
                
                
                print('filtered_buffer:', len(filtered_buffer))
                #print('filtered_dpo_buffer:', len(filtered_replay_buffer))
                info_bc = trainer.bc_update(filtered_buffer)
                info.update(info_bc)
            elif ppo_update:
                filtered_buffer = ReplayBuffer(batch_size= batch_size, capacity=capacity)
                episode_rewards = [d[0]["trajectory_reward"] for d in bc_trajectories]
                cutoff = np.quantile(episode_rewards, 1 - 0.1)
                
                filtered_trajectories = list(filter(lambda x: x[0]["trajectory_reward"] >= cutoff, bc_trajectories))
                print("Episode Reward Cutoff: ", cutoff)
                info.update({"Episode Reward Cutoff": cutoff})
                print('before:', len(bc_trajectories))
                #print('filtered_trajectories:', len(filtered_trajectories))
                data = sum(filtered_trajectories, [])
                for d in data:
                    filtered_buffer.insert(**d)
                
                
                print('filtered_buffer:', len(filtered_buffer))
                print('filtered_dpo_buffer:', len(filtered_replay_buffer))
                info_bc = trainer.reward_update(filtered_buffer)
                info.update(info_bc)            
            else:
                info_dpo = trainer.dpo_update(filtered_replay_buffer,update_ref=update_ref)
                info.update(info_dpo)
                if clear and training_step % clear_freq == 0:
                    print('Clear')
                    all_trajectories = []
                    bc_trajectories = []
                    
                    replay_buffer = DPO_ReplayBuffer(batch_size= batch_size, capacity=capacity)
        if use_wandb and accelerator.is_main_process:
            info.update({"DPO_Buffer_size": len(replay_buffer)})
            info.update({"BC_Buffer_size": len(bc_trajectories)})
            info.update({"training_step": training_step})
            wandb.log(info)
            #wandb.log({"DPO_Buffer_size": len(replay_buffer)})
            #wandb.log({"BC_Buffer_size": len(bc_buffer)})
            #wandb.log({"training_step": training_step})
        if (i+1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving")
            trainer.save(os.path.join(save_path, 'trainer.pt'))
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
    # return model