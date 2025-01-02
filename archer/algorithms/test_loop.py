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
def test_loop(env,\
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
                use_bc: bool = False,
                loss_type: str = 'sigmoid',
                label_smoothing:float = 0.0,
                ppo_update: bool = False,
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
                            bc_epochs=bc_epochs,
                            loss_type=loss_type,
                            label_smoothing=label_smoothing,
                            use_ppo=ppo_update)
    replay_buffer= DPO_ReplayBuffer(batch_size= batch_size, capacity=capacity)
    bc_buffer = ReplayBuffer(batch_size= batch_size, capacity=capacity)

    
    
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
        info = {}
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
                
        
        print("Training")
        accelerator.wait_for_everyone()
        
        training_step+=1
        if reward_filtering:
            filtered_replay_buffer = filter_top_10_percent(replay_buffer)    
        else:
            filtered_replay_buffer = replay_buffer
            filtered_replay_buffer.batch_size = batch_size
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
            print('filtered_dpo_buffer:', len(filtered_replay_buffer))
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
    # return model