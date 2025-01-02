from archer.environment import batch_interact_environment
from archer.data import DummyDataset,  ReplayBuffer, DPO_ReplayBuffer
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
                eval_freq: int = 25,
                agent_type: str = "archer",
                decode_f: callable = lambda x: x,
                **kwargs):
    trainer = DPOTrainer(agent=agent,\
                            tokenizer=tokenizer,\
                            accelerator=accelerator,
                            lm_lr = lm_lr,\
                            epochs = actor_epochs,\
                            grad_accum_steps=grad_accum_steps,
                            max_grad_norm=max_grad_norm)
    replay_buffer= DPO_ReplayBuffer(batch_size= batch_size, capacity=capacity)
    all_trajectories = []
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
    #main training loop
    print(">>>start iterations")
    for i in tqdm(range(iterations)):
        # print(">>>Interacting with Environment")
        if accelerator.is_main_process:
            trajectories = generate_trajectories(env = env,
                                                agent=agent,
                                                num_trajectories= rollout_size,
                                                decode_f = decode_f,
                                                use_tqdm=False,
                                                debug=True)
            print('Length :', len(trajectories))
            if len(trajectories) == 0:
                continue
            info = {"rollout.mean": np.mean([d["chosen_reward"] + d['rejected_reward'] for d in trajectories]),\
                    "rollout.chosen": np.max([d["chosen_reward"] for d in trajectories]),\
                    "rollout.rejected": np.min([d["rejected_reward"] for d in trajectories])}
            if (i+1) % eval_freq == 0:
                old_sample = agent.do_sample
                agent.do_sample = False
                eval_trajectories =  batch_interact_environment(agent = agent,\
                                                    tokenizer= tokenizer,\
                                                    env = eval_env,\
                                                    num_trajectories=  max(eval_size, eval_env.bsize),\
                                                    env_idx = env_idx,
                                                    use_tqdm=False,
                                                    decode_f = decode_f,
                                                    debug=True)
                agent.do_sample = old_sample
                info.update({"eval_rollout.mean": np.mean([d[0]["trajectory_reward"] for d in eval_trajectories]),\
                        "eval_rollout.max": np.max([d[0]["trajectory_reward"] for d in eval_trajectories]),\
                        "eval_rollout.min": np.min([d[0]["trajectory_reward"] for d in eval_trajectories]),})
            all_trajectories += trajectories
            data = sum(trajectories, [])
            for t in data:
                replay_buffer.insert(**t)
            info.update({"rollout.reward.mean": np.mean([d["chosen_reward"] + d["rejected_reward"] for d in data]),\
                    "rollout.reward.chosen": np.max([d["chosen_reward"] for d in data]),\
                    "rollout.reward.rejected": np.min([d["rejected_reward"] for d in data])})
            print(">>> Saving Replay Buffer")
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
            torch.save(all_trajectories, os.path.join(save_path, 'trajectories.pt'))
            print(">>> Saved Replay Buffer")
            time.sleep(15)
        else:
            info = {}
        accelerator.wait_for_everyone()
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
        replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
        print("Training")
        info_dpo = trainer.dpo_update(replay_buffer)
        info.update(info_dpo)
        
        
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
        if (i+1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving")
            trainer.save(os.path.join(save_path, 'trainer.pt'))
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
    # return model