from archer.environment import batch_interact_environment,batch_interact_environment_dpo
from archer.data import DummyDataset,  ReplayBuffer
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
from archer.environment import batch_interact_environment_dpo
def offpolicy_train_loop(env,\
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
                grad_accum_steps: int = 2,\
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
                dpo : bool = False,
                use_mcts: bool = False,
                **kwargs):
    if agent_type.lower() == "chai" or agent_type.lower() == "archer"\
        or agent_type.lower() == "archer_llm":
        trainer = ArcherTrainer(agent=agent,\
                            accelerator=accelerator,\
                                tokenizer=tokenizer,\
                                critic_lr = critic_lr,\
                                lm_lr = lm_lr,\
                                gamma = gamma,\
                                tau = tau,\
                                epochs = epochs,\
                                actor_epochs = actor_epochs,
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm)
    elif agent_type.lower() == "online_filteredbc":
        trainer = BCTrainer(agent=agent,\
                                tokenizer=tokenizer,\
                                accelerator=accelerator,
                                lm_lr = lm_lr,\
                                epochs = actor_epochs,\
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm)
    replay_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
    all_trajectories = []
    print(f"Process rank: {accelerator.process_index}, Main process: {accelerator.is_main_process}")
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
        if i>=0:
            start_time= time.time()
            #여기 딴거쓸때는 수정
            if use_mcts:
                if accelerator.is_main_process:
                    print("Using MCTS")
                trajectories,_ = batch_interact_environment_dpo(agent = agent,\
                                            tokenizer= tokenizer,\
                                            env = env,\
                                            num_trajectories= rollout_size // accelerator.num_processes,\
                                            env_idx = env_idx,
                                            use_tqdm=False,
                                            decode_f = decode_f,
                                            debug=False)
            else:
                trajectories = batch_interact_environment(agent = agent,\
                                            tokenizer= tokenizer,\
                                            env = env,\
                                            num_trajectories= rollout_size // accelerator.num_processes,\
                                            env_idx = env_idx,
                                            use_tqdm=False,
                                            decode_f = decode_f,
                                            debug=False)
            rewards = [d[0]["trajectory_reward"] for d in trajectories]
            #print(rewards)
            info = {"rollout.mean": np.mean([d[0]["trajectory_reward"] for d in trajectories]),\
                    "rollout.max": np.max([d[0]["trajectory_reward"] for d in trajectories]),\
                    "rollout.min": np.min([d[0]["trajectory_reward"] for d in trajectories])}
            #all_trajectories = accelerator.gather(trajectories_)
            #trajectories = all_trajectories
            gather_time = time.time()
            print(f'Gather {len(trajectories)} trajectories')
            print(f"Gather time: {gather_time - start_time}")
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
                                                    debug=False,
                                                    eval_flag=True)
                agent.do_sample = old_sample
                info.update({"eval_rollout.mean": np.mean([d[0]["trajectory_reward"] for d in eval_trajectories]),\
                        "eval_rollout.max": np.max([d[0]["trajectory_reward"] for d in eval_trajectories]),\
                        "eval_rollout.min": np.min([d[0]["trajectory_reward"] for d in eval_trajectories]),})
                gather_time = time.time()
                print(f"Eval time: {time.time() - gather_time}")
            all_trajectories += trajectories
            data = sum(trajectories, [])
            for t in data:
                replay_buffer.insert(**t)
            info.update({"rollout.reward.mean": np.mean([d["reward"] for d in data]),\
                    "rollout.reward.max": np.max([d["reward"] for d in data]),\
                    "rollout.reward.min": np.min([d["reward"] for d in data])})
            #print('Here we go')
            update_time = time.time()
            print(f"Update memory time: {update_time - gather_time}")
            if accelerator.is_main_process:
                print(">>> Saving Replay Buffer")
                torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
                torch.save(all_trajectories, os.path.join(save_path, 'trajectories.pt'))
                print(">>> Saved Replay Buffer")
            accelerator.wait_for_everyone()
            time.sleep(15)
        else:
            info = {}
        accelerator.wait_for_everyone()
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
        replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
        print("Training")
        if 'filtered' in agent_type.lower():
            filtered_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
            episode_rewards = [d[0]["trajectory_reward"] for d in all_trajectories]
            cutoff = np.quantile(episode_rewards, 1 - 0.1)
            print("Episode Reward Cutoff: ", cutoff)
            filtered_trajectories = list(filter(lambda x: x[0]["trajectory_reward"] >= cutoff, all_trajectories))
            data = sum(filtered_trajectories, [])
            for d in data:
                filtered_buffer.insert(**d)
            info.update(trainer.update(filtered_buffer, no_update_actor = (i < warmup_iter)))
            print('finishi update')
        else:
            # data = list(filter(lambda x: x["reward"] >0, data))
            print('start update')
            info.update(trainer.update(replay_buffer, no_update_actor = (i < warmup_iter)))
        opt_time = time.time()
        print(f"Optimization time: {opt_time - update_time}")
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
        if (i+1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving")
            trainer.save(os.path.join(save_path, 'trainer.pt'))
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
    # return model