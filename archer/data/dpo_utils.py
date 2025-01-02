from torch.utils.data import Dataset, DataLoader
import numpy as np



class DPO_ReplayBuffer:
    def __init__(self, batch_size=2, capacity=10000):
        self.max_size = capacity
        self.size = 0
        self.batch_size = batch_size

        # 문자열 데이터를 저장할 배열
        self.observations = np.empty(self.max_size, dtype=object)  # 문자열 저장
        self.chosen_actions = np.empty(self.max_size, dtype=object)
        self.rejected_actions = np.empty(self.max_size, dtype=object)
        
        # 숫자 데이터를 저장할 배열
        self.chosen_rewards = np.empty(self.max_size, dtype=np.float32)
        self.rejected_rewards = np.empty(self.max_size, dtype=np.float32)
        
    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observation": self.observations[rand_indices],
            "chosen_reward": self.chosen_rewards[rand_indices],
            "rejected_reward": self.rejected_rewards[rand_indices],
            "chosen_action": self.chosen_actions[rand_indices],
            "rejected_action": self.rejected_actions[rand_indices],
        }

    def __len__(self):
        return self.size

    def insert(
        self,
        observation: str,  # 문자열
        chosen_action: str,  # 문자열
        rejected_action: str,  # 문자열
        chosen_reward: float,  # 숫자
        rejected_reward: float,  # 숫자
        **kwargs
    ):
        """
        ReplayBuffer에 데이터 삽입
        """
        idx = self.size % self.max_size

        # 문자열 그대로 저장
        self.observations[idx] = observation  # ex: "Who are you?"
        self.chosen_actions[idx] = chosen_action  # ex: "I am a boy"
        self.rejected_actions[idx] = rejected_action  # ex: "good for you"
        
        # 숫자 데이터 저장
        self.chosen_rewards[idx] = chosen_reward  # ex: 0.01
        self.rejected_rewards[idx] = rejected_reward  # ex: -10.0

        self.size += 1
        
        
import numpy as np
from collections import Counter

def filter_top_10_percent(buffer: DPO_ReplayBuffer) -> DPO_ReplayBuffer:
    """
    기존 DPO_ReplayBuffer에서 chosen_reward 기준 상위 10%만 포함하는 새로운 DPO_ReplayBuffer를 생성합니다.
    
    Args:
        buffer (DPO_ReplayBuffer): 기존 ReplayBuffer
    
    Returns:
        DPO_ReplayBuffer: 상위 10% trajectory만 포함된 새로운 ReplayBuffer
    """
    # chosen_reward 기준 상위 10% 경계값 계산
    threshold = np.percentile(buffer.chosen_rewards[:buffer.size], 90)
    print('Threshold:', threshold)
    
    # chosen_reward - rejected_reward 계산
    reward_differences = buffer.chosen_rewards[:buffer.size] - buffer.rejected_rewards[:buffer.size]
    print('Reward Differences Sample:', reward_differences[:10])  # 샘플 출력
    
    # observation 안의 '\n' 개수 계산
    newline_counts = [obs.count('\n') if isinstance(obs, str) else 0 for obs in buffer.observations[:buffer.size]]
    print('Newline Counts Sample:', newline_counts[:10])  # 샘플 출력
    
    # chosen_reward - rejected_reward와 newline_counts 간 상관관계 계산
    correlation = np.corrcoef(reward_differences, newline_counts)[0, 1]
    print(f'Correlation between reward differences and newline counts: {correlation:.4f}')
    
    # 상위 10% 인덱스 추출
    top_indices = np.where(buffer.chosen_rewards[:buffer.size] >= threshold)[0]
    
    # 상위 10%에서 observation의 '\n' 개수 분포 확인
    top_newline_counts = [newline_counts[idx] for idx in top_indices]
    distribution = Counter(top_newline_counts)
    print('Newline Counts Distribution in Top 10%:', dict(distribution))
    
    # 새로운 DPO_ReplayBuffer 생성
    new_buffer = DPO_ReplayBuffer(batch_size=buffer.batch_size, capacity=len(top_indices))
    
    # 상위 10% trajectory 추가
    for idx in top_indices:
        new_buffer.insert(
            observation=buffer.observations[idx],
            chosen_action=buffer.chosen_actions[idx],
            rejected_action=buffer.rejected_actions[idx],
            chosen_reward=buffer.chosen_rewards[idx],
            rejected_reward=buffer.rejected_rewards[idx],
        )
    
    return new_buffer