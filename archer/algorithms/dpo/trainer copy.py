import torch
import torch.nn.functional as F
import copy
import torch
import transformers
from tqdm import tqdm
from archer.algorithms.bc import plain_bc_loss
import copy
import random
from torch.utils.data import DataLoader
from archer.data import DummyDataset
def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
def text_collate_fn(batch):
    # 텍스트 데이터를 그대로 반환
    return batch
def custom_collate_fn(batch):
    observations = [item["observation"] for item in batch]  # 텍스트 리스트
    chosen_actions = [item["chosen_action"] for item in batch]  # 텍스트 리스트
    rejected_actions = [item["rejected_action"] for item in batch]  # 텍스트 리스트

    return {
        "observation": observations,
        "chosen_action": chosen_actions,
        "rejected_action": rejected_actions,
    }
def get_log_prob(logits, input_ids, mask=None):
    # logits: [batch_size, seq_len, vocab_size]
    # input_ids: [batch_size, seq_len]
    # mask: [batch_size, seq_len], where 1 indicates positions to include in loss calc.

    # gather the log probabilities at the positions of the input_ids
    log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
    gathered_log_probs = torch.gather(log_probs, -1, input_ids.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]

    if mask is not None:
        # mask를 적용해서 prompt 부분은 제외
        gathered_log_probs = gathered_log_probs * mask
        # 평균을 낼 때 mask가 0인 부분은 제외하기 위해 sum(mask)로 나눈다.
        denom = mask.sum(dim=-1, keepdim=True)
        # division by zero 방지
        denom = torch.clamp(denom, min=1e-9)
        avg_log_prob = (gathered_log_probs.sum(dim=-1, keepdim=True) / denom).squeeze(-1)
    else:
        avg_log_prob = gathered_log_probs.mean(dim=-1)

    return avg_log_prob


def calculate_DPO_loss(
    model_preferred_logprob,
    model_dispreferred_logprob,
    ref_preferred_logprob,
    ref_dispreferred_logprob,
    beta=0.1,
    use_ref=False,
    loss_type="sigmoid",       # ["sigmoid", "hinge"]
    label_smoothing=0.0        # 0.0 ~ 0.2 정도 권장
):
    """
    model_preferred_logprob, model_dispreferred_logprob: [batch]
    ref_preferred_logprob, ref_dispreferred_logprob: [batch]
    beta: DPO에서 사용하는 스케일 파라미터 (0.1~0.5)
    use_ref: ref 모델 사용 여부
    loss_type: "sigmoid" or "hinge" (필요시 추가 가능)
    label_smoothing: 시그모이드 DPO 로스에 라벨 스무딩을 적용할 때 사용

    Returns:
      loss: 스칼라 로스 (tensor)
      p_rel: model vs ref 차이 (preferred)
      d_rel: model vs ref 차이 (dispreferred)
      reward_accuracies: (relative_logprob > 0) 정확도
      reward_margins: margin 평균값
    """
    # 1) reference를 사용하는 경우, 모델 로그확률 - ref 로그확률 형태로 만든다
    if use_ref:
        prefered_relative_logprob = model_preferred_logprob - ref_preferred_logprob
        disprefered_relative_logprob = model_dispreferred_logprob - ref_dispreferred_logprob
    else:
        prefered_relative_logprob = model_preferred_logprob
        disprefered_relative_logprob = model_dispreferred_logprob

    # 2) DPO에서 중요한 포인트: Δ = (preferred - dispreferred)
    dpo_gap = prefered_relative_logprob - disprefered_relative_logprob

    # 3) 진짜 로스 계산
    if loss_type == "sigmoid":
        # Label smoothing이 0.0일 때: 기존 DPO = -E[ log σ(β * Δ) ]
        # Label smoothing이 α>0라면:  -(1-α)*log σ(βΔ)  - α*log σ(-βΔ)
        #    => winner=1, loser=0 대신 winner=(1-α), loser=α로 보는 개념
        alpha = label_smoothing
        # (1 - alpha) * logsigmoid( beta * Δ ) + alpha * logsigmoid( -beta * Δ )
        # sign에 유의: 로그의 부호 때문에 - [...]
        # => 최종 -((1 - alpha) logsigmoid(...) + alpha logsigmoid(...))
        loss = -(
            (1 - alpha) * F.logsigmoid(beta * dpo_gap) +
            alpha        * F.logsigmoid(-beta * dpo_gap)
        ).mean()

    elif loss_type == "hinge":
        # Hinge loss: E[ relu(1 - beta * Δ) ]
        #  => Δ가 1 이상이면(=pref가 훨씬 크면) 로스 0
        #     Δ가 작으면 그만큼 벌점을 준다
        loss = F.relu(1 - beta * dpo_gap).mean()

    else:
        # 혹은 default: 원본 DPO (label smoothing 없는 sigmoid)
        loss = -F.logsigmoid(beta * dpo_gap).mean()

    # 4) 기타 metric들 계산
    reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float().mean()
    reward_margins = dpo_gap.mean()

    return loss, prefered_relative_logprob.mean(), disprefered_relative_logprob.mean(), reward_accuracies, reward_margins


# ================================================
#  기존 DPOTrainer 내지 Trainer class에서 
#  수정 부분: dpo_loss() 호출 시에 loss_type과 smoothing을 넘기면 됨
# ================================================

class DPOTrainer():
    def __init__(self,
                 agent,
                 tokenizer,
                 accelerator,
                 lm_lr: float = 1e-6,
                 epochs: int = 2,
                 max_grad_norm: float = 0.01,
                 grad_accum_steps: int = 8,
                 beta: float = 0.5,
                 use_ref=False,
                 bc_epochs: int = 2,
                 loss_type: str = "sigmoid",        # "sigmoid", "hinge"
                 label_smoothing: float = 0.0       # label smoothing
                 ):
        super().__init__()
        self.agent = agent
        self.tokenizer = tokenizer
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=lm_lr)
        self.grad_accum_steps = grad_accum_steps
        self.epochs = epochs
        self.step = 0
        self.use_ref = use_ref
        self.max_grad_norm = max_grad_norm
        self.beta = beta
        self.accelerator = accelerator
        self.bc_epochs = bc_epochs

        self.loss_type = loss_type
        self.label_smoothing = label_smoothing

        self.agent, self.lm_optimizer = self.accelerator.prepare(self.agent, self.lm_optimizer)

    def actor_loss(self, observation, action, **kwargs):
        loss = plain_bc_loss(self.accelerator.unwrap_model(self.agent).model, self.tokenizer, observation, action)
        self.accelerator.backward(loss)
        return {"bc.loss": loss.detach().cpu().item()}
    
    def actor_loss_not_backward(self,observation,action,**kwargs):
        loss = plain_bc_loss(self.accelerator.unwrap_model(self.agent).model, self.tokenizer, observation, action)
        return loss


    def dpo_loss(self, batch,
                backward=True,
                debug=False,  # <-- debug 키워드 추가
                **kwargs):

        
        device = self.accelerator.device

        # 1) 준비된 batch 데이터에서 observation과 action 추출
        observation = [b for b in batch["observation"]]
        chosen_action = [b for b in batch["chosen_action"]]
        rejected_action = [b for b in batch["rejected_action"]]

        # Prompt와 Action을 개별적으로 토크나이즈
        prompt_ids = self.tokenizer(observation, 
                                    return_tensors="pt", 
                                    truncation=True, 
                                    padding=True)
        prefer_answer_ids = self.tokenizer(chosen_action, 
                                        return_tensors="pt", 
                                        truncation=True, 
                                        padding=True, 
                                        add_special_tokens=False)
        disprefer_answer_ids = self.tokenizer(rejected_action, 
                                            return_tensors="pt", 
                                            truncation=True, 
                                            padding=True, 
                                            add_special_tokens=False)

        # 현재 디바이스로 이동
        prompt_ids = {k: v.to(device) for k, v in prompt_ids.items()}
        prefer_answer_ids = {k: v.to(device) for k, v in prefer_answer_ids.items()}
        disprefer_answer_ids = {k: v.to(device) for k, v in disprefer_answer_ids.items()}

        # 디버그 정보 출력 (1) - 토큰 길이와 실제 예시
        if debug:
            print("=== [DEBUG] Tokenized Prompt ===")
            print(f"prompt_ids['input_ids'].shape: {prompt_ids['input_ids'].shape}")
            print("Sample prompt tokens:", prompt_ids["input_ids"][0])
            print("Sample prompt:", observation[0])
            print()
            
            print("=== [DEBUG] Tokenized Preferred Answer ===")
            print(f"prefer_answer_ids['input_ids'].shape: {prefer_answer_ids['input_ids'].shape}")
            print("Sample prefer tokens:", prefer_answer_ids["input_ids"][0])
            print("Sample prefer answer:", chosen_action[0])
            print()
            
            print("=== [DEBUG] Tokenized Dispreferred Answer ===")
            print(f"disprefer_answer_ids['input_ids'].shape: {disprefer_answer_ids['input_ids'].shape}")
            print("Sample disprefer tokens:", disprefer_answer_ids["input_ids"][0])
            print("Sample disprefer answer:", rejected_action[0])
            print()

        # 2) Prefer와 Disprefer의 input_ids 및 attention_mask 생성
        prefer_input_ids = torch.cat([prompt_ids["input_ids"], prefer_answer_ids["input_ids"]], dim=1)
        prefer_attention_mask = torch.cat([prompt_ids["attention_mask"], prefer_answer_ids["attention_mask"]], dim=1)

        disprefer_input_ids = torch.cat([prompt_ids["input_ids"], disprefer_answer_ids["input_ids"]], dim=1)
        disprefer_attention_mask = torch.cat([prompt_ids["attention_mask"], disprefer_answer_ids["attention_mask"]], dim=1)

        # 손실 계산에서 prompt 부분 제외를 위해 마스크 생성
        prompt_length = prompt_ids["input_ids"].shape[1]

        prefer_loss_mask = torch.zeros_like(prefer_input_ids)
        prefer_loss_mask[:, prompt_length:] = 1

        disprefer_loss_mask = torch.zeros_like(disprefer_input_ids)
        disprefer_loss_mask[:, prompt_length:] = 1

        # 디버그 정보 출력 (2) - 마스크 적용 전/후 모양 확인
        if debug:
            print("=== [DEBUG] Prefer Input Shapes ===")
            print(f"prefer_input_ids.shape: {prefer_input_ids.shape}")
            print(f"prefer_attention_mask.shape: {prefer_attention_mask.shape}")
            print(f"prefer_loss_mask.shape: {prefer_loss_mask.shape}")
            print()
            print("=== [DEBUG] Disprefer Input Shapes ===")
            print(f"disprefer_input_ids.shape: {disprefer_input_ids.shape}")
            print(f"disprefer_attention_mask.shape: {disprefer_attention_mask.shape}")
            print(f"disprefer_loss_mask.shape: {disprefer_loss_mask.shape}")
            print()

        # 3) 모델 출력
        model_prefer_logits = self.accelerator.unwrap_model(self.agent).model(
            input_ids=prefer_input_ids, 
            attention_mask=prefer_attention_mask
        ).logits
        
        model_disprefer_logits = self.accelerator.unwrap_model(self.agent).model(
            input_ids=disprefer_input_ids, 
            attention_mask=disprefer_attention_mask
        ).logits

        if self.use_ref:
            ref_prefer_logits = self.accelerator.unwrap_model(self.agent).ref_model(
                input_ids=prefer_input_ids, 
                attention_mask=prefer_attention_mask
            ).logits
            ref_disprefer_logits = self.accelerator.unwrap_model(self.agent).ref_model(
                input_ids=disprefer_input_ids, 
                attention_mask=disprefer_attention_mask
            ).logits
        else:
            ref_prefer_logits = None
            ref_disprefer_logits = None

        # log-prob 계산 (mask를 이용해 prompt 부분 제외)
        model_prefer_log_prob = get_log_prob(model_prefer_logits, 
                                            prefer_input_ids, 
                                            mask=prefer_loss_mask)
        model_disprefer_log_prob = get_log_prob(model_disprefer_logits, 
                                                disprefer_input_ids, 
                                                mask=disprefer_loss_mask)

        if self.use_ref:
            ref_prefer_log_prob = get_log_prob(ref_prefer_logits, 
                                            prefer_input_ids, 
                                            mask=prefer_loss_mask)
            ref_disprefer_log_prob = get_log_prob(ref_disprefer_logits, 
                                                disprefer_input_ids, 
                                                mask=disprefer_loss_mask)
        else:
            ref_prefer_log_prob = None
            ref_disprefer_log_prob = None

        # 디버그 정보 출력 (3) - 로그 확률 값 확인
        if debug:
            print("=== [DEBUG] Model Prefer Log Prob ===")
            print(model_prefer_log_prob)
            print()
            print("=== [DEBUG] Model Disprefer Log Prob ===")
            print(model_disprefer_log_prob)
            print()
            if self.use_ref:
                print("=== [DEBUG] Ref Prefer Log Prob ===")
                print(ref_prefer_log_prob)
                print()
                print("=== [DEBUG] Ref Disprefer Log Prob ===")
                print(ref_disprefer_log_prob)
                print()

        # 4) DPO 손실 계산
        loss, p_rel, d_rel, reward_acc, reward_margin = calculate_DPO_loss(
            model_preferred_logprob=model_prefer_log_prob,
            model_dispreferred_logprob=model_disprefer_log_prob,
            ref_preferred_logprob=ref_prefer_log_prob,
            ref_dispreferred_logprob=ref_disprefer_log_prob,
            beta=self.beta,
            use_ref=self.use_ref
        )

        dpo_dict = {
            "dpo.loss": loss.detach().cpu().item(),
            "p_rel": p_rel.detach().cpu().item(),
            "d_rel": d_rel.detach().cpu().item(),
            "reward_accuracy": reward_acc.detach().cpu().item(),
            "reward_margin": reward_margin.detach().cpu().item()
        }

        # 디버그 정보 출력 (4) - 최종 DPO 결과
        if debug:
            print("=== [DEBUG] DPO Loss Results ===")
            print(f"loss: {loss}")
            print(f"p_rel: {p_rel}")
            print(f"d_rel: {d_rel}")
            print(f"reward_acc: {reward_acc}")
            print(f"reward_margin: {reward_margin}")
            print()

        # 역전파
        if backward:
            self.accelerator.backward(loss)
            return dpo_dict
        else:
            return loss, dpo_dict

    def combined_update(self,
                        replay_buffer_bc,
                        replay_buffer_dpo, 
                        no_update_actor=False,
                        bc_weight=1.0, 
                        dpo_weight=1.0,
                        update_ref=False):
        """
        BC와 DPO 손실을 동시에 계산하여 업데이트합니다.

        Args:
            no_update_actor (bool): True일 경우, 업데이트를 생략합니다.
            bc_weight (float): BC 손실의 가중치.
            dpo_weight (float): DPO 손실의 가중치.
        
        Returns:
            dict: 평균화된 손실 메트릭.
        """
        self.step += 1
        info = {}
        #info_list = []
        list_bc = []
        list_dpo = []
        if update_ref:
            ref_cache = copy.deepcopy(self.agent.model)
        
        if not no_update_actor:
            # 배치 크기 결정
            action_bsize_bc = 1 if 'llama' in self.accelerator.unwrap_model(self.agent).policy_lm else replay_buffer_bc.batch_size
            action_bsize_dpo = 1 if 'llama' in self.accelerator.unwrap_model(self.agent).policy_lm else replay_buffer_dpo.batch_size

            # 데이터 샘플링
            num_samples = self.grad_accum_steps * max(replay_buffer_bc.batch_size, replay_buffer_dpo.batch_size)
            data_bc = [replay_buffer_bc.sample(1) for _ in range(num_samples)]
            data_dpo = [replay_buffer_dpo.sample(1) for _ in range(num_samples)]

            # 데이터 정제 (각 샘플의 첫 번째 요소 추출)
            data_bc = [ {k: v[0] for k, v in sample.items()} for sample in data_bc ]
            data_dpo = [ {k: v[0] for k, v in sample.items()} for sample in data_dpo ]

            # DataLoader 준비
            dataloader_bc = DataLoader(DummyDataset(data_bc), batch_size=action_bsize_bc, shuffle=True)
            dataloader_dpo = DataLoader(DummyDataset(data_dpo), batch_size=action_bsize_dpo, shuffle=True, collate_fn=custom_collate_fn)

            dataloader_bc = self.accelerator.prepare(dataloader_bc)
            dataloader_dpo = self.accelerator.prepare(dataloader_dpo)

            # 손실 합산을 위한 루프
            for epoch in range(self.epochs):
                self.lm_optimizer.zero_grad()
                
                # BC 손실 계산
                for batch_bc in tqdm(dataloader_bc, desc=f"Epoch {epoch+1}/{self.bc_epochs} - BC Update"):
                    
                    bc_loss = self.actor_loss_not_backward(**batch_bc)
                    #bc_loss = bc_weight * bc_losses["bc.loss"]
                    self.accelerator.backward(bc_loss)
                    bc_dict = {
                        "bc.loss": bc_loss.detach().cpu().item()
                    }
                    list_bc.append(bc_dict)
                
                # DPO 손실 계산
                for batch_dpo in tqdm(dataloader_dpo, desc=f"Epoch {epoch+1}/{self.epochs} - DPO Update"):
                    dpo_loss,dpo_dict = self.dpo_loss(batch_dpo,backward=False)
                    dpo_loss = dpo_weight * dpo_loss
                    dpo_dict["dpo.loss"] = dpo_loss.detach().cpu().item()
                    self.accelerator.backward(dpo_loss)
                    list_dpo.append(dpo_dict)
                
                # Gradient Clipping 및 Optimizer Step
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.lm_optimizer.step()

        if update_ref:
            print('update ref model')
            self.agent.ref_model = copy.deepcopy(ref_cache)

        # 평균 메트릭 계산
        info.update(dict_mean(list_bc))
        info.update(dict_mean(list_dpo))
        #print(info)
        return info



    def bc_update(self,replay_buffer,no_update_actor=False):
        self.step += 1
        info = {}
        info_list = []
        #update actor
        if  not no_update_actor:
            action_bsize = 1 if 'llama' in self.accelerator.unwrap_model(self.agent).policy_lm else replay_buffer.batch_size
            for _ in range(self.epochs):
                self.lm_optimizer.zero_grad()
                data = [replay_buffer.sample(1) for _ in range(self.grad_accum_steps*replay_buffer.batch_size)]
                grad_index = 0
                for d in data:
                    for k,v in d.items():
                        d[k] = v[0]
                dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False)
                dataloader = self.accelerator.prepare(dataloader)
                for batch in dataloader:
                    info_list.append(self.actor_loss(**batch))
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.lm_optimizer.step()
        info.update(dict_mean(info_list))
        return info        

    def dpo_update(self, replay_buffer, no_update_actor=False,update_ref=False):
        self.step += 1
        info = {}
        info_list = []
        if update_ref:
            ref_cache = copy.deepcopy(self.agent.model)
        if not no_update_actor:
            # 두 버퍼 중 더 짧은 길이에 맞춰 페어링
            # 여기서는 batch_size는 두 buffer 모두 동일하다고 가정
            action_bsize = 1 if 'llama' in self.accelerator.unwrap_model(self.agent).policy_lm else replay_buffer.batch_size
            num_pairs = self.grad_accum_steps * replay_buffer.batch_size  # 기존 로직 유지
            if num_pairs > len(replay_buffer):
                num_pairs = len(replay_buffer)
            data = [replay_buffer.sample(1) for _ in range(num_pairs)]
            data = [ {k: v[0] for k, v in sample.items()} for sample in data ]
            #print(data[0])
            for _ in range(self.epochs):
                self.lm_optimizer.zero_grad()
                dataloader = DataLoader(DummyDataset(data), batch_size=action_bsize, shuffle=False,collate_fn=custom_collate_fn)
                dataloader = self.accelerator.prepare(dataloader)
                for batch in dataloader:
                    # bad=False 경우 DPO Loss 사용
                    info_list.append(self.dpo_loss(batch))
                self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.lm_optimizer.step()

        info.update(dict_mean(info_list))
        if update_ref:
            self.agent.ref_model = copy.deepcopy(ref_cache)
        return info
    def save(self, path):
        try:
            torch.save({'model_state_dict': self.accelerator.unwrap_model(self.agent.module).state_dict(),
                        'critic_state_dict': self.accelerator.unwrap_model(self.agent.critic.module).state_dict(),
                        'target_critic_state_dict': self.accelerator.unwrap_model(self.agent.target_critic.module).state_dict(),
                        'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                        'lm_optimizer_state_dict': self.lm_optimizer.state_dict()}, path)
        except:
            print("Failed to save model")

    def load(self, path):
        checkpoint = torch.load(path)
        self.agent.model.load_state_dict(checkpoint['model_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        return self.agent