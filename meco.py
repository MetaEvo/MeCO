import torch
from collections import deque
import random
import numpy as np
import math
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


class DDQN(nn.Module):
    def __init__(self,n_channel,n_output):
        super(DDQN,self).__init__()
        self.fc1 = torch.nn.Linear(n_channel,64 )
        self.value = torch.nn.Linear(64, n_output)
    def forward(self,x):
        x = F.selu(self.fc1(x))
        v = F.sigmoid(self.value(x))

        return v


class MeCO:
    def __init__(self, n_state, n_action):
        self.n_state = n_state
        self.n_action = n_action
        self.lr = 5 * 1e-3
        self.ddqn = DDQN(n_state, n_action)
        self.optimizer = torch.optim.AdamW(self.ddqn.parameters(), lr=self.lr)
        
        # T_max 50个epoch 27个问题 
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50 * 27, eta_min=1e-4)
        self.criterion = torch.nn.MSELoss()
        
        self.actions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # 复制一个目标网络
        self.target_ddqn = copy.deepcopy(self.ddqn)
        self.target_ddqn.eval() # 目标网络只做前向传播，不更新梯度
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=64) # 回放池子为64
        self.update_step = 10                 # 每隔10步更新一次目标网络
        self.replay_size = 8                    # 每次训练采样8个
        
        self.train_step = 0
        self.gamma = 1.0 # 无折扣因子
        self.grad_clip_norm = 1.0

    
    def train_episode(self, env, seed=42):
        self.ddqn.train()
        
        state = env.reset(seed)
        done = False
        state = torch.FloatTensor(np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)).unsqueeze(0) # [1, n_state]
        
        Return = 0
        
        while not done:
            # --- 1. 动作选择 (epsilon-greedy) ---
            q = self.ddqn(state)
            if torch.rand(1).item() < 0.1:
                action_idx = int(torch.randint(0, self.n_action, (1,)).item())
            else:
                action_idx = int(torch.argmax(q).item())
                
            # --- 2. 与环境交互 ---
            next_state, reward, done = env.step(self.actions[action_idx])
            reward = float(np.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0))
            next_state = torch.FloatTensor(np.nan_to_num(next_state, nan=0.0, posinf=1e6, neginf=-1e6)).unsqueeze(0) # [1, n_state]
            
            # --- 3. 存入经验回放池 ---
            self.replay_buffer.append((state, action_idx, reward, next_state, done))
            state = next_state
            
            Return += reward
            
            self.train_step += 1 # 累计步数更新
            
            # --- 4. 采样并训练 ---
            # 只有当池子里的数据量大于等于 batch_size 时才开始训练
            if len(self.replay_buffer) >= 16: # 最小训练数据量为16
                # 随机采样一个批次
                # batch = np.random.sample(self.replay_buffer, 8) # 采样8个
                batch = np.random.choice(len(self.replay_buffer), self.replay_size, replace=False)
                batch = [self.replay_buffer[idx] for idx in batch]
                # 解包数据
                b_states = torch.cat([transition[0] for transition in batch])
                b_actions = torch.tensor([transition[1] for transition in batch], dtype=torch.int64).unsqueeze(1)
                b_rewards = torch.tensor([transition[2] for transition in batch], dtype=torch.float32).unsqueeze(1)
                b_next_states = torch.cat([transition[3] for transition in batch])
                # 将 done 转为浮点数，用于屏蔽终止状态的未来奖励
                b_dones = torch.tensor([transition[4] for transition in batch], dtype=torch.float32).unsqueeze(1)
                
                # 获取当前状态对应的 Q 值
                q_values = self.ddqn(b_states).gather(1, b_actions)
                
                # DDQN 核心逻辑：
                # 1. 用 当前网络 (self.ddqn) 选出 next_state 下最大 Q 值的动作
                # 2. 用 目标网络 (self.target_ddqn) 评估这个动作的 Q 值
                with torch.no_grad():
                    next_action_idx = self.ddqn(b_next_states).argmax(1, keepdim=True)
                    next_q_values = self.target_ddqn(b_next_states).gather(1, next_action_idx)
                    # 计算 TD 目标值
                    target_q_values = b_rewards + self.gamma * next_q_values * (1 - b_dones)
                
                # 计算损失并反向传播
                
                
                if not torch.isfinite(q_values).all() or not torch.isfinite(target_q_values).all():
                    continue
                
                loss = self.criterion(q_values, target_q_values)
                if not torch.isfinite(loss):
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ddqn.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            # --- 5. 更新目标网络 ---
            if self.train_step % self.update_step == 0:
                self.target_ddqn.load_state_dict(self.ddqn.state_dict())
        
        self.scheduler.step()
        return Return
        
        
    def run_episode(self, env, seed=42):
        torch.no_grad()
        self.ddqn.eval()
        
        state = env.reset(seed)
        
        done = False
        
        state = torch.FloatTensor(np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)).unsqueeze(0) # [1, n_state]
        Return = 0
        with torch.no_grad():
            while not done:
                q = self.ddqn(state) # [1, n_action]
                action_idx = int(torch.argmax(q).item())
                
                next_state, reward, done = env.step(self.actions[action_idx])
                Return += reward
                next_state = torch.FloatTensor(np.nan_to_num(next_state, nan=0.0, posinf=1e6, neginf=-1e6)).unsqueeze(0) # [1, n_state]
                
                state = next_state
        results = env.get_results()
        results['Return'] = Return
        return results

