import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, hidden_dim=256, name='default'):
        super().__init__()
        self.name = name
        
        # 输入是单通道图像 (1, 64, 64)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        self.fc = nn.Linear(256 * 5 * 5, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.ln(x)
        return x

class Predictor(nn.Module):
    def __init__(self, hidden_dim=512, action_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 合并状态和动作
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        
        # 修改网络结构，确保维度匹配
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # state_dim + action_proj_dim = hidden_dim * 2
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, state, action):
        """
        Args:
            state: (batch_size, hidden_dim)
            action: (batch_size, action_dim)
        Returns:
            next_state: (batch_size, hidden_dim)
        """
        # 投影动作到高维空间
        action_embedding = self.action_proj(action)  # (batch_size, hidden_dim)
        
        # 连接状态和动作嵌入
        combined = torch.cat([state, action_embedding], dim=1)  # (batch_size, hidden_dim * 2)
        
        # 预测下一个状态
        next_state = self.net(combined)  # (batch_size, hidden_dim)
        
        return next_state

class JEPA(nn.Module):
    def __init__(self, hidden_dim=256, action_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.repr_dim = hidden_dim  # 确保表示维度与隐藏维度一致
        
        self.agent_encoder = Encoder(hidden_dim=hidden_dim)
        self.wall_encoder = Encoder(hidden_dim=hidden_dim)
        self.predictor = Predictor(hidden_dim=hidden_dim, action_dim=action_dim)
        
        self.target_agent_encoder = Encoder(hidden_dim=hidden_dim)
        self.target_wall_encoder = Encoder(hidden_dim=hidden_dim)
        self.ema_decay = 0.99
        
        self._init_target_encoders()
            
    def _init_target_encoders(self):
        for param_q, param_k in zip(self.agent_encoder.parameters(), 
                                  self.target_agent_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        for param_q, param_k in zip(self.wall_encoder.parameters(), 
                                  self.target_wall_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def update_target_encoder(self):
        for param_q, param_k in zip(self.agent_encoder.parameters(), 
                                  self.target_agent_encoder.parameters()):
            param_k.data = param_k.data * self.ema_decay + \
                          param_q.data * (1 - self.ema_decay)
        
        for param_q, param_k in zip(self.wall_encoder.parameters(), 
                                  self.target_wall_encoder.parameters()):
            param_k.data = param_k.data * self.ema_decay + \
                          param_q.data * (1 - self.ema_decay)

    def forward(self, states=None, actions=None, next_obs=None, teacher_forcing=False):
        """
        Args:
            states: 当前状态
            actions: 动作
            next_obs: 下一个状态（用于teacher forcing）
            teacher_forcing: 是否使用teacher forcing
        Returns:
            训练模式 (teacher_forcing=True):
                (pred_state, target_state) - 用于计算损失
            评估模式 (teacher_forcing=False):
                representations - 预测的表示序列
        """
        if teacher_forcing and next_obs is not None:
            # 获取当前状态的表示
            current_state = self.get_representation(states)
            # 预测下一个状态
            pred_state = self.predictor(current_state, actions)
            # 获取目标状态的表示
            target_state = self.get_representation(next_obs)
            return pred_state, target_state
        
        # 评估模式的代码保持不变
        # ...

    def get_representation(self, obs):
        """获取单个观察的表示"""
        agent_obs = obs[:, 0:1]  # (batch_size, 1, 64, 64)
        wall_obs = obs[:, 1:2]   # (batch_size, 1, 64, 64)
        
        agent_repr = self.agent_encoder(agent_obs)
        wall_repr = self.wall_encoder(wall_obs)
        
        return agent_repr + wall_repr

    def predict_next_repr(self, current_repr, action):
        """预测下一个时间步的表示"""
        return self.predictor(current_repr, action)
