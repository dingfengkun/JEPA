import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, hidden_dim=128, name='default'):
        super().__init__()
        self.name = name
        
        # 输入是单通道图像 (1, 64, 64)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # ResNet块
        self.layer1 = self._make_layer(16, 32, 1, stride=2)
        self.layer2 = self._make_layer(32, 64, 1, stride=2)
        self.layer3 = self._make_layer(64, 128, 1, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Linear(128, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.ln(x)
        return x

class Predictor(nn.Module):
    def __init__(self, hidden_dim=256, action_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 动作投影层
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 状态-动作融合网络
        self.net = nn.Sequential(
            # 第一层：扩大维度
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 4),
            
            # 第二层：处理融合特征
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 4),
            
            # 第三层：降维
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            
            # 第四层：输出层
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, state, action):
        """
        Args:
            state: [B, hidden_dim]
            action: [B, action_dim]
        Returns:
            next_state: [B, hidden_dim]
        """
        # 投影动作到隐藏空间
        action_proj = self.action_proj(action)  # [B, hidden_dim]
        
        # 合并状态和动作
        x = torch.cat([state, action_proj], dim=1)  # [B, hidden_dim * 2]
        
        # 通过预测网络
        x = self.net(x)  # [B, hidden_dim]
        
        return x

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
        
        # 评估模式
        batch_size = states.size(0)
        seq_len = actions.size(1)
        hidden_dim = self.hidden_dim
        
        # 初始化预测序列
        pred_encs = torch.zeros((batch_size, seq_len + 1, hidden_dim), device=states.device)
        
        # 获取初始状态的表示
        current_state = self.get_representation(states)
        pred_encs[:, 0] = current_state
        
        # 逐步预测
        for t in range(seq_len):
            current_action = actions[:, t]
            next_state = self.predict_next_repr(current_state, current_action)
            pred_encs[:, t + 1] = next_state
            current_state = next_state
        
        return pred_encs

    def get_representation(self, obs):
        """获取单个观察的表示
        Args:
            obs: [B, 1, C, H, W] 或 [B, C, H, W]
        Returns:
            representation: [B, hidden_dim]
        """
        # 确保输入是4D或5D
        if len(obs.shape) == 5:
            # [B, 1, C, H, W] -> [B, C, H, W]
            obs = obs.squeeze(1)
        
        # 分离智能体和墙壁的观察
        agent_obs = obs[:, 0:1]  # [B, 1, H, W]
        wall_obs = obs[:, 1:2]   # [B, 1, H, W]
        
        # 编码
        agent_repr = self.agent_encoder(agent_obs)
        wall_repr = self.wall_encoder(wall_obs)
        
        # 合并表示
        return agent_repr + wall_repr

    def predict_next_repr(self, current_repr, action):
        """预测下一个时间步的表示"""
        return self.predictor(current_repr, action)
