import os
import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import numpy as np

def load_data(data_dir):
    """
    加载训练数据
    Args:
        data_dir: 数据目录路径
    Returns:
        states: shape (num_trajectories, trajectory_length, 2, 64, 64)
        actions: shape (num_trajectories, trajectory_length-1, 2)
    """
    states = np.load(f"{data_dir}/states.npy")
    actions = np.load(f"{data_dir}/actions.npy")
    
    print(f"加载数据 - States shape: {states.shape}, Actions shape: {actions.shape}")
    return states, actions

class TrajectoryDataset(Dataset):
    def __init__(self, states, actions, transform=None):
        """
        Args:
            states: shape (num_trajectories, trajectory_length, 2, 64, 64)
            actions: shape (num_trajectories, trajectory_length-1, 2)
            transform: 可选的数据增强
        """
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        self.transform = transform

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        states = self.states[idx]
        actions = self.actions[idx]
        
        if self.transform is not None:
            # 对每个时间步的状态应用变换
            transformed_states = []
            for t in range(states.shape[0]):
                transformed_states.append(self.transform(states[t]))
            states = torch.stack(transformed_states)
            
        return states, actions

# 数据增强
class TrajectoryTransform:
    """轨迹数据增强"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, state):
        """
        Args:
            state: shape (2, 64, 64)
        Returns:
            transformed_state: shape (2, 64, 64)
        """
        # 随机水平翻转
        if torch.rand(1) < self.p:
            state = torch.flip(state, dims=[-1])
        
        # 可以添加其他数据增强方法
        return state

def setup_device():
    """设置计算设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

class CheckpointManager:
    def __init__(self, save_dir, max_to_keep=5):
        """
        Args:
            save_dir: 保存checkpoint的目录
            max_to_keep: 最多保存多少个checkpoint
        """
        self.save_dir = Path(save_dir)
        self.max_to_keep = max_to_keep
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = []

    def save(self, model, optimizer, epoch, loss, additional_info=None):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        if additional_info:
            checkpoint.update(additional_info)

        # 保存checkpoint文件
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最新的训练状态
        latest = {
            'latest_checkpoint': str(checkpoint_path),
            'epoch': epoch,
            'loss': loss
        }
        with open(self.save_dir / 'latest.json', 'w') as f:
            json.dump(latest, f, indent=4)

        # 管理checkpoint数量
        self.checkpoints.append(checkpoint_path)
        if len(self.checkpoints) > self.max_to_keep:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

    def load_latest(self, model, optimizer=None):
        """加载最新的checkpoint"""
        latest_path = self.save_dir / 'latest.json'
        if not latest_path.exists():
            return None, 0, float('inf')  # 没有checkpoint

        with open(latest_path, 'r') as f:
            latest = json.load(f)

        checkpoint_path = latest['latest_checkpoint']
        if not os.path.exists(checkpoint_path):
            return None, 0, float('inf')

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint, checkpoint['epoch'], checkpoint['loss']

def save_config(config, save_dir):
    """保存配置文件"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

def load_config(save_dir):
    """加载配置文件"""
    config_path = Path(save_dir) / 'config.json'
    if not config_path.exists():
        return None
    with open(config_path, 'r') as f:
        return json.load(f)

def save_checkpoint(checkpoint, exp_dir):
    """保存检查点到指定目录"""
    checkpoint_path = os.path.join(exp_dir, f'checkpoint_epoch_{checkpoint["epoch"]}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}") 