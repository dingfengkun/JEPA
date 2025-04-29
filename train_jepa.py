import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import os
import platform
from tqdm import tqdm  # 需要先 pip install tqdm

from JEPA import JEPA
from losses import VICRegLoss, L2Loss
from utils import (TrajectoryDataset, setup_device, load_data, 
                  CheckpointManager, save_config, load_config,
                  TrajectoryTransform)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    log_interval = 100  # 每100个batch输出一次
    
    # 使用tqdm但减少更新频率
    pbar = tqdm(enumerate(dataloader), total=num_batches, 
                desc=f'Epoch {epoch+1}', ncols=100,
                mininterval=10.0,  # 最小更新间隔10秒
                miniters=log_interval)  # 最小更新batch数
    
    running_metrics = {
        'loss': 0,
        'inv-loss': 0,
        'var-loss': 0,
        'cov-loss': 0
    }
    
    for batch_idx, (states, actions) in pbar:
        states = states.to(device)
        actions = actions.to(device)
        batch_size, traj_len, _, _, _ = states.shape

        batch_total_loss = 0
        batch_metrics = None
        
        for t in range(traj_len - 1):
            obs = states[:, t]
            act = actions[:, t]
            next_obs = states[:, t + 1]
            
            pred_state, target_state = model(obs, act, next_obs, teacher_forcing=True)
            metrics = criterion(pred_state, target_state)
            loss = metrics['loss']
            
            batch_total_loss += loss
            
            if batch_metrics is None:
                batch_metrics = {k: v.item() for k, v in metrics.items()}
            else:
                for k, v in metrics.items():
                    batch_metrics[k] += v.item()

        # 计算平均损失
        batch_total_loss = batch_total_loss / (traj_len - 1)
        for k in batch_metrics:
            batch_metrics[k] /= (traj_len - 1)
        
        # 反向传播
        optimizer.zero_grad()
        batch_total_loss.backward()
        optimizer.step()
        model.update_target_encoder()

        # 累积运行指标
        for k, v in batch_metrics.items():
            running_metrics[k] += v
        
        # 每log_interval个batch更新一次进度条
        if (batch_idx + 1) % log_interval == 0:
            # 计算平均指标
            avg_metrics = {k: v/log_interval for k, v in running_metrics.items()}
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{avg_metrics["loss"]:.4f}',
                'inv': f'{avg_metrics["inv-loss"]:.4f}',
                'var': f'{avg_metrics["var-loss"]:.4f}',
                'cov': f'{avg_metrics["cov-loss"]:.4f}'
            })
            
            # 重置运行指标
            running_metrics = {k: 0 for k in running_metrics}
        
        # 更新总损失
        total_loss += batch_total_loss.item()

    return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='experiments/default',
                       help='Experiment directory')
    parser.add_argument('--data_dir', type=str, default='/scratch/DL25SP/train',
                       help='Data directory')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    args = parser.parse_args()

    # 配置
    config = {
        'hidden_dim': 256,
        'action_dim': 2,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'loss_type': 'vicreg',
        'save_freq': 5,
        'max_checkpoints': 5,
        'use_data_augmentation': True
    }

    if args.resume:
        loaded_config = load_config(args.exp_dir)
        if loaded_config is not None:
            config.update(loaded_config)
    else:
        save_config(config, args.exp_dir)

    # 设置设备
    device = setup_device()
    print(f"Using device: {device}")
    
    # 加载数据
    print("正在加载数据...")
    states, actions = load_data(args.data_dir)
    
    # 数据增强
    transform = TrajectoryTransform() if config['use_data_augmentation'] else None
    
    # 创建数据集
    dataset = TrajectoryDataset(states, actions, transform=transform)
    
    # 根据操作系统设置 DataLoader 参数
    if platform.system() == 'Windows':
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,  # Windows下设置为0
            pin_memory=False  # Windows下关闭pin_memory
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    print(f"数据集大小: {len(dataset)} 轨迹")
    
    # 初始化模型、损失函数和优化器
    model = JEPA(config['hidden_dim'], config['action_dim']).to(device)
    criterion = VICRegLoss(
        inv_coeff=25.0,
        var_coeff=25.0,
        cov_coeff=1.0,
        gamma=1.0
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 初始化checkpoint管理器
    ckpt_manager = CheckpointManager(
        args.exp_dir,
        max_to_keep=config['max_checkpoints']
    )

    # 恢复训练
    start_epoch = 0
    best_loss = float('inf')
    model_weights_path = os.path.join(args.exp_dir, 'model_weights.pth')
    if args.resume:
        checkpoint, epoch, loss = ckpt_manager.load_latest(model, optimizer)
        if checkpoint is not None:
            start_epoch = epoch + 1
            best_loss = loss
            print(f"恢复训练从epoch {start_epoch}，之前最佳loss: {best_loss:.4f}")
    
    # 训练循环
    print("开始训练...")
    try:
        for epoch in range(start_epoch, config['num_epochs']):
            train_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch)
            
            # 每个epoch结束时输出总结
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            print(f"Average Loss: {train_loss:.4f}")
            
            # 保存最佳模型
            if train_loss < best_loss:
                best_loss = train_loss
                save_model(model, os.path.join(args.exp_dir, 'model_weights.pth'))
                print(f"New best model saved! Loss: {best_loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % config['save_freq'] == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'best_loss': best_loss,
                    'config': config
                }, args.exp_dir)

    except KeyboardInterrupt:
        print("\n训练被中断...")
        # 如果中断时的模型比最佳模型更好，则保存它
        if train_loss < best_loss:
            torch.save(model.state_dict(), model_weights_path)
            print(f"中断时的模型性能更好，已保存到: {model_weights_path}")
    
    finally:
        # 训练结束后打印最佳结果
        print("\n训练完成!")
        print(f"最佳 Loss: {best_loss:.4f}")
        print(f"最佳模型保存在: {model_weights_path}")

if __name__ == "__main__":
    main() 