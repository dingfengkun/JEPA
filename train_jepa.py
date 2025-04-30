import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import os
import platform
from tqdm import tqdm  # 需要先 pip install tqdm
import json
from datetime import datetime
import torch.nn as nn

from JEPA import JEPA
from losses import VICRegLoss, L2Loss
from utils import (TrajectoryDataset, setup_device, load_data, 
                  CheckpointManager, save_config, load_config,
                  TrajectoryTransform, save_checkpoint)

def save_model(model, exp_dir):
    """保存模型权重到指定路径"""
    # 保存到实验目录
    exp_path = os.path.join(exp_dir, 'model_weights.pth')
    torch.save(model.state_dict(), exp_path)
    
    # 同时保存到当前目录
    current_path = 'model_weights.pth'
    torch.save(model.state_dict(), current_path)
    
    print(f"Model saved to: {exp_path}")
    print(f"Model also saved to: {current_path}")

def train_epoch(model, dataloader, criterion, mse_criterion, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    # 使用tqdm，每个batch都更新
    pbar = tqdm(enumerate(dataloader), total=num_batches, 
                desc=f'Epoch {epoch+1}', ncols=100)
    
    running_metrics = {
        'loss': 0,
        'inv-loss': 0,
        'var-loss': 0,
        'cov-loss': 0,
        'mse-loss': 0
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
            
            # 计算VICReg损失
            vicreg_metrics = criterion(pred_state, target_state)
            vicreg_loss = vicreg_metrics['loss']
            
            # 计算MSE损失
            mse_loss = mse_criterion(pred_state, target_state)
            
            # 混合损失
            loss = vicreg_loss + mse_loss
            
            batch_total_loss += loss
            
            if batch_metrics is None:
                batch_metrics = {k: v.item() for k, v in vicreg_metrics.items()}
                batch_metrics['mse-loss'] = mse_loss.item()
            else:
                for k, v in vicreg_metrics.items():
                    batch_metrics[k] += v.item()
                batch_metrics['mse-loss'] += mse_loss.item()

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
        
        # 每个batch都更新进度条
        pbar.set_postfix({
            'loss': f'{batch_metrics["loss"]:.4f}',
            'inv': f'{batch_metrics["inv-loss"]:.4f}',
            'var': f'{batch_metrics["var-loss"]:.4f}',
            'cov': f'{batch_metrics["cov-loss"]:.4f}',
            'mse': f'{batch_metrics["mse-loss"]:.4f}'
        })
        
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
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.002,
                       help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-4,
                       help='Minimum learning rate')
    args = parser.parse_args()

    # Configuration
    config = {
        'hidden_dim': 128,
        'action_dim': 2,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'min_learning_rate': args.min_lr,
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

    # Setup device
    device = setup_device()
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    states, actions = load_data(args.data_dir)
    
    # Data augmentation
    transform = TrajectoryTransform() if config['use_data_augmentation'] else None
    
    # Create dataset
    dataset = TrajectoryDataset(states, actions, transform=transform)
    
    # Set DataLoader parameters based on OS
    if platform.system() == 'Windows':
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows
            pin_memory=False  # Disable pin_memory for Windows
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    print(f"Dataset size: {len(dataset)} trajectories")
    
    # Initialize model, loss function and optimizer
    model = JEPA(config['hidden_dim'], config['action_dim']).to(device)
    criterion = VICRegLoss(
        inv_coeff=25.0,
        var_coeff=25.0,
        cov_coeff=1.0,
        gamma=1.0
    ).to(device)
    
    # 添加MSE损失函数
    mse_criterion = nn.MSELoss().to(device)
    
    # Use AdamW optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['min_learning_rate']
    )
    
    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(
        args.exp_dir,
        max_to_keep=config['max_checkpoints']
    )

    # Resume training
    start_epoch = 0
    best_loss = float('inf')
    model_weights_path = os.path.join(args.exp_dir, 'model_weights.pth')
    if args.resume:
        checkpoint, epoch, loss = ckpt_manager.load_latest(model, optimizer)
        if checkpoint is not None:
            start_epoch = epoch + 1
            best_loss = loss
            print(f"Resuming training from epoch {start_epoch}, previous best loss: {best_loss:.4f}")
    
    # Training loop
    print("Starting training...")
    try:
        for epoch in range(start_epoch, config['num_epochs']):
            train_loss = train_epoch(model, dataloader, criterion, mse_criterion, optimizer, device, epoch)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")
            
            # Print summary at the end of each epoch
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            print(f"Average Loss: {train_loss:.4f}")
            
            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                save_model(model, args.exp_dir)
                print(f"New best model saved! Loss: {best_loss:.4f}")
            
            # Save checkpoint
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
        print("\nTraining interrupted...")
        # Save the model if it's better than the best model
        if train_loss < best_loss:
            save_model(model, args.exp_dir)
            print(f"Model at interruption is better, saved to: {model_weights_path}")
    
    finally:
        # Print final results
        print("\nTraining completed!")
        print(f"Best Loss: {best_loss:.4f}")
        print(f"Best model saved at: {model_weights_path}")

if __name__ == "__main__":
    main() 