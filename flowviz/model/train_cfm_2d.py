#!/usr/bin/env python3
"""
Flow Matching Training Script for 2D Velocity Fields

This script trains a continuous flow matching model to learn velocity fields
for trajectory planning with obstacle avoidance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import wandb

from nets import FlowMatchingNet, VelocityFieldNet
from losses import FlowMatchingLoss, VelocityConsistencyLoss
from datasets.toy2d import FlowDataset2D


def setup_device():
    """Setup compute device (prefer MPS on Apple Silicon)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def create_model(config):
    """Create flow matching model"""
    if config['model_type'] == 'flow_matching':
        model = FlowMatchingNet(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            time_embedding_dim=config['time_embedding_dim']
        )
    else:
        model = VelocityFieldNet(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        )
    
    return model


def create_loss_function(config):
    """Create loss function"""
    if config['model_type'] == 'flow_matching':
        return FlowMatchingLoss(
            velocity_weight=config['velocity_weight'],
            consistency_weight=config['consistency_weight'],
            boundary_weight=config['boundary_weight']
        )
    else:
        return VelocityConsistencyLoss(
            mse_weight=config['mse_weight'],
            divergence_weight=config['divergence_weight']
        )


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        positions = batch['positions'].to(device)  # [B, N, 2]
        velocities = batch['velocities'].to(device)  # [B, N, 2]
        obstacles = batch['obstacles'].to(device)  # [B, M, 3] (x, y, radius)
        context = batch['context'].to(device)  # [B, 4] (start_x, start_y, goal_x, goal_y)
        
        if config['model_type'] == 'flow_matching':
            times = batch['times'].to(device)  # [B, N, 1]
            predicted_velocities = model(positions, obstacles, context, times)
        else:
            predicted_velocities = model(positions, obstacles, context)
        
        loss = criterion(predicted_velocities, velocities, positions, obstacles)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate_epoch(model, dataloader, criterion, device, config):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            positions = batch['positions'].to(device)
            velocities = batch['velocities'].to(device)
            obstacles = batch['obstacles'].to(device)
            context = batch['context'].to(device)
            
            if config['model_type'] == 'flow_matching':
                times = batch['times'].to(device)
                predicted_velocities = model(positions, obstacles, context, times)
            else:
                predicted_velocities = model(positions, obstacles, context)
            
            loss = criterion(predicted_velocities, velocities, positions, obstacles)
            total_loss += loss.item()
    
    return total_loss / num_batches


def visualize_predictions(model, dataset, device, config, save_path=None):
    """Visualize model predictions"""
    model.eval()
    
    # Get a sample from the dataset
    sample = dataset[0]
    positions = sample['positions'].unsqueeze(0).to(device)
    obstacles = sample['obstacles'].unsqueeze(0).to(device)
    context = sample['context'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        if config['model_type'] == 'flow_matching':
            times = torch.zeros_like(positions[:, :, :1])
            predicted_velocities = model(positions, obstacles, context, times)
        else:
            predicted_velocities = model(positions, obstacles, context)
    
    # Convert to numpy
    pos_np = positions[0].cpu().numpy()
    vel_np = predicted_velocities[0].cpu().numpy()
    obs_np = obstacles[0].cpu().numpy()
    ctx_np = context[0].cpu().numpy()
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot velocity field as arrows
    ax.quiver(pos_np[:, 0], pos_np[:, 1], vel_np[:, 0], vel_np[:, 1], 
              alpha=0.7, scale=10, color='blue')
    
    # Plot obstacles
    for obs in obs_np:
        if obs[2] > 0:  # Valid obstacle
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.5)
            ax.add_patch(circle)
    
    # Plot start and goal
    ax.plot(ctx_np[0], ctx_np[1], 'go', markersize=10, label='Start')
    ax.plot(ctx_np[2], ctx_np[3], 'ro', markersize=10, label='Goal')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Predicted Velocity Field')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def save_checkpoint(model, optimizer, epoch, loss, config, path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None):
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['config']


def main():
    parser = argparse.ArgumentParser(description='Train Flow Matching Model')
    parser.add_argument('--config', type=str, default='configs/default.json',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup device
    device = setup_device()
    
    # Initialize wandb
    if args.wandb:
        wandb.init(project="flowviz", config=config)
    
    # Create datasets
    train_dataset = FlowDataset2D(
        num_samples=config['train_samples'],
        grid_size=config['grid_size'],
        num_obstacles_range=config['num_obstacles_range'],
        mode='train'
    )
    
    val_dataset = FlowDataset2D(
        num_samples=config['val_samples'],
        grid_size=config['grid_size'],
        num_obstacles_range=config['num_obstacles_range'],
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Create model
    model = create_model(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function and optimizer
    criterion = create_loss_function(config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _, _ = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, config)
        
        # Update learning rate
        scheduler.step()
        
        # Logging
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, config,
                Path(config['output_dir']) / 'best_model.pt'
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, config,
                Path(config['output_dir']) / f'checkpoint_epoch_{epoch + 1}.pt'
            )
        
        # Visualize predictions
        if (epoch + 1) % config['vis_every'] == 0:
            vis_path = Path(config['output_dir']) / f'predictions_epoch_{epoch + 1}.png'
            visualize_predictions(model, val_dataset, device, config, vis_path)
    
    print("Training completed!")
    
    # Final visualization
    visualize_predictions(model, val_dataset, device, config)
    
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
