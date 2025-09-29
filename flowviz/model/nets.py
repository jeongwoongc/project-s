"""
Neural network architectures for flow matching and velocity field prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for time and spatial coordinates"""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [..., 1]
        Returns:
            Embedded tensor of shape [..., dim]
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -np.log(self.max_period) * torch.arange(half_dim, device=x.device) / half_dim
        )
        
        args = x * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding


class AttentionBlock(nn.Module):
    """Multi-head attention block for processing spatial relationships"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        
        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x


class ObstacleEncoder(nn.Module):
    """Encode obstacle information into feature vectors"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Positional embedding for obstacle positions
        self.pos_embedding = SinusoidalPositionalEmbedding(output_dim // 2)
    
    def forward(self, obstacles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obstacles: Tensor of shape [B, M, 3] (x, y, radius)
        Returns:
            Encoded obstacles of shape [B, M, output_dim]
        """
        # Separate position and size
        positions = obstacles[..., :2]  # [B, M, 2]
        radius = obstacles[..., 2:3]    # [B, M, 1]
        
        # Encode positions
        pos_embedded = self.pos_embedding(positions.norm(dim=-1, keepdim=True))
        
        # Encode full obstacle info
        obstacle_features = self.encoder(obstacles)
        
        # Combine position embedding with obstacle features
        return obstacle_features + pos_embedded


class FlowMatchingNet(nn.Module):
    """
    Flow Matching Network for learning continuous velocity fields.
    
    This network learns to predict velocity vectors at given positions and times,
    conditioned on obstacle configurations and boundary conditions.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 6,
        time_embedding_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding
        self.time_embedding = SinusoidalPositionalEmbedding(time_embedding_dim)
        
        # Position embedding
        self.pos_embedding = SinusoidalPositionalEmbedding(hidden_dim // 2)
        
        # Context encoder (start/goal points)
        self.context_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Obstacle encoder
        self.obstacle_encoder = ObstacleEncoder(
            input_dim=3,
            hidden_dim=hidden_dim // 2,
            output_dim=hidden_dim
        )
        
        # Input projection
        self.input_proj = nn.Linear(
            input_dim + time_embedding_dim + hidden_dim,  # pos + time + context
            hidden_dim
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # 2D velocity
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        obstacles: torch.Tensor,
        context: torch.Tensor,
        times: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            positions: [B, N, 2] - Query positions
            obstacles: [B, M, 3] - Obstacle positions and radii
            context: [B, 4] - Start and goal positions (start_x, start_y, goal_x, goal_y)
            times: [B, N, 1] - Time values for flow matching
        
        Returns:
            velocities: [B, N, 2] - Predicted velocity vectors
        """
        B, N, _ = positions.shape
        B, M, _ = obstacles.shape
        
        # Embed time
        time_embedded = self.time_embedding(times)  # [B, N, time_embedding_dim]
        
        # Embed positions
        pos_embedded = self.pos_embedding(positions.norm(dim=-1, keepdim=True))  # [B, N, hidden_dim//2]
        
        # Encode context (broadcast to all positions)
        context_encoded = self.context_encoder(context)  # [B, hidden_dim]
        context_encoded = context_encoded.unsqueeze(1).expand(B, N, -1)  # [B, N, hidden_dim]
        
        # Combine position, time, and context
        x = torch.cat([positions, time_embedded, context_encoded], dim=-1)
        x = self.input_proj(x)  # [B, N, hidden_dim]
        
        # Add positional embedding
        x = x + pos_embedded
        
        # Encode obstacles
        if M > 0:
            obstacle_encoded = self.obstacle_encoder(obstacles)  # [B, M, hidden_dim]
            
            # Concatenate positions and obstacles for attention
            x_with_obstacles = torch.cat([x, obstacle_encoded], dim=1)  # [B, N+M, hidden_dim]
            
            # Apply transformer layers
            for layer in self.layers:
                x_with_obstacles = layer(x_with_obstacles)
            
            # Extract position features (ignore obstacle features)
            x = x_with_obstacles[:, :N]
        else:
            # No obstacles - just apply transformer to positions
            for layer in self.layers:
                x = layer(x)
        
        # Output projection
        velocities = self.output_proj(x)
        
        return velocities


class VelocityFieldNet(nn.Module):
    """
    Simpler velocity field network for direct velocity prediction.
    
    This network predicts velocity fields without explicit time modeling,
    suitable for steady-state flow problems.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Position embedding
        self.pos_embedding = SinusoidalPositionalEmbedding(hidden_dim // 2)
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Obstacle encoder
        self.obstacle_encoder = ObstacleEncoder(
            input_dim=3,
            hidden_dim=hidden_dim // 2,
            output_dim=hidden_dim
        )
        
        # Main network
        layers = []
        layers.append(nn.Linear(input_dim + hidden_dim, hidden_dim))
        
        for _ in range(num_layers):
            layers.extend([
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ])
        
        layers.append(nn.Linear(hidden_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(
        self,
        positions: torch.Tensor,
        obstacles: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            positions: [B, N, 2] - Query positions
            obstacles: [B, M, 3] - Obstacle positions and radii
            context: [B, 4] - Start and goal positions
        
        Returns:
            velocities: [B, N, 2] - Predicted velocity vectors
        """
        B, N, _ = positions.shape
        B, M, _ = obstacles.shape
        
        # Encode context
        context_encoded = self.context_encoder(context)  # [B, hidden_dim]
        context_encoded = context_encoded.unsqueeze(1).expand(B, N, -1)  # [B, N, hidden_dim]
        
        # Combine position and context
        x = torch.cat([positions, context_encoded], dim=-1)  # [B, N, input_dim + hidden_dim]
        
        # TODO: Incorporate obstacle information
        # For now, we'll use a simple approach - this can be enhanced with attention mechanisms
        
        # Apply network
        velocities = self.network(x)
        
        return velocities


class UNet2D(nn.Module):
    """
    U-Net architecture for dense velocity field prediction.
    
    This network takes a 2D grid as input and outputs a dense velocity field,
    suitable for grid-based flow visualization.
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # obstacle mask + start/goal channels
        out_channels: int = 2,  # 2D velocity
        hidden_channels: int = 64
    ):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, hidden_channels)
        self.enc2 = self._conv_block(hidden_channels, hidden_channels * 2)
        self.enc3 = self._conv_block(hidden_channels * 2, hidden_channels * 4)
        self.enc4 = self._conv_block(hidden_channels * 4, hidden_channels * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(hidden_channels * 8, hidden_channels * 16)
        
        # Decoder
        self.dec4 = self._conv_block(hidden_channels * 16 + hidden_channels * 8, hidden_channels * 8)
        self.dec3 = self._conv_block(hidden_channels * 8 + hidden_channels * 4, hidden_channels * 4)
        self.dec2 = self._conv_block(hidden_channels * 4 + hidden_channels * 2, hidden_channels * 2)
        self.dec1 = self._conv_block(hidden_channels * 2 + hidden_channels, hidden_channels)
        
        # Output
        self.output = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - Input grid (obstacle mask + boundary conditions)
        
        Returns:
            velocity_field: [B, 2, H, W] - Dense velocity field
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.upsample(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        # Output
        velocity_field = self.output(d1)
        
        return velocity_field


def test_models():
    """Test model architectures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test FlowMatchingNet
    print("Testing FlowMatchingNet...")
    model = FlowMatchingNet().to(device)
    
    B, N, M = 2, 100, 5
    positions = torch.randn(B, N, 2).to(device)
    obstacles = torch.randn(B, M, 3).to(device)
    context = torch.randn(B, 4).to(device)
    times = torch.randn(B, N, 1).to(device)
    
    velocities = model(positions, obstacles, context, times)
    print(f"Input shape: {positions.shape}, Output shape: {velocities.shape}")
    
    # Test VelocityFieldNet
    print("\nTesting VelocityFieldNet...")
    model2 = VelocityFieldNet().to(device)
    velocities2 = model2(positions, obstacles, context)
    print(f"Input shape: {positions.shape}, Output shape: {velocities2.shape}")
    
    # Test UNet2D
    print("\nTesting UNet2D...")
    model3 = UNet2D().to(device)
    grid_input = torch.randn(B, 3, 64, 64).to(device)
    velocity_field = model3(grid_input)
    print(f"Input shape: {grid_input.shape}, Output shape: {velocity_field.shape}")
    
    print("All models tested successfully!")


if __name__ == '__main__':
    test_models()
