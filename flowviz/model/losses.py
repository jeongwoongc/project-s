"""
Loss functions for flow matching and velocity field training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching Loss for training continuous flow models.
    
    This loss combines velocity matching, flow consistency, and boundary conditions
    to train models that generate smooth, physically plausible flow fields.
    """
    
    def __init__(
        self,
        velocity_weight: float = 1.0,
        consistency_weight: float = 0.1,
        boundary_weight: float = 0.5,
        obstacle_weight: float = 1.0,
        divergence_weight: float = 0.1
    ):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.consistency_weight = consistency_weight
        self.boundary_weight = boundary_weight
        self.obstacle_weight = obstacle_weight
        self.divergence_weight = divergence_weight
    
    def forward(
        self,
        predicted_velocities: torch.Tensor,
        target_velocities: torch.Tensor,
        positions: torch.Tensor,
        obstacles: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predicted_velocities: [B, N, 2] - Predicted velocity vectors
            target_velocities: [B, N, 2] - Target velocity vectors
            positions: [B, N, 2] - Position coordinates
            obstacles: [B, M, 3] - Obstacle positions and radii
            boundary_mask: [B, N] - Optional mask for boundary points
        
        Returns:
            total_loss: Scalar loss value
        """
        # Basic velocity matching loss
        velocity_loss = F.mse_loss(predicted_velocities, target_velocities)
        
        # Flow consistency loss (neighboring points should have similar velocities)
        consistency_loss = self._compute_consistency_loss(predicted_velocities, positions)
        
        # Boundary condition loss
        boundary_loss = self._compute_boundary_loss(
            predicted_velocities, positions, boundary_mask
        )
        
        # Obstacle avoidance loss
        obstacle_loss = self._compute_obstacle_loss(
            predicted_velocities, positions, obstacles
        )
        
        # Divergence regularization (encourage incompressible flow)
        divergence_loss = self._compute_divergence_loss(predicted_velocities, positions)
        
        # Combine losses
        total_loss = (
            self.velocity_weight * velocity_loss +
            self.consistency_weight * consistency_loss +
            self.boundary_weight * boundary_loss +
            self.obstacle_weight * obstacle_loss +
            self.divergence_weight * divergence_loss
        )
        
        return total_loss
    
    def _compute_consistency_loss(
        self,
        velocities: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute flow consistency loss between neighboring points"""
        B, N, _ = positions.shape
        
        if N < 2:
            return torch.tensor(0.0, device=velocities.device)
        
        # Compute pairwise distances
        pos_expanded = positions.unsqueeze(2)  # [B, N, 1, 2]
        pos_tiled = positions.unsqueeze(1)     # [B, 1, N, 2]
        distances = torch.norm(pos_expanded - pos_tiled, dim=-1)  # [B, N, N]
        
        # Find k nearest neighbors (excluding self)
        k = min(5, N - 1)
        distances_no_self = distances + torch.eye(N, device=distances.device) * 1e6
        _, neighbor_indices = torch.topk(distances_no_self, k, dim=-1, largest=False)
        
        # Compute velocity differences with neighbors
        vel_expanded = velocities.unsqueeze(2).expand(-1, -1, k, -1)  # [B, N, k, 2]
        neighbor_velocities = torch.gather(
            velocities.unsqueeze(1).expand(-1, N, -1, -1),
            2,
            neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, 2)
        )  # [B, N, k, 2]
        
        velocity_differences = torch.norm(vel_expanded - neighbor_velocities, dim=-1)  # [B, N, k]
        
        # Weight by inverse distance
        neighbor_distances = torch.gather(distances, 2, neighbor_indices)
        weights = 1.0 / (neighbor_distances + 1e-6)
        
        consistency_loss = torch.mean(velocity_differences * weights)
        
        return consistency_loss
    
    def _compute_boundary_loss(
        self,
        velocities: torch.Tensor,
        positions: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute boundary condition loss"""
        if boundary_mask is None:
            # Create boundary mask based on position (near edges)
            boundary_threshold = 0.05
            x, y = positions[..., 0], positions[..., 1]
            boundary_mask = (
                (x < boundary_threshold) | (x > 1 - boundary_threshold) |
                (y < boundary_threshold) | (y > 1 - boundary_threshold)
            )
        
        if not boundary_mask.any():
            return torch.tensor(0.0, device=velocities.device)
        
        # At boundaries, velocity should be tangential (no normal component)
        boundary_velocities = velocities[boundary_mask]
        boundary_positions = positions[boundary_mask]
        
        # Compute normal vectors at boundaries
        normals = self._compute_boundary_normals(boundary_positions)
        
        # Penalize normal component of velocity
        normal_components = torch.sum(boundary_velocities * normals, dim=-1)
        boundary_loss = torch.mean(normal_components ** 2)
        
        return boundary_loss
    
    def _compute_boundary_normals(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute outward normal vectors at boundary positions"""
        x, y = positions[..., 0], positions[..., 1]
        
        # Determine which boundary each point is closest to
        dist_to_left = x
        dist_to_right = 1 - x
        dist_to_bottom = y
        dist_to_top = 1 - y
        
        min_dist = torch.stack([dist_to_left, dist_to_right, dist_to_bottom, dist_to_top], dim=-1)
        boundary_type = torch.argmin(min_dist, dim=-1)
        
        # Create normal vectors
        normals = torch.zeros_like(positions)
        normals[boundary_type == 0] = torch.tensor([-1.0, 0.0])  # Left boundary
        normals[boundary_type == 1] = torch.tensor([1.0, 0.0])   # Right boundary
        normals[boundary_type == 2] = torch.tensor([0.0, -1.0])  # Bottom boundary
        normals[boundary_type == 3] = torch.tensor([0.0, 1.0])   # Top boundary
        
        return normals
    
    def _compute_obstacle_loss(
        self,
        velocities: torch.Tensor,
        positions: torch.Tensor,
        obstacles: torch.Tensor
    ) -> torch.Tensor:
        """Compute obstacle avoidance loss"""
        B, N, _ = positions.shape
        B, M, _ = obstacles.shape
        
        if M == 0:
            return torch.tensor(0.0, device=velocities.device)
        
        obstacle_loss = 0.0
        
        for b in range(B):
            for m in range(M):
                obstacle_center = obstacles[b, m, :2]
                obstacle_radius = obstacles[b, m, 2]
                
                if obstacle_radius <= 0:
                    continue
                
                # Compute distances to obstacle
                distances = torch.norm(positions[b] - obstacle_center, dim=-1)
                
                # Find points near obstacle
                influence_radius = obstacle_radius * 2.0
                near_obstacle = distances < influence_radius
                
                if not near_obstacle.any():
                    continue
                
                # For points near obstacle, velocity should point away from obstacle
                near_positions = positions[b][near_obstacle]
                near_velocities = velocities[b][near_obstacle]
                near_distances = distances[near_obstacle]
                
                # Compute desired direction (away from obstacle)
                desired_directions = (near_positions - obstacle_center) / (near_distances.unsqueeze(-1) + 1e-6)
                
                # Weight by proximity to obstacle
                weights = torch.exp(-(near_distances - obstacle_radius) / (obstacle_radius * 0.5))
                
                # Penalize velocities pointing toward obstacle
                velocity_projections = torch.sum(near_velocities * desired_directions, dim=-1)
                penalty = torch.mean(weights * torch.clamp(-velocity_projections, min=0) ** 2)
                
                obstacle_loss += penalty
        
        return obstacle_loss / (B * max(M, 1))
    
    def _compute_divergence_loss(
        self,
        velocities: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute divergence regularization loss"""
        B, N, _ = positions.shape
        
        if N < 4:
            return torch.tensor(0.0, device=velocities.device)
        
        # Estimate divergence using finite differences
        # This is a simplified approximation - for better results, use proper numerical methods
        
        divergence_loss = 0.0
        num_samples = min(N // 4, 100)  # Sample subset for efficiency
        
        for _ in range(num_samples):
            # Sample random points
            indices = torch.randperm(N, device=positions.device)[:4]
            sample_positions = positions[:, indices]  # [B, 4, 2]
            sample_velocities = velocities[:, indices]  # [B, 4, 2]
            
            # Estimate partial derivatives
            dx = sample_positions[:, 1:2, 0] - sample_positions[:, 0:1, 0]
            dy = sample_positions[:, 2:3, 1] - sample_positions[:, 0:1, 1]
            
            dvx_dx = (sample_velocities[:, 1:2, 0] - sample_velocities[:, 0:1, 0]) / (dx + 1e-6)
            dvy_dy = (sample_velocities[:, 2:3, 1] - sample_velocities[:, 0:1, 1]) / (dy + 1e-6)
            
            divergence = dvx_dx + dvy_dy
            divergence_loss += torch.mean(divergence ** 2)
        
        return divergence_loss / num_samples


class VelocityConsistencyLoss(nn.Module):
    """
    Simpler loss function for velocity field training without flow matching.
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        divergence_weight: float = 0.05
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.smoothness_weight = smoothness_weight
        self.divergence_weight = divergence_weight
    
    def forward(
        self,
        predicted_velocities: torch.Tensor,
        target_velocities: torch.Tensor,
        positions: torch.Tensor,
        obstacles: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predicted_velocities: [B, N, 2] - Predicted velocity vectors
            target_velocities: [B, N, 2] - Target velocity vectors
            positions: [B, N, 2] - Position coordinates
            obstacles: [B, M, 3] - Obstacle information
        
        Returns:
            total_loss: Scalar loss value
        """
        # Basic MSE loss
        mse_loss = F.mse_loss(predicted_velocities, target_velocities)
        
        # Smoothness loss
        smoothness_loss = self._compute_smoothness_loss(predicted_velocities, positions)
        
        # Divergence loss
        divergence_loss = self._compute_simple_divergence_loss(predicted_velocities)
        
        total_loss = (
            self.mse_weight * mse_loss +
            self.smoothness_weight * smoothness_loss +
            self.divergence_weight * divergence_loss
        )
        
        return total_loss
    
    def _compute_smoothness_loss(
        self,
        velocities: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute smoothness loss using velocity differences"""
        B, N, _ = velocities.shape
        
        if N < 2:
            return torch.tensor(0.0, device=velocities.device)
        
        # Compute pairwise velocity differences
        vel_diff = velocities.unsqueeze(2) - velocities.unsqueeze(1)  # [B, N, N, 2]
        vel_diff_norm = torch.norm(vel_diff, dim=-1)  # [B, N, N]
        
        # Compute pairwise position distances
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, N, N, 2]
        pos_dist = torch.norm(pos_diff, dim=-1) + 1e-6  # [B, N, N]
        
        # Weight by inverse distance
        weights = 1.0 / pos_dist
        weights = weights - torch.diag_embed(torch.diag(weights[0]))  # Remove self-connections
        
        smoothness_loss = torch.mean(vel_diff_norm * weights)
        
        return smoothness_loss
    
    def _compute_simple_divergence_loss(self, velocities: torch.Tensor) -> torch.Tensor:
        """Compute simple divergence loss"""
        # Approximate divergence as variance of velocity magnitudes
        velocity_magnitudes = torch.norm(velocities, dim=-1)
        divergence_loss = torch.var(velocity_magnitudes)
        
        return divergence_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning flow representations.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,
        positive_pairs: torch.Tensor,
        negative_pairs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, N, D] - Feature embeddings
            positive_pairs: [B, P, 2] - Indices of positive pairs
            negative_pairs: [B, Q, 2] - Indices of negative pairs
        
        Returns:
            contrastive_loss: Scalar loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Compute similarities
        similarity_matrix = torch.matmul(embeddings, embeddings.transpose(-2, -1))
        similarity_matrix = similarity_matrix / self.temperature
        
        # Positive loss
        pos_similarities = torch.gather(
            similarity_matrix.view(-1, similarity_matrix.size(-1)),
            1,
            positive_pairs.view(-1, 1)
        )
        pos_loss = -torch.log(torch.sigmoid(pos_similarities)).mean()
        
        # Negative loss
        neg_similarities = torch.gather(
            similarity_matrix.view(-1, similarity_matrix.size(-1)),
            1,
            negative_pairs.view(-1, 1)
        )
        neg_loss = -torch.log(torch.sigmoid(-neg_similarities)).mean()
        
        return pos_loss + neg_loss


def test_losses():
    """Test loss functions"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    B, N, M = 2, 100, 3
    predicted_velocities = torch.randn(B, N, 2, device=device)
    target_velocities = torch.randn(B, N, 2, device=device)
    positions = torch.rand(B, N, 2, device=device)
    obstacles = torch.rand(B, M, 3, device=device)
    
    # Test FlowMatchingLoss
    print("Testing FlowMatchingLoss...")
    flow_loss = FlowMatchingLoss().to(device)
    loss1 = flow_loss(predicted_velocities, target_velocities, positions, obstacles)
    print(f"FlowMatchingLoss: {loss1.item():.4f}")
    
    # Test VelocityConsistencyLoss
    print("Testing VelocityConsistencyLoss...")
    velocity_loss = VelocityConsistencyLoss().to(device)
    loss2 = velocity_loss(predicted_velocities, target_velocities, positions, obstacles)
    print(f"VelocityConsistencyLoss: {loss2.item():.4f}")
    
    print("All losses tested successfully!")


if __name__ == '__main__':
    test_losses()
