"""
2D toy dataset for flow matching training.

This module generates synthetic 2D flow data with various obstacle configurations
and boundary conditions for training flow matching models.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ObstacleConfig:
    """Configuration for obstacle generation"""
    min_radius: float = 0.02
    max_radius: float = 0.08
    min_obstacles: int = 0
    max_obstacles: int = 8
    avoid_boundaries: bool = True
    boundary_margin: float = 0.1


@dataclass
class FlowConfig:
    """Configuration for flow field generation"""
    flow_type: str = "potential"  # "potential", "vortex", "saddle", "source_sink"
    noise_level: float = 0.01
    boundary_conditions: str = "dirichlet"  # "dirichlet", "neumann", "periodic"
    time_steps: int = 50
    max_time: float = 1.0


class FlowDataset2D(Dataset):
    """
    2D Flow Dataset for training flow matching models.
    
    Generates synthetic flow fields with obstacles and boundary conditions.
    Each sample contains positions, velocities, obstacles, and context information.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        grid_size: int = 64,
        num_obstacles_range: Tuple[int, int] = (0, 5),
        obstacle_config: Optional[ObstacleConfig] = None,
        flow_config: Optional[FlowConfig] = None,
        mode: str = "train"  # "train", "val", "test"
    ):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.num_obstacles_range = num_obstacles_range
        self.obstacle_config = obstacle_config or ObstacleConfig()
        self.flow_config = flow_config or FlowConfig()
        self.mode = mode
        
        # Set random seed for reproducibility
        if mode == "train":
            np.random.seed(42)
            torch.manual_seed(42)
        elif mode == "val":
            np.random.seed(123)
            torch.manual_seed(123)
        else:  # test
            np.random.seed(456)
            torch.manual_seed(456)
        
        # Pre-generate grid positions
        self.grid_positions = self._create_grid_positions()
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single sample"""
        # Set sample-specific seed for consistent generation
        sample_seed = idx + (42 if self.mode == "train" else 123 if self.mode == "val" else 456)
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        
        # Generate obstacles
        obstacles = self._generate_obstacles()
        
        # Generate start and goal points
        start_point, goal_point = self._generate_start_goal_points(obstacles)
        
        # Create context vector
        context = torch.tensor([start_point[0], start_point[1], goal_point[0], goal_point[1]], 
                              dtype=torch.float32)
        
        # Generate flow field
        velocities = self._generate_flow_field(obstacles, start_point, goal_point)
        
        # Generate time values for flow matching
        times = torch.rand(self.grid_positions.shape[0], 1) * self.flow_config.max_time
        
        return {
            "positions": self.grid_positions,
            "velocities": velocities,
            "obstacles": obstacles,
            "context": context,
            "times": times,
            "start_point": torch.tensor(start_point, dtype=torch.float32),
            "goal_point": torch.tensor(goal_point, dtype=torch.float32)
        }
    
    def _create_grid_positions(self) -> torch.Tensor:
        """Create regular grid positions"""
        x = torch.linspace(0, 1, self.grid_size)
        y = torch.linspace(0, 1, self.grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        return positions.float()
    
    def _generate_obstacles(self) -> torch.Tensor:
        """Generate random obstacles"""
        num_obstacles = np.random.randint(
            self.num_obstacles_range[0], 
            self.num_obstacles_range[1] + 1
        )
        
        if num_obstacles == 0:
            # Return empty obstacles tensor with correct shape
            return torch.zeros(10, 3, dtype=torch.float32)  # Max 10 obstacles
        
        obstacles = []
        
        for _ in range(num_obstacles):
            # Generate position
            if self.obstacle_config.avoid_boundaries:
                margin = self.obstacle_config.boundary_margin
                x = np.random.uniform(margin, 1 - margin)
                y = np.random.uniform(margin, 1 - margin)
            else:
                x = np.random.uniform(0, 1)
                y = np.random.uniform(0, 1)
            
            # Generate radius
            radius = np.random.uniform(
                self.obstacle_config.min_radius,
                self.obstacle_config.max_radius
            )
            
            obstacles.append([x, y, radius])
        
        # Pad with zeros to fixed size
        max_obstacles = 10
        obstacles_tensor = torch.zeros(max_obstacles, 3, dtype=torch.float32)
        for i, obs in enumerate(obstacles[:max_obstacles]):
            obstacles_tensor[i] = torch.tensor(obs, dtype=torch.float32)
        
        return obstacles_tensor
    
    def _generate_start_goal_points(self, obstacles: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Generate start and goal points that don't intersect obstacles"""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Generate random points
            start = np.random.uniform(0.1, 0.9, 2)
            goal = np.random.uniform(0.1, 0.9, 2)
            
            # Check if points are far enough apart
            if np.linalg.norm(goal - start) < 0.3:
                continue
            
            # Check if points intersect with obstacles
            valid = True
            for i in range(obstacles.shape[0]):
                obs = obstacles[i]
                if obs[2] > 0:  # Valid obstacle
                    if (np.linalg.norm(start - obs[:2].numpy()) < obs[2] * 1.2 or
                        np.linalg.norm(goal - obs[:2].numpy()) < obs[2] * 1.2):
                        valid = False
                        break
            
            if valid:
                return start, goal
        
        # Fallback to default positions
        return np.array([0.1, 0.5]), np.array([0.9, 0.5])
    
    def _generate_flow_field(
        self,
        obstacles: torch.Tensor,
        start_point: np.ndarray,
        goal_point: np.ndarray
    ) -> torch.Tensor:
        """Generate flow field based on configuration"""
        positions = self.grid_positions.numpy()
        
        if self.flow_config.flow_type == "potential":
            velocities = self._generate_potential_flow(positions, obstacles, start_point, goal_point)
        elif self.flow_config.flow_type == "vortex":
            velocities = self._generate_vortex_flow(positions, obstacles)
        elif self.flow_config.flow_type == "saddle":
            velocities = self._generate_saddle_flow(positions, obstacles)
        elif self.flow_config.flow_type == "source_sink":
            velocities = self._generate_source_sink_flow(positions, start_point, goal_point)
        else:
            raise ValueError(f"Unknown flow type: {self.flow_config.flow_type}")
        
        # Add noise
        if self.flow_config.noise_level > 0:
            noise = np.random.normal(0, self.flow_config.noise_level, velocities.shape)
            velocities += noise
        
        return torch.tensor(velocities, dtype=torch.float32)
    
    def _generate_potential_flow(
        self,
        positions: np.ndarray,
        obstacles: torch.Tensor,
        start_point: np.ndarray,
        goal_point: np.ndarray
    ) -> np.ndarray:
        """Generate potential flow from start to goal with obstacle avoidance"""
        velocities = np.zeros_like(positions)
        
        for i, pos in enumerate(positions):
            # Basic flow toward goal
            to_goal = goal_point - pos
            distance_to_goal = np.linalg.norm(to_goal) + 1e-6
            base_velocity = to_goal / distance_to_goal * (1.0 / (1.0 + distance_to_goal))
            
            # Obstacle avoidance
            avoidance_velocity = np.zeros(2)
            for j in range(obstacles.shape[0]):
                obs = obstacles[j]
                if obs[2] > 0:  # Valid obstacle
                    obs_pos = obs[:2].numpy()
                    obs_radius = obs[2].item()
                    
                    to_obstacle = pos - obs_pos
                    distance_to_obs = np.linalg.norm(to_obstacle) + 1e-6
                    
                    if distance_to_obs < obs_radius * 3:  # Within influence radius
                        repulsion_strength = 1.0 / (distance_to_obs - obs_radius + 0.01)
                        avoidance_velocity += (to_obstacle / distance_to_obs) * repulsion_strength * 0.1
            
            velocities[i] = base_velocity + avoidance_velocity
        
        return velocities
    
    def _generate_vortex_flow(self, positions: np.ndarray, obstacles: torch.Tensor) -> np.ndarray:
        """Generate vortex flow around obstacles"""
        velocities = np.zeros_like(positions)
        center = np.array([0.5, 0.5])
        
        for i, pos in enumerate(positions):
            # Circular flow around center
            offset = pos - center
            distance = np.linalg.norm(offset) + 1e-6
            
            # Perpendicular vector for circulation
            tangent = np.array([-offset[1], offset[0]]) / distance
            velocities[i] = tangent * (1.0 / (1.0 + distance))
        
        return velocities
    
    def _generate_saddle_flow(self, positions: np.ndarray, obstacles: torch.Tensor) -> np.ndarray:
        """Generate saddle point flow"""
        velocities = np.zeros_like(positions)
        center = np.array([0.5, 0.5])
        
        for i, pos in enumerate(positions):
            offset = pos - center
            # Saddle point: expand in x, contract in y
            velocities[i] = np.array([offset[0], -offset[1]]) * 0.5
        
        return velocities
    
    def _generate_source_sink_flow(
        self,
        positions: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray
    ) -> np.ndarray:
        """Generate flow from source to sink"""
        velocities = np.zeros_like(positions)
        
        for i, pos in enumerate(positions):
            # Flow from start (source) to goal (sink)
            from_start = pos - start_point
            to_goal = goal_point - pos
            
            distance_from_start = np.linalg.norm(from_start) + 1e-6
            distance_to_goal = np.linalg.norm(to_goal) + 1e-6
            
            # Combine source and sink effects
            source_effect = from_start / distance_from_start * (1.0 / distance_from_start)
            sink_effect = to_goal / distance_to_goal * (1.0 / distance_to_goal)
            
            velocities[i] = source_effect * 0.3 + sink_effect * 0.7
        
        return velocities
    
    def visualize_sample(self, idx: int = 0, save_path: Optional[str] = None):
        """Visualize a sample from the dataset"""
        sample = self[idx]
        
        positions = sample["positions"].numpy()
        velocities = sample["velocities"].numpy()
        obstacles = sample["obstacles"].numpy()
        start_point = sample["start_point"].numpy()
        goal_point = sample["goal_point"].numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot 1: Velocity field as arrows
        x = positions[:, 0].reshape(self.grid_size, self.grid_size)
        y = positions[:, 1].reshape(self.grid_size, self.grid_size)
        u = velocities[:, 0].reshape(self.grid_size, self.grid_size)
        v = velocities[:, 1].reshape(self.grid_size, self.grid_size)
        
        # Subsample for cleaner visualization
        step = max(1, self.grid_size // 16)
        ax1.quiver(x[::step, ::step], y[::step, ::step], 
                  u[::step, ::step], v[::step, ::step], 
                  alpha=0.7, scale=20, color='blue')
        
        # Plot obstacles
        for obs in obstacles:
            if obs[2] > 0:  # Valid obstacle
                circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.5)
                ax1.add_patch(circle)
        
        # Plot start and goal
        ax1.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
        ax1.plot(goal_point[0], goal_point[1], 'ro', markersize=10, label='Goal')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal')
        ax1.set_title('Velocity Field')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Velocity magnitude as heatmap
        velocity_magnitude = np.sqrt(u**2 + v**2)
        im = ax2.imshow(velocity_magnitude, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
        
        # Overlay obstacles
        for obs in obstacles:
            if obs[2] > 0:
                circle = plt.Circle((obs[0], obs[1]), obs[2], 
                                  fill=False, edgecolor='white', linewidth=2)
                ax2.add_patch(circle)
        
        ax2.plot(start_point[0], start_point[1], 'wo', markersize=8)
        ax2.plot(goal_point[0], goal_point[1], 'wo', markersize=8)
        
        ax2.set_title('Velocity Magnitude')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def create_dataset_splits(
    total_samples: int = 10000,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    **kwargs
) -> Tuple[FlowDataset2D, FlowDataset2D, FlowDataset2D]:
    """Create train/val/test dataset splits"""
    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)
    test_samples = total_samples - train_samples - val_samples
    
    train_dataset = FlowDataset2D(num_samples=train_samples, mode="train", **kwargs)
    val_dataset = FlowDataset2D(num_samples=val_samples, mode="val", **kwargs)
    test_dataset = FlowDataset2D(num_samples=test_samples, mode="test", **kwargs)
    
    return train_dataset, val_dataset, test_dataset


def test_dataset():
    """Test the dataset implementation"""
    print("Testing FlowDataset2D...")
    
    # Create dataset
    dataset = FlowDataset2D(
        num_samples=10,
        grid_size=32,
        num_obstacles_range=(2, 4)
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Positions shape: {sample['positions'].shape}")
    print(f"Velocities shape: {sample['velocities'].shape}")
    print(f"Obstacles shape: {sample['obstacles'].shape}")
    print(f"Context shape: {sample['context'].shape}")
    
    # Visualize sample
    dataset.visualize_sample(0)
    
    print("Dataset test completed!")


if __name__ == '__main__':
    test_dataset()
