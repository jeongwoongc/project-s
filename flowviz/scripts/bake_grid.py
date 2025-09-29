#!/usr/bin/env python3
"""
Bake velocity grids for real-time visualization.

This script pre-computes velocity fields for various scene configurations
and saves them as optimized data structures for fast loading in the Swift app.
"""

import numpy as np
import json
import argparse
from pathlib import Path
import h5py
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model.datasets.toy2d import FlowDataset2D, FlowConfig, ObstacleConfig


class GridBaker:
    """Bakes velocity grids for different scene configurations"""
    
    def __init__(self, grid_resolution: int = 128):
        self.grid_resolution = grid_resolution
        self.grid_positions = self._create_grid_positions()
    
    def _create_grid_positions(self) -> np.ndarray:
        """Create regular grid positions"""
        x = np.linspace(0, 1, self.grid_resolution)
        y = np.linspace(0, 1, self.grid_resolution)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        positions = np.stack([xx.flatten(), yy.flatten()], axis=1)
        return positions
    
    def bake_scene(self, scene_config: Dict) -> Dict[str, np.ndarray]:
        """Bake velocity field for a single scene configuration"""
        print(f"Baking scene: {scene_config['name']}")
        
        # Extract scene parameters
        start_point = np.array(scene_config['start_point'])
        goal_point = np.array(scene_config['goal_point'])
        obstacles = scene_config.get('obstacles', [])
        flow_params = scene_config.get('flow_parameters', {})
        
        # Convert obstacles to format expected by dataset
        obstacle_tensor = self._convert_obstacles(obstacles)
        
        # Generate different flow types
        flow_types = ['potential', 'vortex', 'saddle', 'source_sink']
        baked_fields = {}
        
        for flow_type in flow_types:
            print(f"  Generating {flow_type} flow...")
            
            # Create flow configuration
            flow_config = FlowConfig(
                flow_type=flow_type,
                noise_level=0.0  # No noise for baked grids
            )
            
            # Generate flow field
            velocities = self._generate_flow_field(
                self.grid_positions,
                obstacle_tensor,
                start_point,
                goal_point,
                flow_config
            )
            
            # Store in different formats
            baked_fields[flow_type] = {
                'velocities': velocities,
                'velocity_magnitude': np.linalg.norm(velocities, axis=1),
                'velocity_angle': np.arctan2(velocities[:, 1], velocities[:, 0])
            }
        
        # Add metadata
        baked_fields['metadata'] = {
            'grid_resolution': self.grid_resolution,
            'start_point': start_point,
            'goal_point': goal_point,
            'num_obstacles': len(obstacles),
            'scene_name': scene_config['name']
        }
        
        return baked_fields
    
    def _convert_obstacles(self, obstacles: List[Dict]) -> np.ndarray:
        """Convert obstacle list to tensor format"""
        max_obstacles = 10
        obstacle_tensor = np.zeros((max_obstacles, 3))
        
        for i, obs in enumerate(obstacles[:max_obstacles]):
            center = obs['center']
            
            if obs['type'] == 'circle':
                radius = obs['radius']
            elif obs['type'] == 'rectangle':
                # Approximate rectangle as circle with equivalent area
                width = obs.get('width', 0.1)
                height = obs.get('height', 0.1)
                radius = np.sqrt(width * height / np.pi)
            else:
                radius = 0.05  # Default radius
            
            obstacle_tensor[i] = [center[0], center[1], radius]
        
        return obstacle_tensor
    
    def _generate_flow_field(
        self,
        positions: np.ndarray,
        obstacles: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray,
        flow_config: FlowConfig
    ) -> np.ndarray:
        """Generate flow field using the same logic as the dataset"""
        if flow_config.flow_type == "potential":
            return self._generate_potential_flow(positions, obstacles, start_point, goal_point)
        elif flow_config.flow_type == "vortex":
            return self._generate_vortex_flow(positions, obstacles)
        elif flow_config.flow_type == "saddle":
            return self._generate_saddle_flow(positions, obstacles)
        elif flow_config.flow_type == "source_sink":
            return self._generate_source_sink_flow(positions, start_point, goal_point)
        else:
            raise ValueError(f"Unknown flow type: {flow_config.flow_type}")
    
    def _generate_potential_flow(
        self,
        positions: np.ndarray,
        obstacles: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray
    ) -> np.ndarray:
        """Generate potential flow with obstacle avoidance"""
        velocities = np.zeros_like(positions)
        
        for i, pos in enumerate(positions):
            # Basic flow toward goal
            to_goal = goal_point - pos
            distance_to_goal = np.linalg.norm(to_goal) + 1e-6
            base_velocity = to_goal / distance_to_goal * (1.0 / (1.0 + distance_to_goal))
            
            # Obstacle avoidance
            avoidance_velocity = np.zeros(2)
            for j in range(obstacles.shape[0]):
                if obstacles[j, 2] > 0:  # Valid obstacle
                    obs_pos = obstacles[j, :2]
                    obs_radius = obstacles[j, 2]
                    
                    to_obstacle = pos - obs_pos
                    distance_to_obs = np.linalg.norm(to_obstacle) + 1e-6
                    
                    if distance_to_obs < obs_radius * 3:
                        repulsion_strength = 1.0 / (distance_to_obs - obs_radius + 0.01)
                        avoidance_velocity += (to_obstacle / distance_to_obs) * repulsion_strength * 0.1
            
            velocities[i] = base_velocity + avoidance_velocity
        
        return velocities
    
    def _generate_vortex_flow(self, positions: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        """Generate vortex flow"""
        velocities = np.zeros_like(positions)
        center = np.array([0.5, 0.5])
        
        for i, pos in enumerate(positions):
            offset = pos - center
            distance = np.linalg.norm(offset) + 1e-6
            tangent = np.array([-offset[1], offset[0]]) / distance
            velocities[i] = tangent * (1.0 / (1.0 + distance))
        
        return velocities
    
    def _generate_saddle_flow(self, positions: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        """Generate saddle point flow"""
        velocities = np.zeros_like(positions)
        center = np.array([0.5, 0.5])
        
        for i, pos in enumerate(positions):
            offset = pos - center
            velocities[i] = np.array([offset[0], -offset[1]]) * 0.5
        
        return velocities
    
    def _generate_source_sink_flow(
        self,
        positions: np.ndarray,
        start_point: np.ndarray,
        goal_point: np.ndarray
    ) -> np.ndarray:
        """Generate source-sink flow"""
        velocities = np.zeros_like(positions)
        
        for i, pos in enumerate(positions):
            from_start = pos - start_point
            to_goal = goal_point - pos
            
            distance_from_start = np.linalg.norm(from_start) + 1e-6
            distance_to_goal = np.linalg.norm(to_goal) + 1e-6
            
            source_effect = from_start / distance_from_start * (1.0 / distance_from_start)
            sink_effect = to_goal / distance_to_goal * (1.0 / distance_to_goal)
            
            velocities[i] = source_effect * 0.3 + sink_effect * 0.7
        
        return velocities
    
    def save_baked_grids(self, baked_data: Dict, output_path: str):
        """Save baked grids to HDF5 format"""
        print(f"Saving baked grids to: {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Save grid positions (shared across all flow types)
            f.create_dataset('grid_positions', data=self.grid_positions)
            f.create_dataset('grid_resolution', data=self.grid_resolution)
            
            # Save each scene's data
            for scene_name, scene_data in baked_data.items():
                scene_group = f.create_group(scene_name)
                
                # Save metadata
                if 'metadata' in scene_data:
                    metadata_group = scene_group.create_group('metadata')
                    for key, value in scene_data['metadata'].items():
                        if isinstance(value, str):
                            metadata_group.attrs[key] = value
                        else:
                            metadata_group.create_dataset(key, data=value)
                
                # Save flow fields
                for flow_type, flow_data in scene_data.items():
                    if flow_type == 'metadata':
                        continue
                    
                    flow_group = scene_group.create_group(flow_type)
                    for data_name, data_array in flow_data.items():
                        flow_group.create_dataset(data_name, data=data_array, compression='gzip')
    
    def visualize_baked_scene(self, baked_scene: Dict, flow_type: str = 'potential', save_path: Optional[str] = None):
        """Visualize a baked scene"""
        if flow_type not in baked_scene:
            print(f"Flow type '{flow_type}' not found in baked scene")
            return
        
        flow_data = baked_scene[flow_type]
        velocities = flow_data['velocities']
        metadata = baked_scene['metadata']
        
        # Reshape for visualization
        resolution = metadata['grid_resolution']
        positions = self.grid_positions.reshape(resolution, resolution, 2)
        velocities_2d = velocities.reshape(resolution, resolution, 2)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot velocity field
        step = max(1, resolution // 16)
        x = positions[::step, ::step, 0]
        y = positions[::step, ::step, 1]
        u = velocities_2d[::step, ::step, 0]
        v = velocities_2d[::step, ::step, 1]
        
        ax1.quiver(x, y, u, v, alpha=0.7, scale=20, color='blue')
        ax1.plot(metadata['start_point'][0], metadata['start_point'][1], 'go', markersize=10, label='Start')
        ax1.plot(metadata['goal_point'][0], metadata['goal_point'][1], 'ro', markersize=10, label='Goal')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal')
        ax1.set_title(f'{flow_type.title()} Flow - {metadata["scene_name"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot velocity magnitude
        velocity_magnitude = flow_data['velocity_magnitude'].reshape(resolution, resolution)
        im = ax2.imshow(velocity_magnitude, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
        ax2.plot(metadata['start_point'][0], metadata['start_point'][1], 'wo', markersize=8)
        ax2.plot(metadata['goal_point'][0], metadata['goal_point'][1], 'wo', markersize=8)
        ax2.set_title('Velocity Magnitude')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Bake velocity grids for FlowViz')
    parser.add_argument('--scenes', type=str, default='../data/presets/scenes.json',
                       help='Path to scenes configuration file')
    parser.add_argument('--output', type=str, default='../data/baked_grids.h5',
                       help='Output path for baked grids')
    parser.add_argument('--resolution', type=int, default=128,
                       help='Grid resolution')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize baked grids')
    parser.add_argument('--scenes_to_bake', nargs='+', default=None,
                       help='Specific scenes to bake (default: all)')
    
    args = parser.parse_args()
    
    # Load scenes configuration
    scenes_path = Path(args.scenes)
    if not scenes_path.exists():
        print(f"Scenes file not found: {scenes_path}")
        return
    
    with open(scenes_path, 'r') as f:
        scenes_config = json.load(f)
    
    # Initialize baker
    baker = GridBaker(grid_resolution=args.resolution)
    
    # Determine which scenes to bake
    scenes_to_process = args.scenes_to_bake or list(scenes_config['scenes'].keys())
    
    # Bake scenes
    baked_data = {}
    
    for scene_name in tqdm(scenes_to_process, desc="Baking scenes"):
        if scene_name not in scenes_config['scenes']:
            print(f"Warning: Scene '{scene_name}' not found in configuration")
            continue
        
        scene_config = scenes_config['scenes'][scene_name]
        baked_scene = baker.bake_scene(scene_config)
        baked_data[scene_name] = baked_scene
        
        # Visualize if requested
        if args.visualize:
            vis_path = f"baked_{scene_name.lower()}.png" if args.visualize else None
            baker.visualize_baked_scene(baked_scene, 'potential', vis_path)
    
    # Save baked data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baker.save_baked_grids(baked_data, str(output_path))
    
    print(f"\n✅ Successfully baked {len(baked_data)} scenes")
    print(f"   Output: {output_path}")
    print(f"   Grid resolution: {args.resolution}x{args.resolution}")
    print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")


if __name__ == '__main__':
    main()
