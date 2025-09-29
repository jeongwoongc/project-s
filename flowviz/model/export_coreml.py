#!/usr/bin/env python3
"""
Export trained PyTorch models to Core ML format for use in the Swift app.
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import argparse
from pathlib import Path
import json

from nets import FlowMatchingNet, VelocityFieldNet, UNet2D


def export_flow_matching_model(model_path: str, output_path: str, config: dict):
    """Export FlowMatchingNet to Core ML"""
    print("Exporting FlowMatchingNet to Core ML...")
    
    # Load the trained model
    device = torch.device('cpu')  # Export on CPU
    model = FlowMatchingNet(
        input_dim=config.get('input_dim', 2),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 6),
        time_embedding_dim=config.get('time_embedding_dim', 128)
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create example inputs
    batch_size = 1
    num_positions = 128 * 128  # Grid size for velocity field
    num_obstacles = 10
    
    example_positions = torch.randn(batch_size, num_positions, 2)
    example_obstacles = torch.randn(batch_size, num_obstacles, 3)
    example_context = torch.randn(batch_size, 4)
    example_times = torch.zeros(batch_size, num_positions, 1)
    
    # Trace the model
    traced_model = torch.jit.trace(
        model,
        (example_positions, example_obstacles, example_context, example_times)
    )
    
    # Define input shapes and types
    inputs = [
        ct.TensorType(
            name="positions",
            shape=(batch_size, num_positions, 2),
            dtype=np.float32
        ),
        ct.TensorType(
            name="obstacles", 
            shape=(batch_size, num_obstacles, 3),
            dtype=np.float32
        ),
        ct.TensorType(
            name="context",
            shape=(batch_size, 4),
            dtype=np.float32
        ),
        ct.TensorType(
            name="times",
            shape=(batch_size, num_positions, 1),
            dtype=np.float32
        )
    ]
    
    # Define output
    outputs = [
        ct.TensorType(
            name="velocity_field",
            dtype=np.float32
        )
    ]
    
    # Convert to Core ML
    coreml_model = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    
    # Add metadata
    coreml_model.short_description = "Flow Matching Network for 2D Velocity Fields"
    coreml_model.input_description["positions"] = "2D positions to query (normalized coordinates)"
    coreml_model.input_description["obstacles"] = "Obstacle positions and radii"
    coreml_model.input_description["context"] = "Start and goal positions (start_x, start_y, goal_x, goal_y)"
    coreml_model.input_description["times"] = "Time values for flow matching"
    coreml_model.output_description["velocity_field"] = "Predicted 2D velocity vectors"
    
    # Save the model
    coreml_model.save(output_path)
    print(f"Flow Matching model exported to: {output_path}")
    
    return coreml_model


def export_velocity_field_model(model_path: str, output_path: str, config: dict):
    """Export VelocityFieldNet to Core ML"""
    print("Exporting VelocityFieldNet to Core ML...")
    
    device = torch.device('cpu')
    model = VelocityFieldNet(
        input_dim=config.get('input_dim', 2),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 4)
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create example inputs
    batch_size = 1
    num_positions = 128 * 128
    num_obstacles = 10
    
    example_positions = torch.randn(batch_size, num_positions, 2)
    example_obstacles = torch.randn(batch_size, num_obstacles, 3)
    example_context = torch.randn(batch_size, 4)
    
    # Trace the model
    traced_model = torch.jit.trace(
        model,
        (example_positions, example_obstacles, example_context)
    )
    
    # Define inputs and outputs
    inputs = [
        ct.TensorType(name="positions", shape=(batch_size, num_positions, 2), dtype=np.float32),
        ct.TensorType(name="obstacles", shape=(batch_size, num_obstacles, 3), dtype=np.float32),
        ct.TensorType(name="context", shape=(batch_size, 4), dtype=np.float32)
    ]
    
    outputs = [ct.TensorType(name="velocity_field", dtype=np.float32)]
    
    # Convert to Core ML
    coreml_model = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    
    # Add metadata
    coreml_model.short_description = "Velocity Field Network for 2D Flow Prediction"
    coreml_model.input_description["positions"] = "2D positions to query"
    coreml_model.input_description["obstacles"] = "Obstacle configurations"
    coreml_model.input_description["context"] = "Boundary conditions"
    coreml_model.output_description["velocity_field"] = "Predicted velocity vectors"
    
    coreml_model.save(output_path)
    print(f"Velocity Field model exported to: {output_path}")
    
    return coreml_model


def export_unet_model(model_path: str, output_path: str, config: dict):
    """Export UNet2D to Core ML"""
    print("Exporting UNet2D to Core ML...")
    
    device = torch.device('cpu')
    model = UNet2D(
        in_channels=config.get('in_channels', 3),
        out_channels=config.get('out_channels', 2),
        hidden_channels=config.get('hidden_channels', 64)
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create example input
    batch_size = 1
    height, width = 128, 128
    in_channels = config.get('in_channels', 3)
    
    example_input = torch.randn(batch_size, in_channels, height, width)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Define inputs and outputs
    inputs = [
        ct.ImageType(
            name="input_grid",
            shape=(batch_size, in_channels, height, width),
            color_layout=ct.colorlayout.RGB if in_channels == 3 else ct.colorlayout.GRAYSCALE
        )
    ]
    
    outputs = [ct.TensorType(name="velocity_field", dtype=np.float32)]
    
    # Convert to Core ML
    coreml_model = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    
    # Add metadata
    coreml_model.short_description = "U-Net for Dense Velocity Field Prediction"
    coreml_model.input_description["input_grid"] = "Input grid with obstacle mask and boundary conditions"
    coreml_model.output_description["velocity_field"] = "Dense 2D velocity field"
    
    coreml_model.save(output_path)
    print(f"U-Net model exported to: {output_path}")
    
    return coreml_model


def validate_coreml_model(coreml_model, pytorch_model, example_inputs):
    """Validate that Core ML model produces similar outputs to PyTorch model"""
    print("Validating Core ML model...")
    
    # Get PyTorch prediction
    pytorch_model.eval()
    with torch.no_grad():
        if len(example_inputs) == 4:  # FlowMatchingNet
            pytorch_output = pytorch_model(*example_inputs)
        elif len(example_inputs) == 3:  # VelocityFieldNet
            pytorch_output = pytorch_model(*example_inputs)
        else:  # UNet2D
            pytorch_output = pytorch_model(example_inputs[0])
    
    # Prepare inputs for Core ML
    if len(example_inputs) == 4:
        coreml_input = {
            "positions": example_inputs[0].numpy(),
            "obstacles": example_inputs[1].numpy(),
            "context": example_inputs[2].numpy(),
            "times": example_inputs[3].numpy()
        }
    elif len(example_inputs) == 3:
        coreml_input = {
            "positions": example_inputs[0].numpy(),
            "obstacles": example_inputs[1].numpy(),
            "context": example_inputs[2].numpy()
        }
    else:
        coreml_input = {"input_grid": example_inputs[0].numpy()}
    
    # Get Core ML prediction
    coreml_output = coreml_model.predict(coreml_input)
    coreml_tensor = torch.from_numpy(coreml_output["velocity_field"])
    
    # Compare outputs
    mse_error = torch.mean((pytorch_output - coreml_tensor) ** 2).item()
    max_error = torch.max(torch.abs(pytorch_output - coreml_tensor)).item()
    
    print(f"Validation results:")
    print(f"  MSE Error: {mse_error:.6f}")
    print(f"  Max Error: {max_error:.6f}")
    
    if mse_error < 1e-4:
        print("✅ Validation passed - models produce similar outputs")
    else:
        print("⚠️  Warning - models produce different outputs")
    
    return mse_error < 1e-4


def create_example_config():
    """Create example configuration for model export"""
    return {
        "model_type": "flow_matching",
        "input_dim": 2,
        "hidden_dim": 256,
        "num_layers": 6,
        "time_embedding_dim": 128,
        "in_channels": 3,
        "out_channels": 2,
        "hidden_channels": 64
    }


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch models to Core ML')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained PyTorch model')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for Core ML model')
    parser.add_argument('--model_type', type=str, 
                       choices=['flow_matching', 'velocity_field', 'unet'],
                       default='flow_matching',
                       help='Type of model to export')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model configuration JSON')
    parser.add_argument('--validate', action='store_true',
                       help='Validate exported model against PyTorch model')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_example_config()
    
    # Create output directory
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Export model based on type
    if args.model_type == 'flow_matching':
        coreml_model = export_flow_matching_model(args.model_path, args.output_path, config)
    elif args.model_type == 'velocity_field':
        coreml_model = export_velocity_field_model(args.model_path, args.output_path, config)
    elif args.model_type == 'unet':
        coreml_model = export_unet_model(args.model_path, args.output_path, config)
    
    print(f"\n✅ Model successfully exported to: {args.output_path}")
    
    # Print model info
    spec = coreml_model.get_spec()
    print(f"Model details:")
    print(f"  Inputs: {len(spec.description.input)}")
    print(f"  Outputs: {len(spec.description.output)}")
    print(f"  Model size: {Path(args.output_path).stat().st_size / (1024*1024):.2f} MB")


if __name__ == '__main__':
    main()
