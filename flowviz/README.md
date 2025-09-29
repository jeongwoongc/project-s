# 🌌 FlowViz: Real-Time Flow Matching Visualizer

FlowViz is a **Mac-native SwiftUI + Metal app** that brings cutting-edge robotics and machine learning methods to life through **interactive, artistic visualizations**. The project combines **mathematical models (Flow Matching, Diffusion, ODEs, Control Barrier Functions)** with **real-time GPU rendering**, turning abstract robotics concepts into dynamic particle flows and glowing trajectories.

![FlowViz Demo](docs/images/flowviz_demo.gif)

---

## 🎯 Project Goals

- **Learn** the math and ML behind Flow Matching, Diffusion, and continuous-time dynamics
- **Experiment** with robotics concepts like safe planning, obstacle avoidance, and optimal transport
- **Visualize** velocity fields and trajectories as interactive art, blending science and creativity
- **Run in real time** on a Mac using Metal and Core ML (lightweight, portable, no server needed)

---

## ✨ Core Features

- **Particle Flow Sandbox**: Drag start/goal points and obstacles, watch glowing streams bend in real time
- **Velocity Field Visualizer**: Precomputed flow fields sampled on a grid, rendered with GPU particles
- **Diffusion vs Flow Mode**: Side-by-side comparison of iterative diffusion vs one-shot flow matching
- **Safety Fields**: Force-field style barrier functions that keep trajectories safe around obstacles
- **Custom Scenes**: Save and load layouts of obstacles/goals for demos and art export

---

## 🧑‍🔬 Future Extensions

- **Neural ODEs**: Continuous dynamics visualizations with learned vector fields
- **Optimal Transport Morphs**: Morph one particle cloud into another along Wasserstein geodesics
- **Swarm Simulations**: Multi-agent flows for flocking, choreography, or multi-robot planning
- **Perception Inputs**: Obstacle maps from Core ML depth models or ARKit
- **Art Modes**: Music-synced flows, 3D flow sculptures, AR overlays

---

## 📂 Project Structure

```
flowviz/
├─ app/                        # macOS SwiftUI + Metal frontend
│  ├─ FlowVizApp.swift         # Main app entry point
│  ├─ ContentView.swift        # Main UI layout
│  ├─ ViewModel.swift          # App state and logic
│  ├─ Renderer/                # Metal rendering pipeline
│  │  ├─ MetalRenderer.swift   # GPU particle system
│  │  ├─ Shaders.metal         # Vertex/fragment/compute shaders
│  │  └─ Buffers.swift         # Buffer management utilities
│  ├─ UX/                      # User interface components
│  │  ├─ ControlsPanel.swift   # Interactive controls sidebar
│  │  └─ HUD.swift            # Heads-up display and overlays
│  └─ Utils/
│     └─ Geometry.swift        # Geometric utilities and math
│
├─ core/                       # Math + glue layer
│  ├─ VelocityGrid.swift       # Velocity field computation
│  ├─ DistanceField.swift      # Signed distance fields for obstacles
│  ├─ Samplers.swift           # Trajectory sampling and integration
│  └─ ModelIO.swift            # Core ML model loading and inference
│
├─ model/                      # ML training + export (Python)
│  ├─ train_cfm_2d.py          # Flow matching training script
│  ├─ nets.py                  # Neural network architectures
│  ├─ losses.py                # Loss functions for training
│  ├─ export_coreml.py         # PyTorch to Core ML conversion
│  └─ datasets/
│     └─ toy2d.py             # 2D synthetic dataset generation
│
├─ data/                       # Saved weights + presets
│  ├─ weights/                 # Core ML model files
│  │  └─ cfm_toy2d_v1.mlmodel # (Generated after training)
│  └─ presets/
│     └─ scenes.json          # Predefined scene configurations
│
├─ scripts/                    # Helpers + tooling
│  ├─ bake_grid.py            # Pre-compute velocity grids
│  └─ convert_weights.sh      # Convert PyTorch models to Core ML
│
└─ README.md                  # This file
```

---

## 🛠 Tech Stack

- **SwiftUI + Metal** for Mac-native UI and GPU rendering
- **PyTorch (MPS)** for training small Flow Matching models
- **Core ML** for on-device model inference
- **JSON/MLModel** export pipeline for portability

---

## 🚀 Getting Started

### Prerequisites

- **macOS 13.0+** (for Core ML and Metal Performance Shaders)
- **Xcode 14.0+** 
- **Python 3.8+** with pip
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/flowviz.git
cd flowviz
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install torch torchvision torchaudio
pip install coremltools numpy matplotlib tqdm h5py
pip install wandb  # Optional: for experiment tracking
```

### 3. Train Your First Model

```bash
cd model

# Train a simple flow matching model
python train_cfm_2d.py --config configs/default.json --wandb

# Export to Core ML format
python export_coreml.py \
    --model_path ../data/checkpoints/best_model.pt \
    --output_path ../data/weights/cfm_toy2d_v1.mlmodel \
    --model_type flow_matching
```

### 4. Pre-compute Velocity Grids (Optional)

```bash
cd scripts

# Bake velocity grids for faster loading
python bake_grid.py \
    --scenes ../data/presets/scenes.json \
    --output ../data/baked_grids.h5 \
    --resolution 128 \
    --visualize
```

### 5. Open in Xcode

```bash
# From the flowviz directory
open app/FlowViz.xcodeproj
```

Build and run the project in Xcode. The app will automatically load available models and scene presets.

---

## 🎮 Usage

### Basic Controls

- **🖱️ Click**: Add circular obstacles
- **🎯 Drag Green/Red Points**: Move start/goal positions
- **⏯️ Spacebar**: Play/pause particle simulation
- **🎛️ Controls Panel**: Adjust flow parameters, particle count, visualization mode

### Visualization Modes

1. **Flow Matching**: One-shot trajectory generation using continuous flows
2. **Diffusion**: Iterative denoising process visualization
3. **Neural ODE**: Learned continuous dynamics (requires trained model)

### Scene Presets

The app comes with several predefined scenes:

- **Default**: Simple obstacle avoidance
- **Maze**: Navigate through corridors
- **Spiral**: Particles spiral around obstacles
- **Vortex**: Multiple vortices create complex patterns
- **Narrow Passage**: Flow acceleration through constraints

---

## 🧪 Training Your Own Models

### Quick Start Training

```bash
cd model

# Train with default settings
python train_cfm_2d.py --config configs/default.json

# Train with custom parameters
python train_cfm_2d.py \
    --config configs/custom.json \
    --wandb \
    --resume ../data/checkpoints/checkpoint_epoch_50.pt
```

### Custom Dataset

Modify `datasets/toy2d.py` to create your own flow scenarios:

```python
# Example: Add new flow type
def _generate_custom_flow(self, positions, obstacles, start, goal):
    # Your custom flow logic here
    velocities = np.zeros_like(positions)
    # ... implement your flow field
    return velocities
```

### Model Architectures

The project includes several neural architectures:

- **FlowMatchingNet**: Transformer-based continuous flow model
- **VelocityFieldNet**: Simple MLP for steady-state flows
- **UNet2D**: Convolutional model for dense grid prediction

### Hyperparameter Tuning

Key hyperparameters in `configs/default.json`:

```json
{
    "hidden_dim": 256,           // Model capacity
    "num_layers": 6,             // Network depth
    "learning_rate": 1e-4,       // Training speed
    "velocity_weight": 1.0,      // Loss term weights
    "consistency_weight": 0.1,
    "obstacle_weight": 1.0
}
```

---

## 🎨 Customization

### Adding New Scene Presets

Edit `data/presets/scenes.json`:

```json
{
  "scenes": {
    "MyCustomScene": {
      "name": "My Custom Scene",
      "description": "A scene I created",
      "start_point": [0.1, 0.2],
      "goal_point": [0.9, 0.8],
      "obstacles": [
        {
          "type": "circle",
          "center": [0.5, 0.5],
          "radius": 0.1
        }
      ],
      "flow_parameters": {
        "flow_speed": 1.2,
        "visualization_mode": "flow_matching",
        "particle_count": 15000
      }
    }
  }
}
```

### Modifying Shaders

Edit `app/Renderer/Shaders.metal` to customize particle appearance:

```metal
// Change particle colors
float3 baseColor = float3(1.0, 0.5, 0.0);  // Orange particles
float3 fastColor = float3(0.0, 1.0, 0.5);  // Green for fast particles
```

### Custom Loss Functions

Add new loss terms in `model/losses.py`:

```python
def _compute_custom_loss(self, velocities, positions):
    # Your custom loss logic
    return loss_value
```

---

## 🔧 Troubleshooting

### Common Issues

**Model Not Loading**
```bash
# Check if Core ML model exists
ls -la data/weights/
# Re-export if missing
cd model && python export_coreml.py --model_path ... --output_path ...
```

**Poor Performance**
```bash
# Reduce particle count in app settings
# Or bake velocity grids for faster loading
cd scripts && python bake_grid.py
```

**Training Fails**
```bash
# Check Python dependencies
pip install -r requirements.txt
# Reduce batch size or model size in config
```

### Debug Mode

Enable debug logging in Swift:

```swift
// In ViewModel.swift
#if DEBUG
print("Debug: Velocity field updated with \(velocityGrid.velocityData.count) points")
#endif
```

---

## 📊 Performance Optimization

### For Real-Time Performance

1. **Pre-bake velocity grids** for common scenes
2. **Reduce particle count** (5K-15K optimal)
3. **Use lower grid resolution** (64x64 instead of 128x128)
4. **Enable Metal Performance Shaders** for better GPU utilization

### For Training Speed

1. **Use Apple Silicon MPS** backend
2. **Reduce dataset size** during development
3. **Use mixed precision** training
4. **Enable gradient checkpointing** for large models

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow Swift style guidelines for iOS/macOS development
- Use Python Black for Python code formatting
- Add unit tests for new functionality
- Update documentation for API changes

---

## 📚 References & Inspiration

### Flow Matching & Diffusion Models
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Score-Based Generative Models](https://arxiv.org/abs/2011.13456)
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)

### Robotics & Planning
- [Optimal Transport for Robotics](https://arxiv.org/abs/2004.08333)
- [Control Barrier Functions](https://arxiv.org/abs/1903.11199)
- [Sampling-Based Motion Planning](http://lavalle.pl/planning/)

### Visualization & Graphics
- [Real-Time Rendering](https://www.realtimerendering.com/)
- [GPU Gems Series](https://developer.nvidia.com/gpugems)
- [Metal Performance Shaders](https://developer.apple.com/metal/Metal-Performance-Shaders/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Flow Matching** research community for mathematical foundations
- **Apple Metal** team for excellent GPU compute framework
- **PyTorch** team for MPS backend enabling Apple Silicon training
- **SwiftUI** team for making native Mac development delightful

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/flowviz&type=Date)](https://star-history.com/#yourusername/flowviz&Date)

---

**Built with ❤️ for the intersection of AI, robotics, and interactive art.**
