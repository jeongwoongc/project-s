# 🧪 FlowViz UI Testing Guide

This guide shows you how to test the FlowViz UI and understand how it works, even without training ML models first.

## 🚀 Quick Start - Testing the UI

### Method 1: Open in Xcode (Recommended)

1. **Open the Project**:
   ```bash
   cd /Users/dchoi/project-s/flowviz
   open FlowViz.xcodeproj
   ```

2. **Build and Run**:
   - Select your Mac as the target
   - Press `⌘R` or click the "Run" button
   - The app should launch with a dark interface

### Method 2: SwiftUI Previews (Quick Testing)

1. **Open any Swift file** in Xcode (e.g., `ContentView.swift`)
2. **Enable Canvas** (View → Canvas or `⌥⌘↩`)
3. **Click "Resume"** to see live previews
4. **Interact** with the preview to test individual components

## 🎮 What You Should See

### Main Interface
- **Dark background** with subtle blue tint
- **Particle area** in the center (may show placeholder until Metal is working)
- **Green dot** (start point) on the left
- **Red dot** (goal point) on the right
- **Controls panel** on the right side
- **HUD** at the top with play/pause and performance info

### Interactive Elements
- **Click anywhere** to add circular obstacles (red circles)
- **Drag green/red dots** to move start/goal points
- **Toggle controls panel** with the sidebar button
- **Adjust settings** in the controls panel tabs

## 🔧 Testing Without ML Models

The UI is designed to work even without trained models. Here's what happens:

### Default Behavior
- **Flow Matching mode**: Uses mathematical flow fields (no ML needed)
- **Test velocity field**: Simple swirl pattern + goal-directed flow
- **Particle simulation**: 10,000 particles following the velocity field
- **Real-time updates**: Changes when you move points or add obstacles

### Test Data
The app includes `TestData.swift` which provides:
- Procedural velocity fields
- Sample obstacle configurations  
- Scene loading from JSON presets

## 🎯 Testing Different Components

### 1. Controls Panel Testing

**Flow Tab**:
- Change visualization mode (Flow Matching, Diffusion, Neural ODE)
- Adjust flow speed slider
- Toggle velocity field/trajectory display
- Reset start/goal points

**Particles Tab**:
- Adjust particle count (1K-50K)
- Play/pause simulation
- Monitor performance metrics

**Scene Tab**:
- Load different presets (Default, Maze, Spiral, etc.)
- Save/load custom scenes
- Add/remove obstacles
- Reset scene

### 2. HUD Testing

**Performance Monitor**:
- FPS counter (should show ~60 FPS)
- Frame time in milliseconds
- Color-coded performance indicator (green/yellow/red)

**Info Popover**:
- Click the info button
- View keyboard shortcuts and controls
- See current settings

### 3. Metal Rendering Testing

**Visual Elements**:
- Particles should appear as glowing blue/pink dots
- Particle trails create flowing patterns
- Smooth 60 FPS animation
- Proper alpha blending for glow effects

**Performance**:
- Monitor GPU usage in Activity Monitor
- Check for smooth animation without stuttering
- Test with different particle counts

## 🐛 Common Issues & Solutions

### Issue: App Won't Build

**Solution**:
```bash
# Check Xcode version (needs 14.0+)
xcode-select --version

# Clean build folder
⌘⇧K in Xcode

# Reset package cache if needed
File → Packages → Reset Package Caches
```

### Issue: Metal Renderer Not Working

**Symptoms**: Black screen or no particles

**Debug Steps**:
1. Check Console.app for Metal errors
2. Verify GPU compatibility (Metal 2.0+ required)
3. Test on different Mac if available

**Fallback**: The UI should still work, just without particle rendering

### Issue: Performance Problems

**Solutions**:
- Reduce particle count in controls
- Lower grid resolution (modify in ViewModel.swift)
- Close other GPU-intensive apps

### Issue: Scene Presets Not Loading

**Debug**:
```swift
// Add to ViewModel.swift
print("Loading scene: \(sceneName)")
if let scene = TestData.loadTestScene(named: sceneName) {
    print("Loaded scene with \(scene.obstacles.count) obstacles")
} else {
    print("Failed to load scene")
}
```

## 🎨 Customizing for Testing

### Change Default Scene

In `ViewModel.swift`:
```swift
private func setupDefaultScene() {
    // Add your test obstacles
    obstacles = [
        Obstacle(center: CGPoint(x: 0.3, y: 0.3), radius: 0.1, type: .circle),
        Obstacle(center: CGPoint(x: 0.7, y: 0.7), radius: 0.08, type: .circle)
    ]
}
```

### Modify Test Velocity Field

In `TestData.swift`:
```swift
static func generateTestVelocityField(width: Int, height: Int) -> [simd_float2] {
    // Create your own flow patterns
    // Examples: vortex, source/sink, uniform flow, etc.
}
```

### Add Debug Overlays

In `ContentView.swift`:
```swift
#if DEBUG
.overlay(
    VStack {
        Text("Debug: \(viewModel.obstacles.count) obstacles")
        Text("Particles: \(Int(viewModel.particleCount))")
    }
    .foregroundColor(.white)
    .padding()
    .background(.ultraThinMaterial)
    .cornerRadius(8),
    alignment: .topLeading
)
#endif
```

## 📱 SwiftUI Preview Testing

Test individual components in isolation:

### Preview ContentView
```swift
#Preview {
    ContentView()
        .frame(width: 1200, height: 800)
}
```

### Preview ControlsPanel
```swift
#Preview {
    ControlsPanel(viewModel: FlowVizViewModel())
        .frame(width: 300, height: 600)
        .background(.black)
}
```

### Preview HUD
```swift
#Preview {
    HUD(viewModel: FlowVizViewModel())
        .background(.black)
}
```

## 🧪 Advanced Testing

### Memory Testing
```bash
# Monitor memory usage
sudo memory_pressure -l critical
# Run app and check for memory leaks
```

### Performance Profiling
1. **Run with Instruments** (`⌘I` in Xcode)
2. **Choose "Metal System Trace"** for GPU profiling
3. **Choose "Allocations"** for memory profiling

### Unit Testing UI Logic
```swift
// Add to test target
func testVelocityFieldUpdate() {
    let viewModel = FlowVizViewModel()
    viewModel.startPoint = CGPoint(x: 0.1, y: 0.1)
    viewModel.goalPoint = CGPoint(x: 0.9, y: 0.9)
    
    // Test that velocity field updates
    XCTAssertNotNil(viewModel.velocityGrid)
}
```

## 🎯 Testing Checklist

**Basic Functionality**:
- [ ] App launches without crashing
- [ ] UI elements are visible and positioned correctly
- [ ] Controls panel opens/closes
- [ ] Scene presets load successfully

**Interaction Testing**:
- [ ] Click to add obstacles works
- [ ] Drag start/goal points works
- [ ] Sliders update values in real-time
- [ ] Play/pause button toggles animation

**Visual Testing**:
- [ ] Particles render correctly (if Metal works)
- [ ] Colors and themes look good
- [ ] Text is readable
- [ ] Icons and buttons are clear

**Performance Testing**:
- [ ] FPS stays above 30 (ideally 60)
- [ ] Memory usage is reasonable
- [ ] No significant frame drops
- [ ] Responsive to user input

## 🚀 Next Steps

Once basic UI testing works:

1. **Train a simple model** using the Python pipeline
2. **Export to Core ML** using the conversion scripts  
3. **Test with real ML inference** in the app
4. **Optimize performance** based on profiling results

The UI is designed to be fully functional even without ML models, so you can develop and test the interface independently of the machine learning components!
