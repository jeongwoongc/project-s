import SwiftUI

struct ControlsPanel: View {
    @ObservedObject var viewModel: FlowVizViewModel
    @State private var selectedTab = 0
    
    var body: some View {
        VStack(spacing: 0) {
            // Tab selector
            VStack(spacing: 0) {
                Picker("Controls", selection: $selectedTab) {
                    Label("Flow", systemImage: "wind").tag(0)
                    Label("Particles", systemImage: "sparkles").tag(1)
                    Label("Scene", systemImage: "square.grid.2x2").tag(2)
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding()
                
                Divider()
            }
            
            // Tab content
            TabView(selection: $selectedTab) {
                FlowControlsView(viewModel: viewModel)
                    .tag(0)
                
                ParticleControlsView(viewModel: viewModel)
                    .tag(1)
                
                SceneControlsView(viewModel: viewModel)
                    .tag(2)
            }
#if os(macOS)
            .tabViewStyle(DefaultTabViewStyle())
#else
            .tabViewStyle(PageTabViewStyle(indexDisplayMode: .never))
#endif
        }
        .frame(maxHeight: 550)
    }
}

struct FlowControlsView: View {
    @ObservedObject var viewModel: FlowVizViewModel
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                GroupBox("Visualization Mode") {
                    Picker("Mode", selection: $viewModel.visualizationMode) {
                        ForEach(VisualizationMode.allCases, id: \.self) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .onChange(of: viewModel.visualizationMode) { _ in
                        viewModel.updateVelocityField()
                    }
                }
                
                GroupBox("Flow Parameters") {
                    VStack(alignment: .leading, spacing: 12) {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Flow Speed")
                                    .font(.subheadline)
                                Spacer()
                                Text(String(format: "%.2fx", viewModel.flowSpeed))
                                    .font(.system(.subheadline, design: .monospaced))
                                    .foregroundColor(.secondary)
                            }
                            Slider(value: $viewModel.flowSpeed, in: 0.1...3.0)
                                .onChange(of: viewModel.flowSpeed) { newValue in
                                    viewModel.setFlowSpeed(newValue)
                                }
                        }
                        
                        Divider()
                        
                        Toggle("Show Velocity Field", isOn: $viewModel.showVelocityField)
                            .toggleStyle(SwitchToggleStyle())
                        Toggle("Show Trajectories", isOn: $viewModel.showTrajectories)
                            .toggleStyle(SwitchToggleStyle())
                    }
                    .padding(4)
                }
                
                GroupBox("Start & Goal Points") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Circle()
                                .fill(.green)
                                .frame(width: 8, height: 8)
                            Text("Start:")
                                .font(.subheadline)
                            Spacer()
                            Text("(\(String(format: "%.2f", viewModel.startPoint.x)), \(String(format: "%.2f", viewModel.startPoint.y)))")
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                        
                        HStack {
                            Circle()
                                .fill(.red)
                                .frame(width: 8, height: 8)
                            Text("Goal:")
                                .font(.subheadline)
                            Spacer()
                            Text("(\(String(format: "%.2f", viewModel.goalPoint.x)), \(String(format: "%.2f", viewModel.goalPoint.y)))")
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                        
                        Button(action: {
                            viewModel.startPoint = CGPoint(x: 0.2, y: 0.5)
                            viewModel.goalPoint = CGPoint(x: 0.8, y: 0.5)
                            viewModel.updateVelocityField()
                        }) {
                            HStack {
                                Image(systemName: "arrow.counterclockwise")
                                Text("Reset to Default")
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(BorderedButtonStyle())
                        .controlSize(.regular)
                    }
                    .padding(4)
                }
                
                if viewModel.visualizationMode == .neuralODE {
                    GroupBox("Neural ODE") {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Image(systemName: "brain.head.profile")
                                    .foregroundColor(.orange)
                                Text("Model: Not Loaded")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                            
                            Button(action: {
                                // TODO: Implement model loading
                            }) {
                                HStack {
                                    Image(systemName: "square.and.arrow.down.on.square")
                                    Text("Load CoreML Model")
                                }
                                .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(BorderedProminentButtonStyle())
                        }
                        .padding(4)
                    }
                }
            }
            .padding()
        }
    }
}

struct ParticleControlsView: View {
    @ObservedObject var viewModel: FlowVizViewModel
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                GroupBox("Particle System") {
                    VStack(alignment: .leading, spacing: 12) {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Particle Count")
                                    .font(.subheadline)
                                Spacer()
                                Text("\(Int(viewModel.particleCount))")
                                    .font(.system(.subheadline, design: .monospaced))
                                    .foregroundColor(.secondary)
                            }
                            Slider(value: $viewModel.particleCount, in: 1000...50000, step: 1000)
                                .onChange(of: viewModel.particleCount) { newValue in
                                    viewModel.setParticleCount(Int(newValue))
                                }
                        }
                        
                        Divider()
                        
                        Toggle("Playing", isOn: $viewModel.isPlaying)
                            .toggleStyle(SwitchToggleStyle())
                            .onChange(of: viewModel.isPlaying) { playing in
                                viewModel.setPlaying(playing)
                            }
                    }
                    .padding(4)
                }
                
                GroupBox("Visual Settings") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Image(systemName: "circle.fill")
                                .font(.caption)
                                .foregroundColor(.blue)
                            Text("Particle Size: 2.0")
                                .font(.subheadline)
                            Spacer()
                        }
                        
                        HStack {
                            Image(systemName: "wand.and.stars")
                                .font(.caption)
                                .foregroundColor(.purple)
                            Text("Blend Mode: Additive")
                                .font(.subheadline)
                            Spacer()
                        }
                        
                        HStack {
                            Image(systemName: "pencil.line")
                                .font(.caption)
                                .foregroundColor(.green)
                            Text("Trail Length: Auto")
                                .font(.subheadline)
                            Spacer()
                        }
                    }
                    .foregroundColor(.secondary)
                    .padding(4)
                }
                
                GroupBox("Performance") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Circle()
                                .fill(.green)
                                .frame(width: 8, height: 8)
                            Text("Target FPS:")
                                .font(.subheadline)
                            Spacer()
                            Text("60")
                                .font(.system(.subheadline, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                        
                        HStack {
                            Circle()
                                .fill(.blue)
                                .frame(width: 8, height: 8)
                            Text("Current FPS:")
                                .font(.subheadline)
                            Spacer()
                            Text("~60")
                                .font(.system(.subheadline, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                        
                        HStack {
                            Circle()
                                .fill(.orange)
                                .frame(width: 8, height: 8)
                            Text("GPU Usage:")
                                .font(.subheadline)
                            Spacer()
                            Text("Active")
                                .font(.system(.subheadline, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(4)
                }
            }
            .padding()
        }
    }
}

struct SceneControlsView: View {
    @ObservedObject var viewModel: FlowVizViewModel
    @State private var newSceneName = ""
    @State private var showingSaveDialog = false
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                GroupBox("Current Scene") {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Scene:")
                                .font(.subheadline)
                            Spacer()
                            Text(viewModel.currentScene)
                                .font(.system(.subheadline, design: .rounded))
                                .bold()
                                .foregroundColor(.accentColor)
                        }
                        
                        HStack(spacing: 8) {
                            Button(action: {
                                viewModel.resetScene()
                            }) {
                                HStack {
                                    Image(systemName: "arrow.clockwise")
                                    Text("Reset")
                                }
                                .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(BorderedButtonStyle())
                            
                            Button(action: {
                                showingSaveDialog = true
                            }) {
                                HStack {
                                    Image(systemName: "square.and.arrow.down")
                                    Text("Save")
                                }
                                .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(BorderedProminentButtonStyle())
                        }
                    }
                    .padding(4)
                }
                
                GroupBox("Obstacles") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Count:")
                                .font(.subheadline)
                            Spacer()
                            Text("\(viewModel.obstacles.count)")
                                .font(.system(.subheadline, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                        
                        if !viewModel.obstacles.isEmpty {
                            Divider()
                            VStack(spacing: 6) {
                                ForEach(Array(viewModel.obstacles.enumerated()), id: \.element.id) { index, obstacle in
                                    HStack(spacing: 8) {
                                        Circle()
                                            .fill(.red.opacity(0.3))
                                            .frame(width: 12, height: 12)
                                            .overlay(
                                                Circle()
                                                    .strokeBorder(.red, lineWidth: 1.5)
                                            )
                                        Text("Obstacle \(index + 1)")
                                            .font(.caption)
                                        Spacer()
                                        Button(action: {
                                            viewModel.removeObstacle(at: index)
                                        }) {
                                            Image(systemName: "xmark.circle.fill")
                                                .foregroundColor(.red)
                                        }
                                        .buttonStyle(PlainButtonStyle())
                                        .controlSize(.small)
                                    }
                                }
                            }
                        } else {
                            Text("No obstacles")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        HStack(spacing: 4) {
                            Image(systemName: "hand.tap")
                                .font(.caption)
                                .foregroundColor(.accentColor)
                            Text("Click on visualization to add obstacles")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.top, 4)
                    }
                    .padding(4)
                }
                
                GroupBox("Presets") {
                    VStack(spacing: 8) {
                        let presets = [
                            ("Default", "circle.grid.2x2"),
                            ("Maze", "square.grid.3x3"),
                            ("Spiral", "tornado"),
                            ("Vortex", "wind")
                        ]
                        
                        ForEach(presets, id: \.0) { preset in
                            Button(action: {
                                viewModel.loadScene(preset.0)
                            }) {
                                HStack(spacing: 12) {
                                    Image(systemName: preset.1)
                                        .font(.title3)
                                        .frame(width: 24)
                                    
                                    Text(preset.0)
                                        .font(.body)
                                    
                                    Spacer()
                                    
                                    if preset.0 == viewModel.currentScene {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundColor(.green)
                                    }
                                }
                                .padding(.vertical, 8)
                                .padding(.horizontal, 12)
                                .frame(maxWidth: .infinity)
                                .background(preset.0 == viewModel.currentScene ? Color.accentColor.opacity(0.1) : Color.clear)
                                .cornerRadius(8)
                            }
                            .buttonStyle(PlainButtonStyle())
                        }
                    }
                    .padding(4)
                }
            }
            .padding()
        }
        .sheet(isPresented: $showingSaveDialog) {
            SaveSceneDialog(sceneName: $newSceneName) { name in
                viewModel.saveScene(name)
                showingSaveDialog = false
            }
        }
    }
}

struct SaveSceneDialog: View {
    @Binding var sceneName: String
    let onSave: (String) -> Void
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        VStack(spacing: 20) {
            HStack(spacing: 12) {
                Image(systemName: "square.and.arrow.down.fill")
                    .font(.title)
                    .foregroundColor(.accentColor)
                
                Text("Save Scene")
                    .font(.title2)
                    .bold()
            }
            
            VStack(alignment: .leading, spacing: 6) {
                Text("Scene Name")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                TextField("Enter scene name", text: $sceneName)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
            }
            
            HStack(spacing: 12) {
                Button(action: {
                    presentationMode.wrappedValue.dismiss()
                }) {
                    Text("Cancel")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(BorderedButtonStyle())
                .keyboardShortcut(.cancelAction)
                
                Button(action: {
                    onSave(sceneName)
                }) {
                    Text("Save")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(BorderedProminentButtonStyle())
                .disabled(sceneName.isEmpty)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 350, height: 180)
    }
}

#Preview {
    ControlsPanel(viewModel: FlowVizViewModel())
        .frame(width: 300, height: 500)
}
