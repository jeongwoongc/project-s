import SwiftUI

struct ControlsPanel: View {
    @ObservedObject var viewModel: FlowVizViewModel
    @State private var selectedTab = 0
    
    var body: some View {
        VStack(spacing: 0) {
            // Tab selector
            Picker("Controls", selection: $selectedTab) {
                Text("Flow").tag(0)
                Text("Particles").tag(1)
                Text("Scene").tag(2)
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding()
            
            // Tab content
            TabView(selection: $selectedTab) {
                FlowControlsView(viewModel: viewModel)
                    .tag(0)
                
                ParticleControlsView(viewModel: viewModel)
                    .tag(1)
                
                SceneControlsView(viewModel: viewModel)
                    .tag(2)
            }
            .tabViewStyle(PageTabViewStyle(indexDisplayMode: .never))
        }
        .frame(maxHeight: 500)
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
                        HStack {
                            Text("Flow Speed")
                            Spacer()
                            Text(String(format: "%.2f", viewModel.flowSpeed))
                        }
                        Slider(value: $viewModel.flowSpeed, in: 0.1...3.0)
                            .onChange(of: viewModel.flowSpeed) { _ in
                                viewModel.updateVelocityField()
                            }
                        
                        Toggle("Show Velocity Field", isOn: $viewModel.showVelocityField)
                        Toggle("Show Trajectories", isOn: $viewModel.showTrajectories)
                    }
                }
                
                GroupBox("Start & Goal Points") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Start:")
                            Spacer()
                            Text("(\(String(format: "%.2f", viewModel.startPoint.x)), \(String(format: "%.2f", viewModel.startPoint.y)))")
                        }
                        
                        HStack {
                            Text("Goal:")
                            Spacer()
                            Text("(\(String(format: "%.2f", viewModel.goalPoint.x)), \(String(format: "%.2f", viewModel.goalPoint.y)))")
                        }
                        
                        Button("Reset to Default") {
                            viewModel.startPoint = CGPoint(x: 0.2, y: 0.5)
                            viewModel.goalPoint = CGPoint(x: 0.8, y: 0.5)
                            viewModel.updateVelocityField()
                        }
                        .buttonStyle(BorderedButtonStyle())
                    }
                }
                
                if viewModel.visualizationMode == .neuralODE {
                    GroupBox("Neural ODE") {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Model: Not Loaded")
                                .foregroundColor(.secondary)
                            
                            Button("Load Model") {
                                // TODO: Implement model loading
                            }
                            .buttonStyle(BorderedButtonStyle())
                        }
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
                        HStack {
                            Text("Particle Count")
                            Spacer()
                            Text("\(Int(viewModel.particleCount))")
                        }
                        Slider(value: $viewModel.particleCount, in: 1000...50000, step: 1000)
                        
                        Toggle("Playing", isOn: $viewModel.isPlaying)
                            .toggleStyle(SwitchToggleStyle())
                    }
                }
                
                GroupBox("Visual Settings") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Particle Size: 2.0")
                        Text("Blend Mode: Additive")
                        Text("Trail Length: Auto")
                    }
                    .foregroundColor(.secondary)
                }
                
                GroupBox("Performance") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Target FPS:")
                            Spacer()
                            Text("60")
                        }
                        
                        HStack {
                            Text("Current FPS:")
                            Spacer()
                            Text("--")
                        }
                        
                        HStack {
                            Text("GPU Usage:")
                            Spacer()
                            Text("--")
                        }
                    }
                    .font(.system(.body, design: .monospaced))
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
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Scene:")
                            Spacer()
                            Text(viewModel.currentScene)
                        }
                        
                        HStack {
                            Button("Reset Scene") {
                                viewModel.resetScene()
                            }
                            .buttonStyle(BorderedButtonStyle())
                            
                            Spacer()
                            
                            Button("Save Scene") {
                                showingSaveDialog = true
                            }
                            .buttonStyle(BorderedProminentButtonStyle())
                        }
                    }
                }
                
                GroupBox("Obstacles") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Count:")
                            Spacer()
                            Text("\(viewModel.obstacles.count)")
                        }
                        
                        if !viewModel.obstacles.isEmpty {
                            ForEach(Array(viewModel.obstacles.enumerated()), id: \.element.id) { index, obstacle in
                                HStack {
                                    Text("Obstacle \(index + 1)")
                                    Spacer()
                                    Button("Remove") {
                                        viewModel.removeObstacle(at: index)
                                    }
                                    .buttonStyle(BorderedButtonStyle())
                                    .controlSize(.small)
                                }
                            }
                        } else {
                            Text("No obstacles")
                                .foregroundColor(.secondary)
                        }
                        
                        Text("Tip: Click on the visualization to add obstacles")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                GroupBox("Presets") {
                    VStack(alignment: .leading, spacing: 8) {
                        let presets = ["Default", "Maze", "Spiral", "Vortex"]
                        
                        ForEach(presets, id: \.self) { preset in
                            HStack {
                                Button(preset) {
                                    viewModel.loadScene(preset)
                                }
                                .buttonStyle(BorderedButtonStyle())
                                
                                Spacer()
                                
                                if preset == viewModel.currentScene {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.green)
                                }
                            }
                        }
                    }
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
            Text("Save Scene")
                .font(.title2)
                .bold()
            
            TextField("Scene Name", text: $sceneName)
                .textFieldStyle(RoundedBorderTextFieldStyle())
            
            HStack {
                Button("Cancel") {
                    presentationMode.wrappedValue.dismiss()
                }
                .buttonStyle(BorderedButtonStyle())
                
                Spacer()
                
                Button("Save") {
                    onSave(sceneName)
                }
                .buttonStyle(BorderedProminentButtonStyle())
                .disabled(sceneName.isEmpty)
            }
        }
        .padding()
        .frame(width: 300, height: 150)
    }
}

#Preview {
    ControlsPanel(viewModel: FlowVizViewModel())
        .frame(width: 300, height: 500)
}
