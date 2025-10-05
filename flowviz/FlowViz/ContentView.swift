import SwiftUI
import MetalKit

struct ContentView: View {
    @StateObject private var viewModel = FlowVizViewModel()
    @State private var showControls = true
    
    var body: some View {
        ZStack {
            // Full-screen Metal rendering view
            MetalView(viewModel: viewModel)
                .ignoresSafeArea(.all)
            
            // Interaction overlay for dragging start/goal points and placing obstacles
            InteractionOverlay(viewModel: viewModel)
                .ignoresSafeArea(.all)
            
            // Overlay HUD - allow clicks through empty space
            VStack {
                HStack {
                    HUD(viewModel: viewModel)
                        .fixedSize()
                    Spacer()
                        .allowsHitTesting(false)
                }
                .padding()
                
                Spacer()
                    .allowsHitTesting(false)
            }
            .allowsHitTesting(true)
            
            // Floating controls panel
            if showControls {
                VStack(spacing: 0) {
                    Spacer()
                        .frame(minHeight: 0)
                    HStack(spacing: 0) {
                        Spacer()
                            .frame(minWidth: 0)
                            .allowsHitTesting(false)
                        
                        FloatingControlsPanel(
                            viewModel: viewModel,
                            onClose: {
                                withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                                    showControls = false
                                }
                            }
                        )
                        .frame(width: 340, height: 500)
                        .background(
                            RoundedRectangle(cornerRadius: 16)
                                .fill(.ultraThinMaterial)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .strokeBorder(Color.white.opacity(0.2), lineWidth: 1)
                        )
                        .shadow(color: Color.black.opacity(0.3), radius: 20, x: 0, y: 10)
                        .padding(.trailing, 16)
                        .padding(.bottom, 16)
                    }
                    .frame(maxWidth: .infinity, alignment: .trailing)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(true)
                .transition(.move(edge: .trailing).combined(with: .opacity))
            }
            
            // Floating toggle button (when panel is hidden)
            if !showControls {
                VStack(spacing: 0) {
                    Spacer()
                        .frame(minHeight: 0)
                    HStack(spacing: 0) {
                        Spacer()
                            .frame(minWidth: 0)
                        Button(action: {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                                showControls = true
                            }
                        }) {
                            Image(systemName: "slider.horizontal.3")
                                .font(.title2)
                                .foregroundColor(.white)
                                .frame(width: 50, height: 50)
                                .background(Circle().fill(.ultraThinMaterial))
                                .overlay(Circle().strokeBorder(Color.white.opacity(0.2), lineWidth: 1))
                        }
                        .buttonStyle(PlainButtonStyle())
                        .shadow(color: Color.black.opacity(0.3), radius: 12)
                        .help("Show Controls")
                        .padding(.trailing, 16)
                        .padding(.bottom, 16)
                        .fixedSize()
                    }
                    .frame(maxWidth: .infinity, alignment: .trailing)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(true)
            }
        }
    }
}

struct MetalView: NSViewRepresentable {
    let viewModel: FlowVizViewModel
    
    func makeNSView(context: Context) -> MTKView {
        let metalView = MTKView()
        metalView.device = MTLCreateSystemDefaultDevice()
        
        // Performance optimizations
        metalView.preferredFramesPerSecond = 60
        metalView.isPaused = false
        metalView.enableSetNeedsDisplay = false
        metalView.framebufferOnly = true // Optimize for non-readable framebuffer
        metalView.autoResizeDrawable = true // Auto-handle resize
        metalView.clearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.08, alpha: 0.1)
        
        // Use triple buffering for smoother rendering
        metalView.presentsWithTransaction = false
        
        viewModel.setupRenderer(metalView: metalView)
        
        return metalView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        // Ensure valid bounds before calculating drawable size
        guard nsView.bounds.width > 0, 
              nsView.bounds.height > 0,
              nsView.bounds.width.isFinite,
              nsView.bounds.height.isFinite else {
            return
        }
        
        let scale = nsView.window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2.0
        guard scale.isFinite, scale > 0 else { return }
        
        let width = max(1, Int(nsView.bounds.width * scale))
        let height = max(1, Int(nsView.bounds.height * scale))
        
        // Validate final size before setting
        guard width > 0, height > 0, 
              width < 16384, height < 16384 else {
            return
        }
        
        let newSize = CGSize(width: width, height: height)
        
        // Only update if size actually changed to avoid unnecessary work
        if nsView.drawableSize != newSize {
            nsView.drawableSize = newSize
        }
    }
}

// MARK: - Floating Controls Panel

struct FloatingControlsPanel: View {
    @ObservedObject var viewModel: FlowVizViewModel
    let onClose: () -> Void
    @State private var flowExpanded = true
    @State private var particlesExpanded = false
    @State private var sceneExpanded = false
    
    var body: some View {
        VStack(spacing: 0) {
            // Header with close button
            HStack {
                Text("Controls")
                    .font(.headline)
                    .fontWeight(.semibold)
                
                Spacer()
                
                Button(action: onClose) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }
                .buttonStyle(PlainButtonStyle())
                .help("Hide Controls")
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            
            Divider()
            
            ScrollView(.vertical, showsIndicators: true) {
                LazyVStack(spacing: 12, pinnedViews: []) {
                // Flow Settings
                DisclosureGroup(isExpanded: $flowExpanded) {
                    VStack(spacing: 12) {
                        // Visualization Mode
                        VStack(alignment: .leading, spacing: 6) {
                            Label("Mode", systemImage: "eye")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            
                            Picker("", selection: $viewModel.visualizationMode) {
                                ForEach(VisualizationMode.allCases, id: \.self) { mode in
                                    Text(mode.rawValue).tag(mode)
                                }
                            }
                            .pickerStyle(MenuPickerStyle())
                            .onChange(of: viewModel.visualizationMode) { _ in
                                viewModel.updateVelocityField()
                            }
                        }
                        
                        Divider()
                        
                        // Flow Speed
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Label("Speed", systemImage: "speedometer")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                Spacer()
                                Text(String(format: "%.1fx", viewModel.flowSpeed))
                                    .font(.subheadline.monospacedDigit())
                                    .foregroundColor(.primary)
                            }
                            Slider(value: $viewModel.flowSpeed, in: 0.1...3.0)
                                .onChange(of: viewModel.flowSpeed) { viewModel.setFlowSpeed($0) }
                        }
                        
                        Divider()
                        
                        // Display Options
                        VStack(spacing: 8) {
                            Toggle(isOn: $viewModel.showVelocityField) {
                                Label("Velocity Field", systemImage: "arrow.up.and.down.and.arrow.left.and.right")
                            }
                            .toggleStyle(SwitchToggleStyle())
                            
                            Toggle(isOn: $viewModel.showTrajectories) {
                                Label("Trajectories", systemImage: "point.3.connected.trianglepath.dotted")
                            }
                            .toggleStyle(SwitchToggleStyle())
                        }
                        
                        Divider()
                        
                        // Start & Goal Points
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Circle()
                                    .fill(.green)
                                    .frame(width: 8, height: 8)
                                Text("Start:")
                                    .font(.subheadline)
                                Spacer()
                                Text("(\(String(format: "%.2f", viewModel.startPoint.x)), \(String(format: "%.2f", viewModel.startPoint.y)))")
                                    .font(.caption.monospacedDigit())
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
                                    .font(.caption.monospacedDigit())
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .padding(.top, 8)
                } label: {
                    Label("Flow Settings", systemImage: "wind")
                        .font(.headline)
                }
                .disclosureGroupStyle(InspectorDisclosureStyle())
                
                // Particle Settings
                DisclosureGroup(isExpanded: $particlesExpanded) {
                    VStack(spacing: 12) {
                        // Particle Count
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Label("Count", systemImage: "number")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                Spacer()
                                Text("\(Int(viewModel.particleCount))")
                                    .font(.subheadline.monospacedDigit())
                                    .foregroundColor(.primary)
                            }
                            Slider(value: $viewModel.particleCount, in: 1000...50000, step: 1000)
                                .onChange(of: viewModel.particleCount) { viewModel.setParticleCount(Int($0)) }
                        }
                        
                        Divider()
                        
                        // Playback Control
                        Toggle(isOn: $viewModel.isPlaying) {
                            Label(viewModel.isPlaying ? "Playing" : "Paused", 
                                  systemImage: viewModel.isPlaying ? "play.fill" : "pause.fill")
                        }
                        .toggleStyle(SwitchToggleStyle())
                        .onChange(of: viewModel.isPlaying) { viewModel.setPlaying($0) }
                    }
                    .padding(.top, 8)
                } label: {
                    Label("Particles", systemImage: "sparkles")
                        .font(.headline)
                }
                .disclosureGroupStyle(InspectorDisclosureStyle())
                
                // Scene Settings
                DisclosureGroup(isExpanded: $sceneExpanded) {
                    VStack(spacing: 12) {
                        // Current Scene
                        HStack {
                            Text("Scene:")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(viewModel.currentScene)
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(.accentColor)
                        }
                        
                        // Reset Button
                        Button(action: { viewModel.resetScene() }) {
                            Label("Reset Scene", systemImage: "arrow.clockwise")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(BorderedButtonStyle())
                        .controlSize(.regular)
                        
                        Divider()
                        
                        // Obstacles
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Label("Obstacles", systemImage: "circle.dashed")
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                                Spacer()
                                Text("\(viewModel.obstacles.count)")
                                    .font(.subheadline.monospacedDigit())
                                    .foregroundColor(.primary)
                            }
                            
                            if !viewModel.obstacles.isEmpty {
                                VStack(spacing: 6) {
                                    ForEach(Array(viewModel.obstacles.enumerated()), id: \.element.id) { index, _ in
                                        HStack {
                                            Circle()
                                                .fill(.red.opacity(0.3))
                                                .frame(width: 8, height: 8)
                                                .overlay(Circle().strokeBorder(.red, lineWidth: 1))
                                            Text("Obstacle \(index + 1)")
                                                .font(.subheadline)
                                            Spacer()
                                            Button(action: { viewModel.removeObstacle(at: index) }) {
                                                Image(systemName: "xmark.circle.fill")
                                                    .foregroundColor(.red.opacity(0.8))
                                            }
                                            .buttonStyle(PlainButtonStyle())
                                        }
                                        .padding(.vertical, 2)
                                    }
                                }
                            }
                            
                            HStack(spacing: 4) {
                                Image(systemName: "hand.tap")
                                    .font(.caption)
                                Text("Click on visualization to add")
                                    .font(.caption)
                            }
                            .foregroundColor(.secondary)
                            .padding(.top, 4)
                        }
                    }
                    .padding(.top, 8)
                } label: {
                    Label("Scene", systemImage: "square.on.square")
                        .font(.headline)
                }
                .disclosureGroupStyle(InspectorDisclosureStyle())
                }
                .padding(12)
            }
            .frame(maxHeight: .infinity)
        }
        .clipped()
    }
}

// MARK: - Custom Disclosure Group Style

struct InspectorDisclosureStyle: DisclosureGroupStyle {
    func makeBody(configuration: Configuration) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                configuration.isExpanded.toggle()
            } label: {
                HStack {
                    configuration.label
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .rotationEffect(.degrees(configuration.isExpanded ? 90 : 0))
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(PlainButtonStyle())
            .padding(.vertical, 8)
            .padding(.horizontal, 12)
            .background(Color.accentColor.opacity(0.08))
            .cornerRadius(8)
            
            if configuration.isExpanded {
                configuration.content
                    .padding(.horizontal, 12)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
    }
}

#Preview {
    ContentView()
}
