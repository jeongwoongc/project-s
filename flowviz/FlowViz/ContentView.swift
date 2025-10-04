import SwiftUI
import MetalKit

struct ContentView: View {
    @StateObject private var viewModel = FlowVizViewModel()
    @State private var showControls = false
    @State private var isHoveringControls = false
    @State private var autoHideTimer: Timer?
    
    var body: some View {
        ZStack {
            // Main Metal rendering view
            MetalView(viewModel: viewModel)
                .ignoresSafeArea(.all)
            
            // Overlay UI - allow clicks through empty space
            VStack {
                // Top HUD
                HStack {
                    HUD(viewModel: viewModel)
                    Spacer()
                        .allowsHitTesting(false)
                }
                .padding()
                
                Spacer()
                    .allowsHitTesting(false)
            }
            .allowsHitTesting(true)
            
            // Floating controls toggle button (when panel is hidden)
            if !showControls {
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        FloatingControlsButton(action: {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                                showControls = true
                            }
                        })
                        .padding(.trailing, 8)
                        .padding(.bottom, 8)
                    }
                }
                .allowsHitTesting(true)
            }
            
            // Compact floating controls panel
            if showControls {
                VStack {
                    Spacer()
                    HStack(spacing: 0) {
                        Spacer()
                            .allowsHitTesting(false)
                        
                        CompactControlsPanel(
                            viewModel: viewModel,
                            isHovering: $isHoveringControls,
                            onClose: {
                                withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                                    showControls = false
                                }
                            }
                        )
                        .frame(width: 320)
                        .frame(maxHeight: 450)
                        .background(
                            LiquidGlassBackground(isHovering: isHoveringControls)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 20)
                                .strokeBorder(Color.white.opacity(0.15), lineWidth: 1.0)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 20))
                        .shadow(color: Color.black.opacity(0.25), radius: 20, x: 0, y: 8)
                        .padding(.trailing, 12)
                        .padding(.bottom, 12)
                        .transition(.move(edge: .trailing).combined(with: .opacity))
                        .onHover { hovering in
                            withAnimation(.easeInOut(duration: 0.3)) {
                                isHoveringControls = hovering
                            }
                        }
                    }
                }
                .allowsHitTesting(true)
            }
        }
    }
}

struct MetalView: NSViewRepresentable {
    let viewModel: FlowVizViewModel
    
    func makeCoordinator() -> Coordinator {
        Coordinator(viewModel: viewModel)
    }
    
    func makeNSView(context: Context) -> MTKView {
        let metalView = MTKView()
        metalView.device = MTLCreateSystemDefaultDevice()
        metalView.preferredFramesPerSecond = 60
        metalView.isPaused = false
        metalView.enableSetNeedsDisplay = false
        metalView.clearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.08, alpha: 1.0)
        
        viewModel.setupRenderer(metalView: metalView)
        
        // Add click gesture recognizer for obstacle placement
        let clickGesture = NSClickGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleClick(_:)))
        metalView.addGestureRecognizer(clickGesture)
        
        return metalView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        // Update view if needed
    }
    
    class Coordinator: NSObject {
        let viewModel: FlowVizViewModel
        
        init(viewModel: FlowVizViewModel) {
            self.viewModel = viewModel
        }
        
        @objc func handleClick(_ gesture: NSClickGestureRecognizer) {
            guard let view = gesture.view else { return }
            let location = gesture.location(in: view)
            
            // Convert to normalized coordinates (0-1)
            let normalizedPoint = CGPoint(
                x: location.x / view.bounds.width,
                y: 1.0 - (location.y / view.bounds.height) // Flip Y
            )
            
            viewModel.addObstacle(at: normalizedPoint)
        }
    }
}

// MARK: - Liquid Glass Background

struct LiquidGlassBackground: View {
    let isHovering: Bool
    
    var body: some View {
        ZStack {
            // Simple ultra-thin material for transparency
            RoundedRectangle(cornerRadius: 20)
                .fill(.ultraThinMaterial)
            
            // Subtle top highlight for glass effect
            VStack(spacing: 0) {
                LinearGradient(
                    colors: [
                        Color.white.opacity(0.05),
                        Color.clear
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .frame(height: 40)
                Spacer()
            }
            .clipShape(RoundedRectangle(cornerRadius: 20))
        }
    }
}

// MARK: - Floating Controls Button

struct FloatingControlsButton: View {
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: "slider.horizontal.3")
                    .font(.title2)
                Text("Controls")
                    .font(.caption)
            }
            .foregroundColor(.white)
            .padding(.horizontal, 14)
            .padding(.vertical, 11)
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(.ultraThinMaterial)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(Color.white.opacity(0.15), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.25), radius: 8)
        }
        .buttonStyle(PlainButtonStyle())
        .help("Show controls")
    }
}

// MARK: - Compact Controls Panel

struct CompactControlsPanel: View {
    @ObservedObject var viewModel: FlowVizViewModel
    @Binding var isHovering: Bool
    let onClose: () -> Void
    @State private var selectedTab = 0
    
    var body: some View {
        VStack(spacing: 0) {
            // Header with close button
            HStack {
                Text("Controls")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                
                Spacer()
                
                Button(action: onClose) {
                    ZStack {
                        Circle()
                            .fill(.ultraThinMaterial)
                            .frame(width: 28, height: 28)
                        
                        Image(systemName: "xmark")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.white.opacity(0.8))
                    }
                }
                .buttonStyle(PlainButtonStyle())
                .help("Hide controls")
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            
            Divider()
                .opacity(0.3)
            
            // Compact tab selector
            Picker("", selection: $selectedTab) {
                Image(systemName: "wind").tag(0)
                Image(systemName: "sparkles").tag(1)
                Image(systemName: "square.on.square").tag(2)
            }
            .pickerStyle(SegmentedPickerStyle())
            .padding(.horizontal, 16)
            .padding(.top, 8)
            
            // Compact content
            ScrollView {
                VStack(spacing: 12) {
                    switch selectedTab {
                    case 0:
                        CompactFlowControls(viewModel: viewModel)
                    case 1:
                        CompactParticleControls(viewModel: viewModel)
                    case 2:
                        CompactSceneControls(viewModel: viewModel)
                    default:
                        EmptyView()
                    }
                }
                .padding(16)
            }
        }
    }
}

// MARK: - Compact Control Views

struct CompactFlowControls: View {
    @ObservedObject var viewModel: FlowVizViewModel
    
    var body: some View {
        VStack(spacing: 10) {
            // Visualization Mode
            VStack(alignment: .leading, spacing: 4) {
                Text("Mode")
                    .font(.caption)
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
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Speed")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(String(format: "%.1fx", viewModel.flowSpeed))
                        .font(.caption.monospacedDigit())
                }
                Slider(value: $viewModel.flowSpeed, in: 0.1...3.0)
                    .onChange(of: viewModel.flowSpeed) { viewModel.setFlowSpeed($0) }
            }
            
            Divider()
            
            // Toggles
            Toggle("Velocity Field", isOn: $viewModel.showVelocityField)
                .font(.caption)
                .toggleStyle(SwitchToggleStyle())
            
            Toggle("Trajectories", isOn: $viewModel.showTrajectories)
                .font(.caption)
                .toggleStyle(SwitchToggleStyle())
        }
    }
}

struct CompactParticleControls: View {
    @ObservedObject var viewModel: FlowVizViewModel
    
    var body: some View {
        VStack(spacing: 10) {
            // Particle Count
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Count")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("\(Int(viewModel.particleCount))")
                        .font(.caption.monospacedDigit())
                }
                Slider(value: $viewModel.particleCount, in: 1000...50000, step: 1000)
                    .onChange(of: viewModel.particleCount) { viewModel.setParticleCount(Int($0)) }
            }
            
            Divider()
            
            // Playing toggle
            Toggle("Playing", isOn: $viewModel.isPlaying)
                .font(.caption)
                .toggleStyle(SwitchToggleStyle())
                .onChange(of: viewModel.isPlaying) { viewModel.setPlaying($0) }
        }
    }
}

struct CompactSceneControls: View {
    @ObservedObject var viewModel: FlowVizViewModel
    
    var body: some View {
        VStack(spacing: 10) {
            // Quick actions
            HStack(spacing: 8) {
                Button(action: { viewModel.resetScene() }) {
                    Label("Reset", systemImage: "arrow.clockwise")
                        .font(.caption)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(BorderedButtonStyle())
                .controlSize(.small)
            }
            
            Divider()
            
            // Obstacles count
            HStack {
                Text("Obstacles")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Text("\(viewModel.obstacles.count)")
                    .font(.caption.monospacedDigit())
            }
            
            if !viewModel.obstacles.isEmpty {
                ForEach(Array(viewModel.obstacles.enumerated()), id: \.element.id) { index, _ in
                    HStack {
                        Text("Obstacle \(index + 1)")
                            .font(.caption2)
                        Spacer()
                        Button(action: { viewModel.removeObstacle(at: index) }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.caption)
                                .foregroundColor(.red)
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                }
            }
            
            Text("Click on visualization to add")
                .font(.caption2)
                .foregroundColor(.secondary)
                .italic()
        }
    }
}

#Preview {
    ContentView()
}
