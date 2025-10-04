import SwiftUI

struct HUD: View {
    @ObservedObject var viewModel: FlowVizViewModel
    @State private var showingInfo = false
    
    var body: some View {
        HStack(spacing: 12) {
            // Play/Pause button
            Button(action: { 
                viewModel.isPlaying.toggle()
                viewModel.setPlaying(viewModel.isPlaying)
            }) {
                ZStack {
                    Circle()
                        .fill(.ultraThinMaterial)
                        .frame(width: 44, height: 44)
                    
                    Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title2)
                        .foregroundColor(.white)
                }
            }
            .buttonStyle(PlainButtonStyle())
            .help(viewModel.isPlaying ? "Pause" : "Play")
            
            // Current mode indicator
            VStack(alignment: .leading, spacing: 2) {
                Text(viewModel.visualizationMode.rawValue)
                    .font(.headline)
                    .foregroundColor(.white)
                
                Text("\(Int(viewModel.particleCount)) particles")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Capsule().fill(.ultraThinMaterial))
            
            // Performance indicator
            PerformanceIndicator()
            
            // Info button
            Button(action: { showingInfo.toggle() }) {
                ZStack {
                    Circle()
                        .fill(.ultraThinMaterial)
                        .frame(width: 44, height: 44)
                    
                    Image(systemName: "info.circle.fill")
                        .font(.title2)
                        .foregroundColor(.white)
                }
            }
            .buttonStyle(PlainButtonStyle())
            .help("Show controls")
            .popover(isPresented: $showingInfo) {
                InfoPopover()
            }
        }
    }
}

struct PerformanceIndicator: View {
    @State private var fps: Double = 60.0
    @State private var frameTime: Double = 16.7
    
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                Circle()
                    .fill(fpsColor)
                    .frame(width: 8, height: 8)
                
                Text("\(Int(fps)) FPS")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.white)
            }
            
            Text("\(String(format: "%.1f", frameTime))ms")
                .font(.system(.caption2, design: .monospaced))
                .foregroundColor(.white.opacity(0.8))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Capsule().fill(.ultraThinMaterial))
        .onReceive(Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()) { _ in
            // Update performance metrics
            updatePerformanceMetrics()
        }
    }
    
    private var fpsColor: Color {
        if fps >= 55 {
            return .green
        } else if fps >= 30 {
            return .yellow
        } else {
            return .red
        }
    }
    
    private func updatePerformanceMetrics() {
        // Simulate performance metrics - in real implementation,
        // these would come from the MetalRenderer
        fps = Double.random(in: 55...62)
        frameTime = 1000.0 / fps
    }
}

struct InfoPopover: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("FlowViz Controls")
                .font(.headline)
                .bold()
            
            VStack(alignment: .leading, spacing: 8) {
                InfoRow(icon: "hand.draw", text: "Click to add obstacles")
                InfoRow(icon: "arrow.up.and.down.and.arrow.left.and.right", text: "Drag start/goal points")
                InfoRow(icon: "play.fill", text: "Space to play/pause")
                InfoRow(icon: "slider.horizontal.3", text: "Use controls panel to adjust settings")
            }
            
            Divider()
            
            VStack(alignment: .leading, spacing: 4) {
                Text("Current Settings")
                    .font(.subheadline)
                    .bold()
                
                Text("• Flow Matching mode")
                Text("• 10,000 particles")
                Text("• Real-time velocity field")
            }
            .font(.caption)
            .foregroundColor(.secondary)
        }
        .padding()
        .frame(width: 250)
    }
}

struct InfoRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .frame(width: 16)
                .foregroundColor(.blue)
            
            Text(text)
                .font(.caption)
        }
    }
}

struct StatusBar: View {
    @ObservedObject var viewModel: FlowVizViewModel
    
    var body: some View {
        HStack {
            // Flow field status
            HStack(spacing: 4) {
                Circle()
                    .fill(.green)
                    .frame(width: 6, height: 6)
                
                Text("Flow Field Active")
                    .font(.caption)
            }
            
            Spacer()
            
            // Particle count
            Text("\(Int(viewModel.particleCount)) particles")
                .font(.caption)
            
            Spacer()
            
            // Current coordinates (would be updated on mouse move)
            Text("(0.50, 0.50)")
                .font(.system(.caption, design: .monospaced))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
        .background(.ultraThinMaterial)
        .foregroundColor(.white)
    }
}

// Interactive overlay for adding obstacles and moving points
struct InteractionOverlay: View {
    @ObservedObject var viewModel: FlowVizViewModel
    @State private var dragOffset = CGSize.zero
    @State private var isDraggingStart = false
    @State private var isDraggingGoal = false
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Invisible interaction area
                Color.clear
                    .contentShape(Rectangle())
                    .onTapGesture { location in
                        let normalizedPoint = CGPoint(
                            x: location.x / geometry.size.width,
                            y: location.y / geometry.size.height
                        )
                        viewModel.addObstacle(at: normalizedPoint)
                    }
                
                // Start point indicator
                Circle()
                    .fill(.green)
                    .frame(width: 20, height: 20)
                    .position(
                        x: viewModel.startPoint.x * geometry.size.width,
                        y: viewModel.startPoint.y * geometry.size.height
                    )
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                let normalizedPoint = CGPoint(
                                    x: value.location.x / geometry.size.width,
                                    y: value.location.y / geometry.size.height
                                )
                                viewModel.updateStartPoint(normalizedPoint)
                            }
                    )
                
                // Goal point indicator
                Circle()
                    .fill(.red)
                    .frame(width: 20, height: 20)
                    .position(
                        x: viewModel.goalPoint.x * geometry.size.width,
                        y: viewModel.goalPoint.y * geometry.size.height
                    )
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                let normalizedPoint = CGPoint(
                                    x: value.location.x / geometry.size.width,
                                    y: value.location.y / geometry.size.height
                                )
                                viewModel.updateGoalPoint(normalizedPoint)
                            }
                    )
                
                // Obstacle indicators
                ForEach(Array(viewModel.obstacles.enumerated()), id: \.element.id) { index, obstacle in
                    Circle()
                        .stroke(.red, lineWidth: 2)
                        .frame(
                            width: obstacle.radius * 2 * geometry.size.width,
                            height: obstacle.radius * 2 * geometry.size.height
                        )
                        .position(
                            x: obstacle.center.x * geometry.size.width,
                            y: obstacle.center.y * geometry.size.height
                        )
                }
            }
        }
    }
}

#Preview {
    ZStack {
        Color.black
        HUD(viewModel: FlowVizViewModel())
    }
}
