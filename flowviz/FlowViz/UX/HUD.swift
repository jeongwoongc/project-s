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
            PerformanceIndicator(fps: viewModel.currentFPS, frameTime: viewModel.currentFrameTime)
            
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
                InfoPopover(viewModel: viewModel)
            }
        }
    }
}

struct PerformanceIndicator: View {
    let fps: Double
    let frameTime: Double
    
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
        .animation(.easeInOut(duration: 0.3), value: fps)
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
}

struct InfoPopover: View {
    @ObservedObject var viewModel: FlowVizViewModel
    
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
                
                Text("• \(viewModel.visualizationMode.rawValue) mode")
                Text("• \(Int(viewModel.particleCount)) particles")
                Text("• \(String(format: "%.1fx", viewModel.flowSpeed)) flow speed")
                Text("• \(Int(viewModel.currentFPS)) FPS")
            }
            .font(.caption)
            .foregroundColor(.secondary)
        }
        .padding()
        .frame(width: 270)
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
                // Invisible interaction area for obstacle placement
                Color.clear
                    .contentShape(Rectangle())
                    .gesture(
                        SpatialTapGesture()
                            .onEnded { event in
                                let location = event.location
                                let normalizedPoint = CGPoint(
                                    x: location.x / geometry.size.width,
                                    y: location.y / geometry.size.height
                                )
                                viewModel.addObstacle(at: normalizedPoint)
                            }
                    )
                
                // Start point indicator (Green Pin)
                StartPointMarker()
                    .position(
                        x: viewModel.startPoint.x * geometry.size.width,
                        y: viewModel.startPoint.y * geometry.size.height
                    )
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                let normalizedPoint = CGPoint(
                                    x: max(0, min(1, value.location.x / geometry.size.width)),
                                    y: max(0, min(1, value.location.y / geometry.size.height))
                                )
                                viewModel.updateStartPoint(normalizedPoint)
                            }
                    )
                    .help("Drag to move start point")
                
                // Goal point indicator (Red Target)
                GoalPointMarker()
                    .position(
                        x: viewModel.goalPoint.x * geometry.size.width,
                        y: viewModel.goalPoint.y * geometry.size.height
                    )
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                let normalizedPoint = CGPoint(
                                    x: max(0, min(1, value.location.x / geometry.size.width)),
                                    y: max(0, min(1, value.location.y / geometry.size.height))
                                )
                                viewModel.updateGoalPoint(normalizedPoint)
                            }
                    )
                    .help("Drag to move goal point")
                
                // Obstacle indicators (Liquid Glass Style)
                ForEach(Array(viewModel.obstacles.enumerated()), id: \.element.id) { index, obstacle in
                    LiquidGlassObstacle(radius: obstacle.radius * geometry.size.width)
                        .position(
                            x: obstacle.center.x * geometry.size.width,
                            y: obstacle.center.y * geometry.size.height
                        )
                }
            }
        }
    }
}

// MARK: - Point Markers (Subtle Particle Style)

struct StartPointMarker: View {
    var body: some View {
        ZStack {
            // Very subtle outer glow (matches particle glow)
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            Color(red: 0.4, green: 1.0, blue: 0.6).opacity(0.15),
                            Color.clear
                        ],
                        center: .center,
                        startRadius: 0,
                        endRadius: 30
                    )
                )
                .frame(width: 60, height: 60)
                .blur(radius: 6)
            
            // Subtle ring (barely visible)
            Circle()
                .stroke(
                    Color(red: 0.5, green: 1.0, blue: 0.7).opacity(0.2),
                    lineWidth: 1
                )
                .frame(width: 24, height: 24)
            
            // Core particle
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            Color(red: 0.6, green: 1.0, blue: 0.8).opacity(0.8),
                            Color(red: 0.4, green: 0.9, blue: 0.6).opacity(0.4),
                            Color.clear
                        ],
                        center: .center,
                        startRadius: 0,
                        endRadius: 6
                    )
                )
                .frame(width: 12, height: 12)
                .blur(radius: 1)
            
            // Bright center (like a particle)
            Circle()
                .fill(Color(red: 0.7, green: 1.0, blue: 0.9).opacity(0.9))
                .frame(width: 4, height: 4)
                .blur(radius: 0.5)
            
            // Interaction area (invisible but captures gestures)
            Circle()
                .fill(Color.clear)
                .contentShape(Circle())
                .frame(width: 80, height: 80)
        }
    }
}

struct GoalPointMarker: View {
    var body: some View {
        ZStack {
            // Very subtle outer glow (matches particle glow)
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            Color(red: 1.0, green: 0.5, blue: 0.6).opacity(0.15),
                            Color.clear
                        ],
                        center: .center,
                        startRadius: 0,
                        endRadius: 30
                    )
                )
                .frame(width: 60, height: 60)
                .blur(radius: 6)
            
            // Subtle ring (barely visible)
            Circle()
                .stroke(
                    Color(red: 1.0, green: 0.6, blue: 0.7).opacity(0.2),
                    lineWidth: 1
                )
                .frame(width: 24, height: 24)
            
            // Core particle
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            Color(red: 1.0, green: 0.7, blue: 0.8).opacity(0.8),
                            Color(red: 1.0, green: 0.5, blue: 0.6).opacity(0.4),
                            Color.clear
                        ],
                        center: .center,
                        startRadius: 0,
                        endRadius: 6
                    )
                )
                .frame(width: 12, height: 12)
                .blur(radius: 1)
            
            // Bright center (like a particle)
            Circle()
                .fill(Color(red: 1.0, green: 0.8, blue: 0.9).opacity(0.9))
                .frame(width: 4, height: 4)
                .blur(radius: 0.5)
            
            // Interaction area (invisible but captures gestures)
            Circle()
                .fill(Color.clear)
                .contentShape(Circle())
                .frame(width: 80, height: 80)
        }
    }
}

// MARK: - Liquid Glass Obstacle (Invisible)

struct LiquidGlassObstacle: View {
    let radius: CGFloat
    
    var body: some View {
        // Invisible but still defines the obstacle position
        Circle()
            .fill(Color.clear)
            .frame(width: radius * 2, height: radius * 2)
    }
}

#Preview {
    ZStack {
        Color.black
        HUD(viewModel: FlowVizViewModel())
    }
}
