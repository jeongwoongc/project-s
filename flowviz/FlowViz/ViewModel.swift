import SwiftUI
import MetalKit
import Combine

class FlowVizViewModel: ObservableObject {
    // MARK: - Published Properties
    @Published var isPlaying = true
    @Published var particleCount: Float = 20000
    @Published var flowSpeed: Float = 1.2
    @Published var visualizationMode: VisualizationMode = .flowMatching
    @Published var showVelocityField = true
    @Published var showTrajectories = true
    @Published var currentScene = "Default"
    
    // MARK: - Flow Properties
    @Published var startPoint = CGPoint(x: 0.2, y: 0.5)
    @Published var goalPoint = CGPoint(x: 0.8, y: 0.5)
    @Published var obstacles: [Obstacle] = []
    
    // MARK: - Rendering
    let renderer = MetalRenderer()
    private var metalView: MTKView?
    
    // MARK: - Core Components
    private var velocityGrid: VelocityGrid?
    private var distanceField: DistanceField?
    private var modelIO = ModelIO()
    
    init() {
        setupDefaultScene()
    }
    
    func setupRenderer(metalView: MTKView) {
        self.metalView = metalView
        if metalView.device == nil {
            metalView.device = MTLCreateSystemDefaultDevice()
        }
        guard let device = metalView.device else {
            assertionFailure("No Metal device available on this machine.")
            return
        }
        renderer.setup(device: device, view: metalView)
        metalView.delegate = renderer
        
        // Initialize core components
        velocityGrid = VelocityGrid(width: 128, height: 128)
        distanceField = DistanceField(width: 128, height: 128)
        
        updateVelocityField()
    }
    
    private func setupDefaultScene() {
        // Add a default obstacle
        obstacles = [
            Obstacle(center: CGPoint(x: 0.5, y: 0.3), radius: 0.1, type: .circle)
        ]
    }
    
    func updateVelocityField() {
        guard let velocityGrid = velocityGrid,
              let distanceField = distanceField else { return }
        
        // Update distance field with current obstacles
        distanceField.updateWithObstacles(obstacles)
        
        // Compute flow field based on current mode
        switch visualizationMode {
        case .flowMatching:
            velocityGrid.computeFlowMatchingField(
                start: startPoint,
                goal: goalPoint,
                distanceField: distanceField
            )
        case .diffusion:
            velocityGrid.computeDiffusionField(
                start: startPoint,
                goal: goalPoint,
                distanceField: distanceField
            )
        case .neuralODE:
            if let model = modelIO.loadedModel {
                velocityGrid.computeNeuralField(model: model, distanceField: distanceField)
            }
        case .vortexStorm:
            velocityGrid.computeVortexStormField(
                start: startPoint,
                goal: goalPoint,
                distanceField: distanceField
            )
        }
        
        // Update renderer with new velocity field, scaled by flowSpeed for visual impact
        let scaledField = velocityGrid.velocityData.map { $0 * flowSpeed }
        renderer.updateVelocityField(scaledField)
    }
    
    // MARK: - Renderer Control

    func setParticleCount(_ count: Int) {
        renderer.setParticleCount(Int(count))
    }

    func setPlaying(_ playing: Bool) {
        renderer.setPlaying(playing)
    }

    func setFlowSpeed(_ speed: Float) {
        // Update local state
        flowSpeed = speed
        // Recompute or rescale velocity field to reflect speed changes
        updateVelocityField()
    }
    
    // MARK: - User Interactions
    func addObstacle(at point: CGPoint) {
        let obstacle = Obstacle(center: point, radius: 0.05, type: .circle)
        obstacles.append(obstacle)
        updateVelocityField()
    }
    
    func removeObstacle(at index: Int) {
        guard index < obstacles.count else { return }
        obstacles.remove(at: index)
        updateVelocityField()
    }
    
    func updateStartPoint(_ point: CGPoint) {
        startPoint = point
        updateVelocityField()
    }
    
    func updateGoalPoint(_ point: CGPoint) {
        goalPoint = point
        updateVelocityField()
    }
    
    func resetScene() {
        obstacles.removeAll()
        startPoint = CGPoint(x: 0.2, y: 0.5)
        goalPoint = CGPoint(x: 0.8, y: 0.5)
        setupDefaultScene()
        updateVelocityField()
    }
    
    func loadScene(_ sceneName: String) {
        // TODO: Load from JSON presets
        currentScene = sceneName
    }
    
    func saveScene(_ sceneName: String) {
        // TODO: Save to JSON presets
    }
}

enum VisualizationMode: String, CaseIterable {
    case flowMatching = "Flow Matching"
    case diffusion = "Diffusion"
    case neuralODE = "Neural ODE"
    case vortexStorm = "Vortex Storm"
}

struct Obstacle {
    let id = UUID()
    var center: CGPoint
    var radius: Double
    var type: ObstacleType
}

enum ObstacleType {
    case circle
    case rectangle
}
