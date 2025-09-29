import SwiftUI
import MetalKit
import Combine

class FlowVizViewModel: ObservableObject {
    // MARK: - Published Properties
    @Published var isPlaying = true
    @Published var particleCount: Float = 10000
    @Published var flowSpeed: Float = 1.0
    @Published var visualizationMode: VisualizationMode = .flowMatching
    @Published var showVelocityField = true
    @Published var showTrajectories = true
    @Published var currentScene = "Default"
    
    // MARK: - Flow Properties
    @Published var startPoint = CGPoint(x: 0.2, y: 0.5)
    @Published var goalPoint = CGPoint(x: 0.8, y: 0.5)
    @Published var obstacles: [Obstacle] = []
    
    // MARK: - Rendering
    var renderer: MetalRenderer!
    private var metalView: MTKView?
    
    // MARK: - Core Components
    private var velocityGrid: VelocityGrid!
    private var distanceField: DistanceField!
    private var modelIO: ModelIO!
    
    init() {
        setupDefaultScene()
    }
    
    func setupMetal() {
        renderer = MetalRenderer()
    }
    
    func setupRenderer(metalView: MTKView) {
        self.metalView = metalView
        renderer.setup(device: metalView.device!, view: metalView)
        
        // Initialize core components
        velocityGrid = VelocityGrid(width: 128, height: 128)
        distanceField = DistanceField(width: 128, height: 128)
        modelIO = ModelIO()
        
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
        }
        
        // Update renderer with new velocity field
        renderer?.updateVelocityField(velocityGrid.velocityData)
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
