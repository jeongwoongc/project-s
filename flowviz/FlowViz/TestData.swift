import Foundation
import simd

/// Test data and utilities for UI development
class TestData {
    
    /// Generate test velocity field for UI testing
    static func generateTestVelocityField(width: Int, height: Int) -> [simd_float2] {
        var velocities: [simd_float2] = []
        
        for y in 0..<height {
            for x in 0..<width {
                let normalizedX = Float(x) / Float(width - 1)
                let normalizedY = Float(y) / Float(height - 1)
                
                // Create a simple swirl pattern
                let centerX: Float = 0.5
                let centerY: Float = 0.5
                let dx = normalizedX - centerX
                let dy = normalizedY - centerY
                let distance = sqrt(dx * dx + dy * dy)
                
                // Swirl velocity
                let swirl = simd_float2(-dy, dx) * (1.0 - distance) * 0.5
                
                // Add flow toward goal
                let goalX: Float = 0.8
                let goalY: Float = 0.5
                let toGoal = simd_float2(goalX - normalizedX, goalY - normalizedY)
                let goalDistance = length(toGoal)
                let goalFlow = goalDistance > 0 ? toGoal / goalDistance * 0.3 : simd_float2(0, 0)
                
                velocities.append(swirl + goalFlow)
            }
        }
        
        return velocities
    }
    
    /// Generate test obstacles for UI testing
    static func generateTestObstacles() -> [Obstacle] {
        return [
            Obstacle(center: CGPoint(x: 0.3, y: 0.3), radius: 0.08, type: .circle),
            Obstacle(center: CGPoint(x: 0.7, y: 0.7), radius: 0.06, type: .circle),
            Obstacle(center: CGPoint(x: 0.5, y: 0.6), radius: 0.05, type: .circle)
        ]
    }
    
    /// Load test scene from JSON
    static func loadTestScene(named sceneName: String) -> TestScene? {
        guard let url = Bundle.main.url(forResource: "scenes", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let scenes = json["scenes"] as? [String: Any],
              let sceneData = scenes[sceneName] as? [String: Any] else {
            return nil
        }
        
        let startPoint = sceneData["start_point"] as? [Double] ?? [0.2, 0.5]
        let goalPoint = sceneData["goal_point"] as? [Double] ?? [0.8, 0.5]
        let obstacleData = sceneData["obstacles"] as? [[String: Any]] ?? []
        
        var obstacles: [Obstacle] = []
        for obsData in obstacleData {
            guard let center = obsData["center"] as? [Double],
                  let radius = obsData["radius"] as? Double else { continue }
            
            let obstacleType: ObstacleType = (obsData["type"] as? String == "rectangle") ? .rectangle : .circle
            obstacles.append(Obstacle(
                center: CGPoint(x: center[0], y: center[1]),
                radius: radius,
                type: obstacleType
            ))
        }
        
        return TestScene(
            name: sceneData["name"] as? String ?? sceneName,
            startPoint: CGPoint(x: startPoint[0], y: startPoint[1]),
            goalPoint: CGPoint(x: goalPoint[0], y: goalPoint[1]),
            obstacles: obstacles
        )
    }
}

struct TestScene {
    let name: String
    let startPoint: CGPoint
    let goalPoint: CGPoint
    let obstacles: [Obstacle]
}
