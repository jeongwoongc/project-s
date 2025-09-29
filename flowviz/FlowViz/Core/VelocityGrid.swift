import Foundation
import simd

/// Manages velocity field computation and storage
class VelocityGrid {
    let width: Int
    let height: Int
    private(set) var velocityData: [simd_float2]
    
    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.velocityData = Array(repeating: simd_float2(0, 0), count: width * height)
    }
    
    /// Get velocity at grid coordinates
    func getVelocity(x: Int, y: Int) -> simd_float2 {
        guard x >= 0 && x < width && y >= 0 && y < height else {
            return simd_float2(0, 0)
        }
        return velocityData[y * width + x]
    }
    
    /// Set velocity at grid coordinates
    func setVelocity(x: Int, y: Int, velocity: simd_float2) {
        guard x >= 0 && x < width && y >= 0 && y < height else { return }
        velocityData[y * width + x] = velocity
    }
    
    /// Sample velocity at normalized coordinates using bilinear interpolation
    func sampleVelocity(at point: simd_float2) -> simd_float2 {
        let x = point.x * Float(width - 1)
        let y = point.y * Float(height - 1)
        
        let x0 = Int(floor(x))
        let y0 = Int(floor(y))
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        
        let fx = x - Float(x0)
        let fy = y - Float(y0)
        
        let v00 = getVelocity(x: x0, y: y0)
        let v10 = getVelocity(x: x1, y: y0)
        let v01 = getVelocity(x: x0, y: y1)
        let v11 = getVelocity(x: x1, y: y1)
        
        let v0 = GeometryUtils.mix(v00, v10, fx)
        let v1 = GeometryUtils.mix(v01, v11, fx)
        
        return GeometryUtils.mix(v0, v1, fy)
    }
    
    /// Compute flow matching velocity field
    func computeFlowMatchingField(start: CGPoint, goal: CGPoint, distanceField: DistanceField) {
        let startPoint = simd_float2(Float(start.x), Float(start.y))
        let goalPoint = simd_float2(Float(goal.x), Float(goal.y))
        
        for y in 0..<height {
            for x in 0..<width {
                let gridPoint = simd_float2(
                    Float(x) / Float(width - 1),
                    Float(y) / Float(height - 1)
                )
                
                // Basic flow matching: direct interpolation between start and goal
                let t = computeFlowTime(point: gridPoint, start: startPoint, goal: goalPoint)
                let velocity = computeFlowVelocity(point: gridPoint, start: startPoint, goal: goalPoint, t: t)
                
                // Apply obstacle avoidance
                let modifiedVelocity = applyObstacleAvoidance(
                    velocity: velocity,
                    point: gridPoint,
                    distanceField: distanceField
                )
                
                setVelocity(x: x, y: y, velocity: modifiedVelocity)
            }
        }
    }
    
    /// Compute diffusion-style velocity field
    func computeDiffusionField(start: CGPoint, goal: CGPoint, distanceField: DistanceField) {
        let startPoint = simd_float2(Float(start.x), Float(start.y))
        let goalPoint = simd_float2(Float(goal.x), Float(goal.y))
        
        for y in 0..<height {
            for x in 0..<width {
                let gridPoint = simd_float2(
                    Float(x) / Float(width - 1),
                    Float(y) / Float(height - 1)
                )
                
                // Diffusion-style: gradient toward goal with noise
                let toGoal = goalPoint - gridPoint
                let distance = length(toGoal)
                
                var velocity = GeometryUtils.safeNormalize(toGoal) * (1.0 / (1.0 + distance))
                
                // Add some curl for visual interest
                let curl = simd_float2(-velocity.y, velocity.x) * 0.2
                velocity += curl
                
                // Apply obstacle avoidance
                let modifiedVelocity = applyObstacleAvoidance(
                    velocity: velocity,
                    point: gridPoint,
                    distanceField: distanceField
                )
                
                setVelocity(x: x, y: y, velocity: modifiedVelocity)
            }
        }
    }
    
    /// Compute neural ODE velocity field (placeholder for ML model)
    func computeNeuralField(model: Any, distanceField: DistanceField) {
        // TODO: Implement Core ML model inference
        // For now, create a simple swirling field
        
        let center = simd_float2(0.5, 0.5)
        
        for y in 0..<height {
            for x in 0..<width {
                let gridPoint = simd_float2(
                    Float(x) / Float(width - 1),
                    Float(y) / Float(height - 1)
                )
                
                let offset = gridPoint - center
                let distance = length(offset)
                
                // Create a swirling pattern
                let angle = atan2(offset.y, offset.x)
                let swirl = simd_float2(-sin(angle), cos(angle)) * (1.0 - distance)
                
                // Apply obstacle avoidance
                let modifiedVelocity = applyObstacleAvoidance(
                    velocity: swirl * 0.5,
                    point: gridPoint,
                    distanceField: distanceField
                )
                
                setVelocity(x: x, y: y, velocity: modifiedVelocity)
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func computeFlowTime(point: simd_float2, start: simd_float2, goal: simd_float2) -> Float {
        // Simple linear interpolation parameter
        let startToPoint = point - start
        let startToGoal = goal - start
        
        let projection = dot(startToPoint, startToGoal) / dot(startToGoal, startToGoal)
        return GeometryUtils.clamp(projection, 0.0, 1.0)
    }
    
    private func computeFlowVelocity(point: simd_float2, start: simd_float2, goal: simd_float2, t: Float) -> simd_float2 {
        // Optimal transport velocity: derivative of the interpolating path
        let direction = GeometryUtils.safeNormalize(goal - start)
        let speed = 1.0 - t * 0.5 // Slow down as we approach the goal
        
        return direction * speed
    }
    
    private func applyObstacleAvoidance(velocity: simd_float2, point: simd_float2, distanceField: DistanceField) -> simd_float2 {
        let distance = distanceField.sampleDistance(at: point)
        let gradient = distanceField.sampleGradient(at: point)
        
        // If we're close to an obstacle, add repulsion
        let repulsionStrength: Float = 2.0
        let repulsionRadius: Float = 0.1
        
        if distance < repulsionRadius {
            let repulsionForce = GeometryUtils.safeNormalize(gradient) * (repulsionStrength * (repulsionRadius - distance) / repulsionRadius)
            return velocity + repulsionForce
        }
        
        return velocity
    }
    
    /// Clear the velocity field
    func clear() {
        velocityData = Array(repeating: simd_float2(0, 0), count: width * height)
    }
    
    /// Apply smoothing filter to reduce noise
    func smooth(iterations: Int = 1) {
        for _ in 0..<iterations {
            var smoothedData = velocityData
            
            for y in 1..<(height - 1) {
                for x in 1..<(width - 1) {
                    let center = getVelocity(x: x, y: y)
                    let neighbors = [
                        getVelocity(x: x - 1, y: y),
                        getVelocity(x: x + 1, y: y),
                        getVelocity(x: x, y: y - 1),
                        getVelocity(x: x, y: y + 1)
                    ]
                    
                    let neighborSum = neighbors.reduce(simd_float2(0, 0), +)
                    let smoothed = (center * 0.6 + neighborSum * 0.1)
                    
                    smoothedData[y * width + x] = smoothed
                }
            }
            
            velocityData = smoothedData
        }
    }
}
