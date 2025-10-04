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
    
    /// Compute chaotic vortex storm field with multiple interacting turbulent vortices
    func computeVortexStormField(start: CGPoint, goal: CGPoint, distanceField: DistanceField) {
        let startPoint = simd_float2(Float(start.x), Float(start.y))
        let goalPoint = simd_float2(Float(goal.x), Float(goal.y))
        
        // Create dynamic vortex centers that move and pulse
        let time = Float(Date().timeIntervalSince1970)
        var vortexCenters: [simd_float2] = []
        var vortexStrengths: [Float] = []
        var vortexRotations: [Float] = [] // 1.0 = clockwise, -1.0 = counter-clockwise
        
        // Primary vortex near start (pulsing)
        let vortex1 = startPoint + simd_float2(
            sin(time * 0.7) * 0.15,
            cos(time * 0.5) * 0.15
        )
        vortexCenters.append(vortex1)
        vortexStrengths.append(1.5 + sin(time * 1.3) * 0.5)
        vortexRotations.append(1.0)
        
        // Secondary vortex near goal (counter-rotating)
        let vortex2 = goalPoint + simd_float2(
            cos(time * 0.6) * 0.15,
            sin(time * 0.8) * 0.15
        )
        vortexCenters.append(vortex2)
        vortexStrengths.append(1.3 + cos(time * 1.1) * 0.4)
        vortexRotations.append(-1.0)
        
        // Wandering vortex in center (chaotic)
        let center = (startPoint + goalPoint) * 0.5
        let vortex3 = center + simd_float2(
            sin(time * 0.9) * 0.25,
            cos(time * 0.7) * 0.25
        )
        vortexCenters.append(vortex3)
        vortexStrengths.append(1.8 + sin(time * 1.5) * 0.7)
        vortexRotations.append(sin(time * 0.3) > 0 ? 1.0 : -1.0)
        
        // Two more smaller chaotic vortices
        for i in 0..<2 {
            let phase = Float(i) * 2.0
            let vortex = simd_float2(
                0.3 + 0.4 * cos(time * 0.4 + phase),
                0.3 + 0.4 * sin(time * 0.6 + phase)
            )
            vortexCenters.append(vortex)
            vortexStrengths.append(0.8 + sin(time * 1.2 + phase) * 0.4)
            vortexRotations.append(i % 2 == 0 ? 1.0 : -1.0)
        }
        
        for y in 0..<height {
            for x in 0..<width {
                let gridPoint = simd_float2(
                    Float(x) / Float(width - 1),
                    Float(y) / Float(height - 1)
                )
                
                var totalVelocity = simd_float2(0, 0)
                var totalWeight: Float = 0
                
                // Combine influences from all vortices
                for i in 0..<vortexCenters.count {
                    let offset = gridPoint - vortexCenters[i]
                    let distance = length(offset)
                    
                    if distance > 0.001 {
                        // Vortex strength falls off with distance but has a strong near field
                        let influence = vortexStrengths[i] / (1.0 + distance * 2.0)
                        let weight = pow(influence, 1.5)
                        
                        // Tangential velocity (rotation)
                        let tangent = simd_float2(-offset.y, offset.x) * vortexRotations[i]
                        let vortexVel = tangent * influence * 2.0
                        
                        // Radial component (attraction/repulsion based on distance)
                        let radialStrength: Float = (distance < 0.2) ? -0.3 : 0.1 // attract when close, repel when far
                        let radialVel = GeometryUtils.safeNormalize(offset) * radialStrength * influence
                        
                        totalVelocity += (vortexVel + radialVel) * weight
                        totalWeight += weight
                    }
                }
                
                // Normalize by total weight
                if totalWeight > 0 {
                    totalVelocity /= totalWeight
                }
                
                // Add turbulent chaos layer
                let turbulence = simd_float2(
                    sin(gridPoint.x * 8.0 + time * 2.0) * cos(gridPoint.y * 6.0 + time * 1.5),
                    cos(gridPoint.y * 7.0 + time * 1.8) * sin(gridPoint.x * 5.0 + time * 2.2)
                ) * 0.15
                
                totalVelocity += turbulence
                
                // Add subtle pull toward goal for navigation
                let toGoal = goalPoint - gridPoint
                let goalInfluence = GeometryUtils.safeNormalize(toGoal) * 0.1
                totalVelocity += goalInfluence
                
                // Apply obstacle avoidance
                let modifiedVelocity = applyObstacleAvoidance(
                    velocity: totalVelocity,
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
