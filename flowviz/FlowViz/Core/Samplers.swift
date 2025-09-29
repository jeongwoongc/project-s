import Foundation
import simd

/// Trajectory sampling and integration utilities
class TrajectorySampler {
    
    /// Sample trajectory using Euler integration
    static func sampleTrajectoryEuler(
        start: simd_float2,
        velocityField: VelocityGrid,
        stepSize: Float = 0.01,
        maxSteps: Int = 1000
    ) -> [simd_float2] {
        var trajectory: [simd_float2] = [start]
        var currentPoint = start
        
        for _ in 0..<maxSteps {
            let velocity = velocityField.sampleVelocity(at: currentPoint)
            
            // Check if velocity is too small (reached equilibrium)
            if length(velocity) < 1e-4 {
                break
            }
            
            // Euler step
            currentPoint += velocity * stepSize
            
            // Boundary check
            if currentPoint.x < 0 || currentPoint.x > 1 || currentPoint.y < 0 || currentPoint.y > 1 {
                break
            }
            
            trajectory.append(currentPoint)
        }
        
        return trajectory
    }
    
    /// Sample trajectory using Runge-Kutta 4th order integration (more accurate)
    static func sampleTrajectoryRK4(
        start: simd_float2,
        velocityField: VelocityGrid,
        stepSize: Float = 0.01,
        maxSteps: Int = 1000
    ) -> [simd_float2] {
        var trajectory: [simd_float2] = [start]
        var currentPoint = start
        
        for _ in 0..<maxSteps {
            let k1 = velocityField.sampleVelocity(at: currentPoint)
            let k2 = velocityField.sampleVelocity(at: currentPoint + k1 * stepSize * 0.5)
            let k3 = velocityField.sampleVelocity(at: currentPoint + k2 * stepSize * 0.5)
            let k4 = velocityField.sampleVelocity(at: currentPoint + k3 * stepSize)
            
            let velocity = (k1 + k2 * 2 + k3 * 2 + k4) / 6
            
            // Check if velocity is too small
            if length(velocity) < 1e-4 {
                break
            }
            
            // RK4 step
            currentPoint += velocity * stepSize
            
            // Boundary check
            if currentPoint.x < 0 || currentPoint.x > 1 || currentPoint.y < 0 || currentPoint.y > 1 {
                break
            }
            
            trajectory.append(currentPoint)
        }
        
        return trajectory
    }
    
    /// Sample multiple trajectories from a grid of starting points
    static func sampleTrajectoryGrid(
        gridSize: Int,
        velocityField: VelocityGrid,
        stepSize: Float = 0.01,
        maxSteps: Int = 500
    ) -> [[simd_float2]] {
        var trajectories: [[simd_float2]] = []
        
        for i in 0..<gridSize {
            for j in 0..<gridSize {
                let start = simd_float2(
                    Float(i) / Float(gridSize - 1),
                    Float(j) / Float(gridSize - 1)
                )
                
                let trajectory = sampleTrajectoryRK4(
                    start: start,
                    velocityField: velocityField,
                    stepSize: stepSize,
                    maxSteps: maxSteps
                )
                
                if trajectory.count > 1 {
                    trajectories.append(trajectory)
                }
            }
        }
        
        return trajectories
    }
}

/// Streamline computation for flow visualization
class StreamlineSampler {
    
    /// Generate streamlines from seed points
    static func generateStreamlines(
        seeds: [simd_float2],
        velocityField: VelocityGrid,
        stepSize: Float = 0.005,
        maxLength: Float = 2.0
    ) -> [[simd_float2]] {
        return seeds.compactMap { seed in
            generateStreamline(
                from: seed,
                velocityField: velocityField,
                stepSize: stepSize,
                maxLength: maxLength
            )
        }
    }
    
    /// Generate a single streamline
    static func generateStreamline(
        from start: simd_float2,
        velocityField: VelocityGrid,
        stepSize: Float = 0.005,
        maxLength: Float = 2.0
    ) -> [simd_float2]? {
        var streamline: [simd_float2] = [start]
        var currentPoint = start
        var totalLength: Float = 0
        
        while totalLength < maxLength {
            let velocity = velocityField.sampleVelocity(at: currentPoint)
            let speed = length(velocity)
            
            // Stop if velocity is too small
            if speed < 1e-4 {
                break
            }
            
            // Adaptive step size based on velocity magnitude
            let adaptiveStepSize = min(stepSize, stepSize / speed)
            let direction = velocity / speed
            
            let nextPoint = currentPoint + direction * adaptiveStepSize
            
            // Boundary check
            if nextPoint.x < 0 || nextPoint.x > 1 || nextPoint.y < 0 || nextPoint.y > 1 {
                break
            }
            
            currentPoint = nextPoint
            totalLength += adaptiveStepSize
            streamline.append(currentPoint)
        }
        
        return streamline.count > 1 ? streamline : nil
    }
    
    /// Generate evenly spaced streamlines
    static func generateEvenlySpacedStreamlines(
        velocityField: VelocityGrid,
        spacing: Float = 0.05,
        stepSize: Float = 0.005,
        maxLength: Float = 2.0
    ) -> [[simd_float2]] {
        var streamlines: [[simd_float2]] = []
        var occupancyGrid = Array(repeating: false, count: 64 * 64) // Lower resolution for spacing check
        
        let gridSize = 64
        let cellSize = 1.0 / Float(gridSize)
        
        for i in 0..<gridSize {
            for j in 0..<gridSize {
                let gridIndex = i * gridSize + j
                if occupancyGrid[gridIndex] {
                    continue
                }
                
                let start = simd_float2(
                    Float(j) * cellSize + cellSize * 0.5,
                    Float(i) * cellSize + cellSize * 0.5
                )
                
                if let streamline = generateStreamline(
                    from: start,
                    velocityField: velocityField,
                    stepSize: stepSize,
                    maxLength: maxLength
                ) {
                    streamlines.append(streamline)
                    
                    // Mark nearby cells as occupied
                    markOccupiedCells(streamline: streamline, occupancyGrid: &occupancyGrid, gridSize: gridSize, spacing: spacing)
                }
            }
        }
        
        return streamlines
    }
    
    private static func markOccupiedCells(
        streamline: [simd_float2],
        occupancyGrid: inout [Bool],
        gridSize: Int,
        spacing: Float
    ) {
        let spacingInCells = Int(spacing * Float(gridSize))
        
        for point in streamline {
            let gridX = Int(point.x * Float(gridSize))
            let gridY = Int(point.y * Float(gridSize))
            
            for dy in -spacingInCells...spacingInCells {
                for dx in -spacingInCells...spacingInCells {
                    let x = gridX + dx
                    let y = gridY + dy
                    
                    if x >= 0 && x < gridSize && y >= 0 && y < gridSize {
                        occupancyGrid[y * gridSize + x] = true
                    }
                }
            }
        }
    }
}

/// Particle advection for dynamic visualization
class ParticleAdvector {
    
    /// Advect particles through velocity field
    static func advectParticles(
        particles: inout [simd_float2],
        velocityField: VelocityGrid,
        deltaTime: Float,
        bounds: simd_float4 = simd_float4(0, 0, 1, 1) // minX, minY, maxX, maxY
    ) {
        for i in 0..<particles.count {
            let velocity = velocityField.sampleVelocity(at: particles[i])
            var newPosition = particles[i] + velocity * deltaTime
            
            // Handle boundary conditions (wrap around)
            if newPosition.x < bounds.x { newPosition.x = bounds.z }
            if newPosition.x > bounds.z { newPosition.x = bounds.x }
            if newPosition.y < bounds.y { newPosition.y = bounds.w }
            if newPosition.y > bounds.w { newPosition.y = bounds.y }
            
            particles[i] = newPosition
        }
    }
    
    /// Advect particles with life cycle management
    static func advectParticlesWithLife(
        particles: inout [(position: simd_float2, life: Float, maxLife: Float)],
        velocityField: VelocityGrid,
        deltaTime: Float,
        respawnRate: Float = 0.1
    ) {
        for i in 0..<particles.count {
            let velocity = velocityField.sampleVelocity(at: particles[i].position)
            var newPosition = particles[i].position + velocity * deltaTime
            var life = particles[i].life - deltaTime
            
            // Handle boundaries and respawn
            if newPosition.x < 0 || newPosition.x > 1 || newPosition.y < 0 || newPosition.y > 1 || life <= 0 {
                // Respawn particle
                if Float.random(in: 0...1) < respawnRate {
                    newPosition = simd_float2(Float.random(in: 0...1), Float.random(in: 0...1))
                    life = particles[i].maxLife
                }
            }
            
            particles[i] = (position: newPosition, life: life, maxLife: particles[i].maxLife)
        }
    }
}

/// Utility for sampling patterns and distributions
class SamplingPatterns {
    
    /// Generate Poisson disk sampling for even distribution
    static func poissonDiskSampling(
        bounds: simd_float4, // minX, minY, maxX, maxY
        radius: Float,
        maxAttempts: Int = 30
    ) -> [simd_float2] {
        var points: [simd_float2] = []
        var activeList: [simd_float2] = []
        
        let cellSize = radius / sqrt(2)
        let gridWidth = Int((bounds.z - bounds.x) / cellSize) + 1
        let gridHeight = Int((bounds.w - bounds.y) / cellSize) + 1
        var grid = Array(repeating: -1, count: gridWidth * gridHeight)
        
        // Start with a random point
        let initialPoint = simd_float2(
            Float.random(in: bounds.x...bounds.z),
            Float.random(in: bounds.y...bounds.w)
        )
        
        points.append(initialPoint)
        activeList.append(initialPoint)
        
        let gridX = Int((initialPoint.x - bounds.x) / cellSize)
        let gridY = Int((initialPoint.y - bounds.y) / cellSize)
        grid[gridY * gridWidth + gridX] = 0
        
        while !activeList.isEmpty {
            let randomIndex = Int.random(in: 0..<activeList.count)
            let point = activeList[randomIndex]
            var found = false
            
            for _ in 0..<maxAttempts {
                let angle = Float.random(in: 0...(2 * Float.pi))
                let distance = Float.random(in: radius...(2 * radius))
                
                let newPoint = simd_float2(
                    point.x + cos(angle) * distance,
                    point.y + sin(angle) * distance
                )
                
                if newPoint.x >= bounds.x && newPoint.x <= bounds.z &&
                   newPoint.y >= bounds.y && newPoint.y <= bounds.w &&
                   isValidPoint(newPoint, points: points, grid: grid, gridWidth: gridWidth, cellSize: cellSize, radius: radius, bounds: bounds) {
                    
                    points.append(newPoint)
                    activeList.append(newPoint)
                    
                    let gridX = Int((newPoint.x - bounds.x) / cellSize)
                    let gridY = Int((newPoint.y - bounds.y) / cellSize)
                    grid[gridY * gridWidth + gridX] = points.count - 1
                    
                    found = true
                    break
                }
            }
            
            if !found {
                activeList.remove(at: randomIndex)
            }
        }
        
        return points
    }
    
    private static func isValidPoint(
        _ point: simd_float2,
        points: [simd_float2],
        grid: [Int],
        gridWidth: Int,
        cellSize: Float,
        radius: Float,
        bounds: simd_float4
    ) -> Bool {
        let gridX = Int((point.x - bounds.x) / cellSize)
        let gridY = Int((point.y - bounds.y) / cellSize)
        
        let searchRadius = 2
        for dy in -searchRadius...searchRadius {
            for dx in -searchRadius...searchRadius {
                let x = gridX + dx
                let y = gridY + dy
                
                if x >= 0 && x < gridWidth && y >= 0 && y < (grid.count / gridWidth) {
                    let index = grid[y * gridWidth + x]
                    if index >= 0 {
                        let existingPoint = points[index]
                        if length(point - existingPoint) < radius {
                            return false
                        }
                    }
                }
            }
        }
        
        return true
    }
}
