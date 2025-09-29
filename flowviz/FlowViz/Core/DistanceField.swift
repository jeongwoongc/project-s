import Foundation
import simd

/// Manages signed distance field for obstacles
class DistanceField {
    let width: Int
    let height: Int
    private(set) var distanceData: [Float]
    private(set) var gradientData: [simd_float2]
    
    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.distanceData = Array(repeating: Float.greatestFiniteMagnitude, count: width * height)
        self.gradientData = Array(repeating: simd_float2(0, 0), count: width * height)
    }
    
    /// Update distance field with current obstacles
    func updateWithObstacles(_ obstacles: [Obstacle]) {
        // Initialize with maximum distance
        distanceData = Array(repeating: Float.greatestFiniteMagnitude, count: width * height)
        
        // Compute distance to each obstacle
        for y in 0..<height {
            for x in 0..<width {
                let gridPoint = simd_float2(
                    Float(x) / Float(width - 1),
                    Float(y) / Float(height - 1)
                )
                
                var minDistance = Float.greatestFiniteMagnitude
                
                for obstacle in obstacles {
                    let obstacleCenter = simd_float2(Float(obstacle.center.x), Float(obstacle.center.y))
                    let distance: Float
                    
                    switch obstacle.type {
                    case .circle:
                        distance = GeometryUtils.distanceToCircle(
                            point: gridPoint,
                            center: obstacleCenter,
                            radius: Float(obstacle.radius)
                        )
                    case .rectangle:
                        let size = simd_float2(Float(obstacle.radius * 2), Float(obstacle.radius * 2))
                        distance = GeometryUtils.distanceToRectangle(
                            point: gridPoint,
                            center: obstacleCenter,
                            size: size
                        )
                    }
                    
                    minDistance = min(minDistance, distance)
                }
                
                distanceData[y * width + x] = minDistance
            }
        }
        
        // Compute gradients
        computeGradients()
        
        // Apply smoothing for better gradients
        smoothDistanceField()
    }
    
    /// Get distance at grid coordinates
    func getDistance(x: Int, y: Int) -> Float {
        guard x >= 0 && x < width && y >= 0 && y < height else {
            return 0.0
        }
        return distanceData[y * width + x]
    }
    
    /// Get gradient at grid coordinates
    func getGradient(x: Int, y: Int) -> simd_float2 {
        guard x >= 0 && x < width && y >= 0 && y < height else {
            return simd_float2(0, 0)
        }
        return gradientData[y * width + x]
    }
    
    /// Sample distance at normalized coordinates using bilinear interpolation
    func sampleDistance(at point: simd_float2) -> Float {
        let x = point.x * Float(width - 1)
        let y = point.y * Float(height - 1)
        
        let x0 = Int(floor(x))
        let y0 = Int(floor(y))
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        
        let fx = x - Float(x0)
        let fy = y - Float(y0)
        
        let d00 = getDistance(x: x0, y: y0)
        let d10 = getDistance(x: x1, y: y0)
        let d01 = getDistance(x: x0, y: y1)
        let d11 = getDistance(x: x1, y: y1)
        
        let d0 = d00 + (d10 - d00) * fx
        let d1 = d01 + (d11 - d01) * fx
        
        return d0 + (d1 - d0) * fy
    }
    
    /// Sample gradient at normalized coordinates using bilinear interpolation
    func sampleGradient(at point: simd_float2) -> simd_float2 {
        let x = point.x * Float(width - 1)
        let y = point.y * Float(height - 1)
        
        let x0 = Int(floor(x))
        let y0 = Int(floor(y))
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        
        let fx = x - Float(x0)
        let fy = y - Float(y0)
        
        let g00 = getGradient(x: x0, y: y0)
        let g10 = getGradient(x: x1, y: y0)
        let g01 = getGradient(x: x0, y: y1)
        let g11 = getGradient(x: x1, y: y1)
        
        let g0 = GeometryUtils.mix(g00, g10, fx)
        let g1 = GeometryUtils.mix(g01, g11, fx)
        
        return GeometryUtils.mix(g0, g1, fy)
    }
    
    /// Check if point is inside any obstacle
    func isInsideObstacle(at point: simd_float2) -> Bool {
        return sampleDistance(at: point) <= 0
    }
    
    /// Get safety factor (0 = inside obstacle, 1 = far from obstacles)
    func getSafetyFactor(at point: simd_float2, safetyRadius: Float = 0.05) -> Float {
        let distance = sampleDistance(at: point)
        return GeometryUtils.clamp(distance / safetyRadius, 0.0, 1.0)
    }
    
    // MARK: - Private Methods
    
    private func computeGradients() {
        for y in 0..<height {
            for x in 0..<width {
                let gradient = computeGradientAt(x: x, y: y)
                gradientData[y * width + x] = gradient
            }
        }
    }
    
    private func computeGradientAt(x: Int, y: Int) -> simd_float2 {
        // Use central differences where possible
        let x0 = max(0, x - 1)
        let x1 = min(width - 1, x + 1)
        let y0 = max(0, y - 1)
        let y1 = min(height - 1, y + 1)
        
        let dx = (getDistance(x: x1, y: y) - getDistance(x: x0, y: y)) / Float(x1 - x0)
        let dy = (getDistance(x: x, y: y1) - getDistance(x: x, y: y0)) / Float(y1 - y0)
        
        return simd_float2(dx, dy)
    }
    
    private func smoothDistanceField() {
        var smoothedData = distanceData
        
        // Apply Gaussian-like smoothing
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let center = getDistance(x: x, y: y)
                let neighbors = [
                    getDistance(x: x - 1, y: y),
                    getDistance(x: x + 1, y: y),
                    getDistance(x: x, y: y - 1),
                    getDistance(x: x, y: y + 1)
                ]
                
                let diagonals = [
                    getDistance(x: x - 1, y: y - 1),
                    getDistance(x: x + 1, y: y - 1),
                    getDistance(x: x - 1, y: y + 1),
                    getDistance(x: x + 1, y: y + 1)
                ]
                
                let neighborSum = neighbors.reduce(0, +)
                let diagonalSum = diagonals.reduce(0, +)
                
                // Weighted average: center has highest weight
                let smoothed = (center * 0.4 + neighborSum * 0.1 + diagonalSum * 0.05) / 1.0
                smoothedData[y * width + x] = smoothed
            }
        }
        
        distanceData = smoothedData
    }
    
    /// Perform fast marching method for better distance field computation
    func fastMarchingMethod() {
        // TODO: Implement fast marching for more accurate distance fields
        // This would provide better quality distance fields for complex obstacle configurations
    }
    
    /// Export distance field as texture data for visualization
    func exportAsTextureData() -> [Float] {
        // Normalize distances for visualization
        let maxDistance = distanceData.max() ?? 1.0
        return distanceData.map { distance in
            return GeometryUtils.clamp(distance / maxDistance, 0.0, 1.0)
        }
    }
    
    /// Clear the distance field
    func clear() {
        distanceData = Array(repeating: Float.greatestFiniteMagnitude, count: width * height)
        gradientData = Array(repeating: simd_float2(0, 0), count: width * height)
    }
}
