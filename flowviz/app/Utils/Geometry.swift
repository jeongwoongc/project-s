import Foundation
import simd

/// Geometric utilities for flow visualization
struct GeometryUtils {
    
    /// Calculate distance from point to circle
    static func distanceToCircle(point: simd_float2, center: simd_float2, radius: Float) -> Float {
        let distance = length(point - center)
        return max(0, distance - radius)
    }
    
    /// Calculate distance from point to rectangle
    static func distanceToRectangle(point: simd_float2, center: simd_float2, size: simd_float2) -> Float {
        let offset = abs(point - center) - size * 0.5
        let outsideDistance = length(max(offset, simd_float2(0, 0)))
        let insideDistance = min(max(offset.x, offset.y), 0.0)
        return outsideDistance + insideDistance
    }
    
    /// Bilinear interpolation for sampling 2D grids
    static func bilinearInterpolation(grid: [[simd_float2]], x: Float, y: Float) -> simd_float2 {
        let width = Float(grid[0].count)
        let height = Float(grid.count)
        
        let fx = x * (width - 1)
        let fy = y * (height - 1)
        
        let x0 = Int(floor(fx))
        let y0 = Int(floor(fy))
        let x1 = min(x0 + 1, Int(width) - 1)
        let y1 = min(y0 + 1, Int(height) - 1)
        
        let dx = fx - Float(x0)
        let dy = fy - Float(y0)
        
        let v00 = grid[y0][x0]
        let v10 = grid[y0][x1]
        let v01 = grid[y1][x0]
        let v11 = grid[y1][x1]
        
        let v0 = mix(v00, v10, dx)
        let v1 = mix(v01, v11, dx)
        
        return mix(v0, v1, dy)
    }
    
    /// Linear interpolation between two vectors
    static func mix(_ a: simd_float2, _ b: simd_float2, _ t: Float) -> simd_float2 {
        return a + (b - a) * t
    }
    
    /// Normalize vector with safe handling of zero vectors
    static func safeNormalize(_ vector: simd_float2) -> simd_float2 {
        let len = length(vector)
        return len > 1e-6 ? vector / len : simd_float2(0, 0)
    }
    
    /// Calculate gradient at a point using central differences
    static func gradient(field: [[Float]], x: Int, y: Int) -> simd_float2 {
        let width = field[0].count
        let height = field.count
        
        let x0 = max(0, x - 1)
        let x1 = min(width - 1, x + 1)
        let y0 = max(0, y - 1)
        let y1 = min(height - 1, y + 1)
        
        let dx = (field[y][x1] - field[y][x0]) / Float(x1 - x0)
        let dy = (field[y1][x] - field[y0][x]) / Float(y1 - y0)
        
        return simd_float2(dx, dy)
    }
    
    /// Convert normalized coordinates to grid indices
    static func normalizedToGrid(_ point: simd_float2, gridSize: simd_int2) -> simd_int2 {
        let x = Int(point.x * Float(gridSize.x - 1))
        let y = Int(point.y * Float(gridSize.y - 1))
        return simd_int2(
            Int32(clamp(x, 0, Int(gridSize.x) - 1)),
            Int32(clamp(y, 0, Int(gridSize.y) - 1))
        )
    }
    
    /// Clamp value to range
    static func clamp<T: Comparable>(_ value: T, _ min: T, _ max: T) -> T {
        return Swift.min(Swift.max(value, min), max)
    }
    
    /// Calculate smooth step function
    static func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
        let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    }
    
    /// Generate random point in unit circle
    static func randomPointInCircle() -> simd_float2 {
        let angle = Float.random(in: 0...(2 * Float.pi))
        let radius = sqrt(Float.random(in: 0...1))
        return simd_float2(cos(angle) * radius, sin(angle) * radius)
    }
    
    /// Check if point is inside circle
    static func isInsideCircle(point: simd_float2, center: simd_float2, radius: Float) -> Bool {
        return length(point - center) <= radius
    }
    
    /// Check if point is inside rectangle
    static func isInsideRectangle(point: simd_float2, center: simd_float2, size: simd_float2) -> Bool {
        let offset = abs(point - center)
        return offset.x <= size.x * 0.5 && offset.y <= size.y * 0.5
    }
}

/// Color utilities for visualization
struct ColorUtils {
    
    /// Convert velocity to color using HSV color space
    static func velocityToColor(velocity: simd_float2) -> simd_float3 {
        let speed = length(velocity)
        let angle = atan2(velocity.y, velocity.x)
        
        // Map angle to hue (0-360 degrees)
        let hue = (angle + Float.pi) / (2 * Float.pi) * 360
        
        // Map speed to saturation and value
        let saturation: Float = min(speed * 2, 1.0)
        let value: Float = 0.8 + min(speed * 0.4, 0.2)
        
        return hsvToRgb(h: hue, s: saturation, v: value)
    }
    
    /// Convert HSV to RGB
    static func hsvToRgb(h: Float, s: Float, v: Float) -> simd_float3 {
        let c = v * s
        let x = c * (1 - abs((h / 60).truncatingRemainder(dividingBy: 2) - 1))
        let m = v - c
        
        var rgb: simd_float3
        
        if h < 60 {
            rgb = simd_float3(c, x, 0)
        } else if h < 120 {
            rgb = simd_float3(x, c, 0)
        } else if h < 180 {
            rgb = simd_float3(0, c, x)
        } else if h < 240 {
            rgb = simd_float3(0, x, c)
        } else if h < 300 {
            rgb = simd_float3(x, 0, c)
        } else {
            rgb = simd_float3(c, 0, x)
        }
        
        return rgb + simd_float3(m, m, m)
    }
    
    /// Generate gradient colors for flow visualization
    static func generateFlowGradient(steps: Int) -> [simd_float3] {
        var colors: [simd_float3] = []
        
        for i in 0..<steps {
            let t = Float(i) / Float(steps - 1)
            
            // Blue to cyan to yellow to red gradient
            let color: simd_float3
            if t < 0.25 {
                // Blue to cyan
                let localT = t / 0.25
                color = mix(simd_float3(0, 0, 1), simd_float3(0, 1, 1), localT)
            } else if t < 0.5 {
                // Cyan to green
                let localT = (t - 0.25) / 0.25
                color = mix(simd_float3(0, 1, 1), simd_float3(0, 1, 0), localT)
            } else if t < 0.75 {
                // Green to yellow
                let localT = (t - 0.5) / 0.25
                color = mix(simd_float3(0, 1, 0), simd_float3(1, 1, 0), localT)
            } else {
                // Yellow to red
                let localT = (t - 0.75) / 0.25
                color = mix(simd_float3(1, 1, 0), simd_float3(1, 0, 0), localT)
            }
            
            colors.append(color)
        }
        
        return colors
    }
    
    /// Mix two colors
    static func mix(_ a: simd_float3, _ b: simd_float3, _ t: Float) -> simd_float3 {
        return a + (b - a) * t
    }
}
