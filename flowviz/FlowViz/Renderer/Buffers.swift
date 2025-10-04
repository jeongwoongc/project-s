import Metal
import simd
import QuartzCore

/// Manages Metal buffers for efficient GPU data transfer
class BufferManager {
    private let device: MTLDevice
    private var buffers: [String: MTLBuffer] = [:]
    
    init(device: MTLDevice) {
        self.device = device
    }
    
    /// Create or update a buffer with the given data
    func createBuffer<T>(name: String, data: [T], options: MTLResourceOptions = .storageModeShared) -> MTLBuffer? {
        let size = MemoryLayout<T>.stride * data.count
        
        if let existingBuffer = buffers[name], existingBuffer.length >= size {
            // Reuse existing buffer if it's large enough
            let bufferPointer = existingBuffer.contents().bindMemory(to: T.self, capacity: data.count)
            for (index, element) in data.enumerated() {
                bufferPointer[index] = element
            }
            return existingBuffer
        } else {
            // Create new buffer
            guard let buffer = device.makeBuffer(length: size, options: options) else {
                return nil
            }
            
            let bufferPointer = buffer.contents().bindMemory(to: T.self, capacity: data.count)
            for (index, element) in data.enumerated() {
                bufferPointer[index] = element
            }
            
            buffers[name] = buffer
            return buffer
        }
    }
    
    /// Get an existing buffer by name
    func getBuffer(name: String) -> MTLBuffer? {
        return buffers[name]
    }
    
    /// Create an empty buffer of the specified size
    func createEmptyBuffer(name: String, size: Int, options: MTLResourceOptions = .storageModeShared) -> MTLBuffer? {
        guard let buffer = device.makeBuffer(length: size, options: options) else {
            return nil
        }
        
        buffers[name] = buffer
        return buffer
    }
    
    /// Update buffer contents
    func updateBuffer<T>(name: String, data: [T]) -> Bool {
        guard let buffer = buffers[name] else { return false }
        
        let requiredSize = MemoryLayout<T>.stride * data.count
        guard buffer.length >= requiredSize else { return false }
        
        let bufferPointer = buffer.contents().bindMemory(to: T.self, capacity: data.count)
        for (index, element) in data.enumerated() {
            bufferPointer[index] = element
        }
        
        return true
    }
    
    /// Clear all buffers
    func clearBuffers() {
        buffers.removeAll()
    }
}

/// Utility for managing texture data
class TextureManager {
    private let device: MTLDevice
    private var textures: [String: MTLTexture] = [:]
    
    init(device: MTLDevice) {
        self.device = device
    }
    
    /// Create a 2D texture for velocity field visualization
    func createVelocityFieldTexture(name: String, width: Int, height: Int) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rg32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            return nil
        }
        
        textures[name] = texture
        return texture
    }
    
    /// Update velocity field texture with data
    func updateVelocityFieldTexture(name: String, data: [simd_float2], width: Int, height: Int) -> Bool {
        guard let texture = textures[name] else { return false }
        
        let region = MTLRegionMake2D(0, 0, width, height)
        let bytesPerRow = MemoryLayout<simd_float2>.stride * width
        
        data.withUnsafeBytes { bytes in
            texture.replace(region: region, mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: bytesPerRow)
        }
        
        return true
    }
    
    /// Get texture by name
    func getTexture(name: String) -> MTLTexture? {
        return textures[name]
    }
    
    /// Clear all textures
    func clearTextures() {
        textures.removeAll()
    }
}

/// Performance monitoring for GPU operations
class PerformanceMonitor {
    private var frameTime: CFTimeInterval = 0
    private var lastFrameTime: CFTimeInterval = 0
    private var frameCount: Int = 0
    private var fps: Double = 0
    
    func beginFrame() {
        lastFrameTime = CACurrentMediaTime()
    }
    
    func endFrame() {
        let currentTime = CACurrentMediaTime()
        frameTime = currentTime - lastFrameTime
        frameCount += 1
        
        // Update FPS every 60 frames
        if frameCount % 60 == 0 {
            fps = 1.0 / frameTime
        }
    }
    
    var currentFPS: Double {
        return fps
    }
    
    var currentFrameTime: Double {
        return frameTime * 1000.0 // Convert to milliseconds
    }
}

