import Metal
import MetalKit
import simd

class MetalRenderer: NSObject, MTKViewDelegate {
    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var renderPipelineState: MTLRenderPipelineState!
    private var computePipelineState: MTLComputePipelineState!
    
    // Buffers
    private var particleBuffer: MTLBuffer!
    private var velocityFieldBuffer: MTLBuffer!
    private var uniformBuffer: MTLBuffer!
    
    // Particle system
    private var particleCount: Int = 10000
    private var particles: [Particle] = []
    
    // Uniforms
    private var uniforms = Uniforms()
    
    override init() {
        super.init()
    }
    
    func setup(device: MTLDevice, view: MTKView) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        setupShaders()
        setupBuffers()
        initializeParticles()
    }
    
    private func setupShaders() {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create Metal library")
        }
        
        // Render pipeline for particles
        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineDescriptor.vertexFunction = library.makeFunction(name: "particle_vertex")
        renderPipelineDescriptor.fragmentFunction = library.makeFunction(name: "particle_fragment")
        renderPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        renderPipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        renderPipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        renderPipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        
        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        } catch {
            fatalError("Could not create render pipeline state: \(error)")
        }
        
        // Compute pipeline for particle updates
        do {
            computePipelineState = try device.makeComputePipelineState(function: library.makeFunction(name: "update_particles")!)
        } catch {
            fatalError("Could not create compute pipeline state: \(error)")
        }
    }
    
    private func setupBuffers() {
        // Particle buffer
        let particleBufferSize = MemoryLayout<Particle>.stride * particleCount
        particleBuffer = device.makeBuffer(length: particleBufferSize, options: .storageModeShared)!
        
        // Velocity field buffer (128x128 grid)
        let velocityFieldSize = MemoryLayout<simd_float2>.stride * 128 * 128
        velocityFieldBuffer = device.makeBuffer(length: velocityFieldSize, options: .storageModeShared)!
        
        // Uniform buffer
        uniformBuffer = device.makeBuffer(length: MemoryLayout<Uniforms>.stride, options: .storageModeShared)!
    }
    
    private func initializeParticles() {
        particles.removeAll()
        
        for _ in 0..<particleCount {
            let particle = Particle(
                position: simd_float2(Float.random(in: 0...1), Float.random(in: 0...1)),
                velocity: simd_float2(0, 0),
                life: Float.random(in: 0.5...1.0),
                maxLife: Float.random(in: 2.0...5.0)
            )
            particles.append(particle)
        }
        
        updateParticleBuffer()
    }
    
    private func updateParticleBuffer() {
        let bufferPointer = particleBuffer.contents().bindMemory(to: Particle.self, capacity: particleCount)
        for (index, particle) in particles.enumerated() {
            bufferPointer[index] = particle
        }
    }
    
    func updateVelocityField(_ velocityData: [simd_float2]) {
        let bufferPointer = velocityFieldBuffer.contents().bindMemory(to: simd_float2.self, capacity: 128 * 128)
        for (index, velocity) in velocityData.enumerated() {
            bufferPointer[index] = velocity
        }
    }
    
    // MARK: - MTKViewDelegate
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        uniforms.screenSize = simd_float2(Float(size.width), Float(size.height))
        uniforms.aspectRatio = Float(size.width / size.height)
    }
    
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor else { return }
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        // Update particles with compute shader
        updateParticles(commandBuffer: commandBuffer)
        
        // Render particles
        renderParticles(commandBuffer: commandBuffer, renderPassDescriptor: renderPassDescriptor)
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    private func updateParticles(commandBuffer: MTLCommandBuffer) {
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setBuffer(particleBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(velocityFieldBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(uniformBuffer, offset: 0, index: 2)
        
        // Update uniforms
        uniforms.time += 1.0 / 60.0 // Assuming 60 FPS
        uniforms.deltaTime = 1.0 / 60.0
        let uniformPointer = uniformBuffer.contents().bindMemory(to: Uniforms.self, capacity: 1)
        uniformPointer[0] = uniforms
        
        let threadsPerGroup = MTLSize(width: 64, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (particleCount + 63) / 64, height: 1, depth: 1)
        
        computeEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
    }
    
    private func renderParticles(commandBuffer: MTLCommandBuffer, renderPassDescriptor: MTLRenderPassDescriptor) {
        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        renderEncoder.setRenderPipelineState(renderPipelineState)
        renderEncoder.setVertexBuffer(particleBuffer, offset: 0, index: 0)
        renderEncoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        renderEncoder.setFragmentBuffer(uniformBuffer, offset: 0, index: 0)
        
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: particleCount)
        renderEncoder.endEncoding()
    }
}

// MARK: - Data Structures

struct Particle {
    var position: simd_float2
    var velocity: simd_float2
    var life: Float
    var maxLife: Float
}

struct Uniforms {
    var time: Float = 0.0
    var deltaTime: Float = 0.0
    var screenSize: simd_float2 = simd_float2(1.0, 1.0)
    var aspectRatio: Float = 1.0
    var flowSpeed: Float = 1.0
    var particleSize: Float = 2.0
}
