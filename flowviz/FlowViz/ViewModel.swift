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
    
    // MARK: - Performance Metrics
    @Published var currentFPS: Double = 60.0
    @Published var currentFrameTime: Double = 16.7
    
    // MARK: - Audio Reactivity (Disabled - needs more work)
    @Published var audioReactiveMode = false
    // let audioAnalyzer = AudioAnalyzer()
    // private var audioCancellables = Set<AnyCancellable>()
    
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
    
    // Throttling for drag updates
    private var updateWorkItem: DispatchWorkItem?
    private let updateQueue = DispatchQueue(label: "com.flowviz.velocityupdate", qos: .userInteractive)
    
    init() {
        setupDefaultScene()
        // setupAudioReactivity() // Disabled for now
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
        
        // Setup performance monitoring callback
        renderer.onPerformanceUpdate = { [weak self] fps, frameTime in
            DispatchQueue.main.async {
                self?.currentFPS = fps
                self?.currentFrameTime = frameTime
            }
        }
        
        // Initialize core components
        velocityGrid = VelocityGrid(width: 128, height: 128)
        distanceField = DistanceField(width: 128, height: 128)
        
        updateVelocityField()
    }
    
    private func setupDefaultScene() {
        // Start with no obstacles - let user add them
        obstacles = []
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
        // Ensure flowSpeed is finite before scaling
        let safeFlowSpeed = flowSpeed.isFinite ? flowSpeed : 1.0
        let scaledField = velocityGrid.velocityData.map { velocity in
            let scaled = velocity * safeFlowSpeed
            // Validate each velocity component
            return simd_float2(
                scaled.x.isFinite ? scaled.x : 0.0,
                scaled.y.isFinite ? scaled.y : 0.0
            )
        }
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
        // Validate and clamp speed
        guard speed.isFinite else { return }
        let clampedSpeed = max(0.1, min(speed, 10.0))
        
        // Update local state
        flowSpeed = clampedSpeed
        
        // Update renderer's flow speed
        renderer.setFlowSpeed(clampedSpeed)
        
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
        throttledUpdateVelocityField()
    }
    
    func updateGoalPoint(_ point: CGPoint) {
        goalPoint = point
        throttledUpdateVelocityField()
    }
    
    // Throttled version for drag operations to prevent lag
    private func throttledUpdateVelocityField() {
        // Cancel any pending update
        updateWorkItem?.cancel()
        
        // Create new work item
        let workItem = DispatchWorkItem { [weak self] in
            DispatchQueue.main.async {
                self?.updateVelocityField()
            }
        }
        
        updateWorkItem = workItem
        
        // Execute after a short delay (16ms = ~60fps)
        updateQueue.asyncAfter(deadline: .now() + 0.016, execute: workItem)
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
    
    // MARK: - Audio Reactivity
    
    func toggleAudioReactive(_ enabled: Bool) {
        // Disabled for now - needs more debugging
        audioReactiveMode = false
        print("⚠️ Audio reactive mode is currently disabled")
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

/*
// MARK: - Audio Analyzer (Embedded - Disabled)

class AudioAnalyzer: ObservableObject {
    // MARK: - Published Properties
    @Published var isActive = false
    @Published var bassLevel: Float = 0.0
    @Published var midLevel: Float = 0.0
    @Published var trebleLevel: Float = 0.0
    @Published var overallAmplitude: Float = 0.0
    
    // MARK: - Audio Components
    private var audioEngine: AVAudioEngine?
    private var inputNode: AVAudioInputNode?
    
    // FFT Configuration
    private let fftSize = 2048
    private var fftSetup: FFTSetup?
    private var windowFunction: [Float] = []
    
    // Buffers
    private var realBuffer: [Float] = []
    private var imagBuffer: [Float] = []
    private var magnitudes: [Float] = []
    
    // Smoothing
    private var smoothedBass: Float = 0.0
    private var smoothedMid: Float = 0.0
    private var smoothedTreble: Float = 0.0
    private let smoothingFactor: Float = 0.15
    
    init() {
        setupFFT()
    }
    
    private func setupFFT() {
        guard fftSize > 0, fftSize.isPowerOfTwo else {
            print("❌ FFT size must be a power of 2")
            return
        }
        
        let log2n = vDSP_Length(log2(Float(fftSize)))
        fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))
        
        guard fftSetup != nil else {
            print("❌ Failed to create FFT setup")
            return
        }
        
        windowFunction = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&windowFunction, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        
        realBuffer = [Float](repeating: 0, count: fftSize / 2)
        imagBuffer = [Float](repeating: 0, count: fftSize / 2)
        magnitudes = [Float](repeating: 0, count: fftSize / 2)
        
        print("✅ FFT setup complete: \(fftSize) samples")
    }
    
    func startCapture() {
        guard !isActive else { return }
        
        audioEngine = AVAudioEngine()
        guard let engine = audioEngine else { return }
        
        inputNode = engine.inputNode
        guard let input = inputNode else { return }
        
        let format = input.inputFormat(forBus: 0)
        
        input.installTap(onBus: 0, bufferSize: AVAudioFrameCount(fftSize), format: format) { [weak self] buffer, _ in
            self?.processAudioBuffer(buffer)
        }
        
        do {
            try engine.start()
            DispatchQueue.main.async {
                self.isActive = true
            }
            print("🎤 Audio engine started successfully")
            print("📊 Format: \(format)")
        } catch {
            print("❌ Failed to start audio engine: \(error)")
        }
    }
    
    func stopCapture() {
        guard let engine = audioEngine, let input = inputNode else { return }
        
        input.removeTap(onBus: 0)
        engine.stop()
        
        DispatchQueue.main.async {
            self.isActive = false
            self.bassLevel = 0
            self.midLevel = 0
            self.trebleLevel = 0
            self.overallAmplitude = 0
        }
    }
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { 
            print("⚠️ No channel data")
            return 
        }
        let frameLength = Int(buffer.frameLength)
        guard frameLength > 0 else { return }
        
        var audioData = [Float](repeating: 0, count: fftSize)
        for i in 0..<min(fftSize, frameLength) {
            audioData[i] = channelData[i]
        }
        
        guard windowFunction.count == fftSize else { 
            print("⚠️ Window function size mismatch")
            return 
        }
        
        var windowedData = [Float](repeating: 0, count: fftSize)
        vDSP_vmul(audioData, 1, windowFunction, 1, &windowedData, 1, vDSP_Length(fftSize))
        
        performFFT(on: windowedData)
        
        let sampleRate: Float = 44100
        extractFrequencyBands(sampleRate: sampleRate)
    }
    
    private func performFFT(on data: [Float]) {
        guard let setup = fftSetup else {
            print("⚠️ FFT setup not initialized")
            return
        }
        
        guard data.count == fftSize else {
            print("⚠️ Data size mismatch: \(data.count) vs \(fftSize)")
            return
        }
        
        guard realBuffer.count == fftSize / 2,
              imagBuffer.count == fftSize / 2,
              magnitudes.count == fftSize / 2 else {
            print("⚠️ Buffer size mismatch")
            return
        }
        
        var splitComplex = DSPSplitComplex(realp: &realBuffer, imagp: &imagBuffer)
        
        data.withUnsafeBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            baseAddress.withMemoryRebound(to: DSPComplex.self, capacity: data.count / 2) { complexPtr in
                vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(fftSize / 2))
            }
        }
        
        let log2n = vDSP_Length(log2(Float(fftSize)))
        vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
        
        vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(fftSize / 2))
        
        var normalizedMagnitudes = magnitudes
        vDSP_vsmul(magnitudes, 1, [1.0 / Float(fftSize)], &normalizedMagnitudes, 1, vDSP_Length(fftSize / 2))
        
        magnitudes = normalizedMagnitudes.map { $0.isFinite ? sqrt(max(0, $0)) : 0 }
    }
    
    private func extractFrequencyBands(sampleRate: Float) {
        let binCount = fftSize / 2
        guard binCount > 0 else { return }
        
        let freqPerBin = sampleRate / Float(fftSize)
        guard freqPerBin > 0 else { return }
        
        let bassBins = max(1, Int(250 / freqPerBin))
        let midBins = max(bassBins + 1, Int(4000 / freqPerBin))
        
        var bassSum: Float = 0
        var midSum: Float = 0
        var trebleSum: Float = 0
        
        guard magnitudes.count >= binCount else { return }
        
        for i in 0..<binCount {
            if i < bassBins {
                bassSum += magnitudes[i]
            } else if i < midBins {
                midSum += magnitudes[i]
            } else {
                trebleSum += magnitudes[i]
            }
        }
        
        let rawBass = bassSum / Float(max(1, bassBins))
        let rawMid = midSum / Float(max(1, midBins - bassBins))
        let rawTreble = trebleSum / Float(max(1, binCount - midBins))
        
        smoothedBass = smoothedBass * (1 - smoothingFactor) + rawBass * smoothingFactor
        smoothedMid = smoothedMid * (1 - smoothingFactor) + rawMid * smoothingFactor
        smoothedTreble = smoothedTreble * (1 - smoothingFactor) + rawTreble * smoothingFactor
        
        let overall = (smoothedBass + smoothedMid + smoothedTreble) / 3.0
        let boost: Float = 5.0
        
        let finalBass = min(1.0, smoothedBass * boost)
        let finalMid = min(1.0, smoothedMid * boost)
        let finalTreble = min(1.0, smoothedTreble * boost)
        let finalAmplitude = min(1.0, overall * boost)
        
        // Debug output every 30 frames (~0.5 seconds)
        if Int.random(in: 0...30) == 0 {
            print("🎵 Audio Levels - Bass: \(String(format: "%.2f", finalBass)), Mid: \(String(format: "%.2f", finalMid)), Treble: \(String(format: "%.2f", finalTreble))")
        }
        
        DispatchQueue.main.async {
            self.bassLevel = finalBass
            self.midLevel = finalMid
            self.trebleLevel = finalTreble
            self.overallAmplitude = finalAmplitude
        }
    }
    
    deinit {
        stopCapture()
        if let setup = fftSetup {
            vDSP_destroy_fftsetup(setup)
        }
    }
}

// MARK: - Helper Extensions

extension Int {
    var isPowerOfTwo: Bool {
        return self > 0 && (self & (self - 1)) == 0
    }
}
*/
