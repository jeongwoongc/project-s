import Foundation
import CoreML

/// Handles Core ML model loading and inference for neural flow fields
class ModelIO {
    private(set) var loadedModel: MLModel?
    private var modelURL: URL?
    
    init() {}
    
    /// Load Core ML model from file
    func loadModel(from url: URL) throws {
        do {
            let model = try MLModel(contentsOf: url)
            self.loadedModel = model
            self.modelURL = url
            print("Successfully loaded model from: \(url.lastPathComponent)")
        } catch {
            print("Failed to load model: \(error)")
            throw ModelIOError.failedToLoadModel(error)
        }
    }
    
    /// Load model from bundle
    func loadModelFromBundle(named modelName: String) throws {
        guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodel") else {
            throw ModelIOError.modelNotFound(modelName)
        }
        
        try loadModel(from: url)
    }
    
    /// Predict velocity field using loaded model
    func predictVelocityField(
        positions: [simd_float2],
        obstacles: [ObstacleFeature] = [],
        context: FlowContext = FlowContext()
    ) throws -> [simd_float2] {
        guard let model = loadedModel else {
            throw ModelIOError.noModelLoaded
        }
        
        // Prepare input features
        let inputFeatures = try prepareInputFeatures(
            positions: positions,
            obstacles: obstacles,
            context: context
        )
        
        // Run inference
        let prediction = try model.prediction(from: inputFeatures)
        
        // Extract velocity vectors from prediction
        return try extractVelocityVectors(from: prediction)
    }
    
    /// Predict single velocity vector at position
    func predictVelocity(
        at position: simd_float2,
        obstacles: [ObstacleFeature] = [],
        context: FlowContext = FlowContext()
    ) throws -> simd_float2 {
        let velocities = try predictVelocityField(
            positions: [position],
            obstacles: obstacles,
            context: context
        )
        
        guard let velocity = velocities.first else {
            throw ModelIOError.predictionFailed("No velocity returned")
        }
        
        return velocity
    }
    
    /// Export trained PyTorch model to Core ML format
    static func exportPyTorchToCoreML(
        pythonScript: String,
        modelPath: String,
        outputPath: String
    ) throws {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        task.arguments = [pythonScript, modelPath, outputPath]
        
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        
        try task.run()
        task.waitUntilExit()
        
        if task.terminationStatus != 0 {
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw ModelIOError.exportFailed(output)
        }
    }
    
    // MARK: - Private Methods
    
    private func prepareInputFeatures(
        positions: [simd_float2],
        obstacles: [ObstacleFeature],
        context: FlowContext
    ) throws -> MLFeatureProvider {
        // Create feature dictionary
        var features: [String: MLFeatureValue] = [:]
        
        // Position features (Nx2 array)
        let positionArray = positions.flatMap { [$0.x, $0.y] }
        let positionMLArray = try MLMultiArray(shape: [NSNumber(value: positions.count), 2], dataType: .float32)
        for (i, value) in positionArray.enumerated() {
            positionMLArray[i] = NSNumber(value: value)
        }
        features["positions"] = MLFeatureValue(multiArray: positionMLArray)
        
        // Obstacle features (Mx4 array: x, y, radius, type)
        if !obstacles.isEmpty {
            let obstacleArray = obstacles.flatMap { [$0.center.x, $0.center.y, $0.radius, Float($0.type.rawValue)] }
            let obstacleMLArray = try MLMultiArray(shape: [NSNumber(value: obstacles.count), 4], dataType: .float32)
            for (i, value) in obstacleArray.enumerated() {
                obstacleMLArray[i] = NSNumber(value: value)
            }
            features["obstacles"] = MLFeatureValue(multiArray: obstacleMLArray)
        } else {
            // Empty obstacle array
            let emptyObstacles = try MLMultiArray(shape: [1, 4], dataType: .float32)
            features["obstacles"] = MLFeatureValue(multiArray: emptyObstacles)
        }
        
        // Context features
        let contextArray = try MLMultiArray(shape: [6], dataType: .float32)
        contextArray[0] = NSNumber(value: context.startPoint.x)
        contextArray[1] = NSNumber(value: context.startPoint.y)
        contextArray[2] = NSNumber(value: context.goalPoint.x)
        contextArray[3] = NSNumber(value: context.goalPoint.y)
        contextArray[4] = NSNumber(value: context.time)
        contextArray[5] = NSNumber(value: context.flowStrength)
        features["context"] = MLFeatureValue(multiArray: contextArray)
        
        return try MLDictionaryFeatureProvider(dictionary: features)
    }
    
    private func extractVelocityVectors(from prediction: MLFeatureProvider) throws -> [simd_float2] {
        guard let velocityFeature = prediction.featureValue(for: "velocity_field"),
              let velocityArray = velocityFeature.multiArrayValue else {
            throw ModelIOError.predictionFailed("No velocity_field output found")
        }
        
        let count = velocityArray.shape[0].intValue
        var velocities: [simd_float2] = []
        
        for i in 0..<count {
            let vx = velocityArray[i * 2].floatValue
            let vy = velocityArray[i * 2 + 1].floatValue
            velocities.append(simd_float2(vx, vy))
        }
        
        return velocities
    }
}

// MARK: - Supporting Types

struct ObstacleFeature {
    let center: simd_float2
    let radius: Float
    let type: ObstacleType
    
    enum ObstacleType: Int {
        case circle = 0
        case rectangle = 1
    }
}

struct FlowContext {
    let startPoint: simd_float2
    let goalPoint: simd_float2
    let time: Float
    let flowStrength: Float
    
    init(
        startPoint: simd_float2 = simd_float2(0.1, 0.5),
        goalPoint: simd_float2 = simd_float2(0.9, 0.5),
        time: Float = 0.0,
        flowStrength: Float = 1.0
    ) {
        self.startPoint = startPoint
        self.goalPoint = goalPoint
        self.time = time
        self.flowStrength = flowStrength
    }
}

enum ModelIOError: Error, LocalizedError {
    case modelNotFound(String)
    case failedToLoadModel(Error)
    case noModelLoaded
    case predictionFailed(String)
    case exportFailed(String)
    case invalidInputFormat
    
    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Model '\(name)' not found in bundle"
        case .failedToLoadModel(let error):
            return "Failed to load model: \(error.localizedDescription)"
        case .noModelLoaded:
            return "No model is currently loaded"
        case .predictionFailed(let message):
            return "Prediction failed: \(message)"
        case .exportFailed(let output):
            return "Export failed: \(output)"
        case .invalidInputFormat:
            return "Invalid input format for model"
        }
    }
}

// MARK: - Model Utilities

extension ModelIO {
    
    /// Get model information
    func getModelInfo() -> ModelInfo? {
        guard let model = loadedModel else { return nil }
        
        let inputDescriptions = model.modelDescription.inputDescriptionsByName
        let outputDescriptions = model.modelDescription.outputDescriptionsByName
        
        return ModelInfo(
            inputFeatures: Array(inputDescriptions.keys),
            outputFeatures: Array(outputDescriptions.keys),
            modelURL: modelURL
        )
    }
    
    /// Validate model compatibility
    func validateModelCompatibility() -> Bool {
        guard let model = loadedModel else { return false }
        
        let requiredInputs = ["positions", "obstacles", "context"]
        let requiredOutputs = ["velocity_field"]
        
        let inputNames = Set(model.modelDescription.inputDescriptionsByName.keys)
        let outputNames = Set(model.modelDescription.outputDescriptionsByName.keys)
        
        let hasRequiredInputs = requiredInputs.allSatisfy { inputNames.contains($0) }
        let hasRequiredOutputs = requiredOutputs.allSatisfy { outputNames.contains($0) }
        
        return hasRequiredInputs && hasRequiredOutputs
    }
}

struct ModelInfo {
    let inputFeatures: [String]
    let outputFeatures: [String]
    let modelURL: URL?
}
