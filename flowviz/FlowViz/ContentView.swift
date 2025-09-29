import SwiftUI
import MetalKit

struct ContentView: View {
    @StateObject private var viewModel = FlowVizViewModel()
    @State private var showControls = true
    
    var body: some View {
        ZStack {
            // Main Metal rendering view
            MetalView(viewModel: viewModel)
                .ignoresSafeArea(.all)
            
            // Overlay UI
            VStack {
                // Top HUD
                HStack {
                    HUD(viewModel: viewModel)
                    Spacer()
                    Button(action: { showControls.toggle() }) {
                        Image(systemName: showControls ? "sidebar.right" : "sidebar.left")
                            .font(.title2)
                            .foregroundColor(.white)
                            .background(Circle().fill(.ultraThinMaterial))
                            .padding(8)
                    }
                }
                .padding()
                
                Spacer()
            }
            
            // Side controls panel
            if showControls {
                HStack {
                    Spacer()
                    ControlsPanel(viewModel: viewModel)
                        .frame(width: 300)
                        .background(.ultraThinMaterial)
                        .cornerRadius(12)
                        .padding()
                }
            }
        }
        .onAppear {
            viewModel.setupMetal()
        }
    }
}

struct MetalView: NSViewRepresentable {
    let viewModel: FlowVizViewModel
    
    func makeNSView(context: Context) -> MTKView {
        let metalView = MTKView()
        metalView.device = MTLCreateSystemDefaultDevice()
        metalView.delegate = viewModel.renderer
        metalView.preferredFramesPerSecond = 60
        metalView.isPaused = false
        metalView.enableSetNeedsDisplay = false
        metalView.clearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1.0)
        
        viewModel.setupRenderer(metalView: metalView)
        
        return metalView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        // Update view if needed
    }
}

#Preview {
    ContentView()
}
