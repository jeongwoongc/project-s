#!/bin/bash
# Quick UI testing script for FlowViz

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🌌 FlowViz UI Testing Script${NC}"
echo "================================="
echo ""

# Check if we're in the right directory
if [ ! -f "FlowViz.xcodeproj/project.pbxproj" ]; then
    echo -e "${YELLOW}⚠️  Please run this script from the flowviz directory${NC}"
    exit 1
fi

# Check if Xcode is available
if ! command -v xcodebuild &> /dev/null; then
    echo -e "${YELLOW}⚠️  Xcode command line tools not found${NC}"
    echo "Install with: xcode-select --install"
    exit 1
fi

echo -e "${GREEN}✅ Found Xcode project${NC}"

# Check project structure
echo -e "${BLUE}📁 Checking project structure...${NC}"
required_files=(
    "FlowViz/FlowVizApp.swift"
    "FlowViz/ContentView.swift"
    "FlowViz/ViewModel.swift"
    "FlowViz/TestData.swift"
    "FlowViz/Renderer/MetalRenderer.swift"
    "FlowViz/Renderer/Shaders.metal"
    "FlowViz/UX/ControlsPanel.swift"
    "FlowViz/UX/HUD.swift"
    "FlowViz/Core/VelocityGrid.swift"
    "FlowViz/Data/scenes.json"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Missing files:${NC}"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "Run the setup script first to copy all source files."
    exit 1
fi

echo -e "${GREEN}✅ All required files found${NC}"

# Try to build the project
echo -e "${BLUE}🔨 Testing build...${NC}"
if xcodebuild -project FlowViz.xcodeproj -scheme FlowViz -configuration Debug build -quiet; then
    echo -e "${GREEN}✅ Build successful!${NC}"
else
    echo -e "${YELLOW}⚠️  Build failed. Opening Xcode to investigate...${NC}"
fi

echo ""
echo -e "${BLUE}🚀 Opening FlowViz in Xcode...${NC}"
open FlowViz.xcodeproj

echo ""
echo -e "${GREEN}📖 Testing Instructions:${NC}"
echo "========================"
echo ""
echo "1. 🎯 BASIC TESTING:"
echo "   • Press ⌘R to build and run"
echo "   • You should see a dark interface with controls"
echo "   • Green dot (start) and red dot (goal) should be visible"
echo ""
echo "2. 🎮 INTERACTION TESTING:"
echo "   • Click anywhere to add obstacles (red circles)"
echo "   • Drag the green/red dots to move start/goal"
echo "   • Use the controls panel on the right"
echo "   • Try different scene presets"
echo ""
echo "3. 🎨 VISUAL TESTING:"
echo "   • Particles should appear as flowing blue/pink dots"
echo "   • Smooth 60 FPS animation (check HUD)"
echo "   • Controls should be responsive"
echo ""
echo "4. 🧪 SWIFTUI PREVIEWS:"
echo "   • Open ContentView.swift"
echo "   • Press ⌥⌘↩ to show Canvas"
echo "   • Test individual components"
echo ""
echo "5. 🐛 IF SOMETHING DOESN'T WORK:"
echo "   • Check Console.app for errors"
echo "   • Try different macOS versions (needs 13.0+)"
echo "   • Test on different Macs if available"
echo ""
echo -e "${YELLOW}💡 TIP: The UI works without ML models!${NC}"
echo "   The app uses procedural flow fields for testing."
echo ""
echo -e "${BLUE}📚 For detailed testing info, see: TESTING.md${NC}"
echo ""
echo -e "${GREEN}🎉 Happy testing!${NC}"
