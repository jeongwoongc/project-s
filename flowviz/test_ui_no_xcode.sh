#!/bin/bash
# FlowViz UI testing without Xcode IDE

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🌌 FlowViz UI Testing (No Xcode IDE)${NC}"
echo "===================================="
echo ""

echo -e "${YELLOW}📋 Since you don't have Xcode IDE, here are alternative testing methods:${NC}"
echo ""

# Check what we have available
echo -e "${BLUE}🔍 Checking available tools...${NC}"

# Check for Swift
if command -v swift &> /dev/null; then
    SWIFT_VERSION=$(swift --version | head -n1)
    echo -e "${GREEN}✅ Swift: $SWIFT_VERSION${NC}"
    HAS_SWIFT=true
else
    echo -e "${RED}❌ Swift not found${NC}"
    HAS_SWIFT=false
fi

# Check for xcodebuild (command line)
if command -v xcodebuild &> /dev/null; then
    echo -e "${GREEN}✅ xcodebuild available${NC}"
    HAS_XCODEBUILD=true
else
    echo -e "${RED}❌ xcodebuild not available${NC}"
    HAS_XCODEBUILD=false
fi

# Check for Python (for web preview)
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ Python: $PYTHON_VERSION${NC}"
    HAS_PYTHON=true
else
    echo -e "${RED}❌ Python not found${NC}"
    HAS_PYTHON=false
fi

echo ""

# Option 1: Install Xcode (recommended)
echo -e "${BLUE}🎯 OPTION 1: Install Xcode (Recommended)${NC}"
echo "========================================="
echo "• Download from Mac App Store (free, ~15GB)"
echo "• Or download from Apple Developer Portal"
echo "• Then run: ./test_ui.sh"
echo ""

# Option 2: Use VS Code with Swift extension
echo -e "${BLUE}🎯 OPTION 2: VS Code + Swift Extension${NC}"
echo "====================================="
if command -v code &> /dev/null; then
    echo -e "${GREEN}✅ VS Code found${NC}"
    echo "• Install Swift extension: code --install-extension sswg.swift-lang"
    echo "• Open project: code ."
    echo "• View Swift files with syntax highlighting"
    echo "• Use integrated terminal for Swift commands"
else
    echo "• Install VS Code: https://code.visualstudio.com"
    echo "• Install Swift extension"
    echo "• Open project folder"
fi
echo ""

# Option 3: Command line compilation (if Swift available)
if [ "$HAS_SWIFT" = true ]; then
    echo -e "${BLUE}🎯 OPTION 3: Command Line Swift Testing${NC}"
    echo "======================================"
    echo "You can test individual Swift components:"
    echo ""
    echo "• Test data structures:"
    echo "  swift -I FlowViz FlowViz/TestData.swift"
    echo ""
    echo "• Compile core math:"
    echo "  swift -c FlowViz/Core/*.swift"
    echo ""
    echo "• Check syntax:"
    echo "  swift -frontend -parse FlowViz/ContentView.swift"
    echo ""
fi

# Option 4: Web-based preview (if Python available)
if [ "$HAS_PYTHON" = true ]; then
    echo -e "${BLUE}🎯 OPTION 4: Web Preview of Code${NC}"
    echo "==============================="
    echo "Create a web-based code browser:"
    echo ""
    echo "• Run: python3 -m http.server 8000"
    echo "• Open: http://localhost:8000"
    echo "• Browse the code structure"
    echo "• View Swift files in browser"
    echo ""
fi

# Option 5: Code analysis
echo -e "${BLUE}🎯 OPTION 5: Static Code Analysis${NC}"
echo "================================="
echo "Analyze the code without running:"
echo ""
echo "• Check project structure:"
echo "  find FlowViz -name '*.swift' | head -10"
echo ""
echo "• Count lines of code:"
echo "  find FlowViz -name '*.swift' -exec wc -l {} + | tail -1"
echo ""
echo "• Search for specific patterns:"
echo "  grep -r 'ObservableObject' FlowViz/"
echo "  grep -r '@Published' FlowViz/"
echo "  grep -r 'Metal' FlowViz/"
echo ""

# Option 6: SwiftUI previews alternatives
echo -e "${BLUE}🎯 OPTION 6: Alternative UI Testing${NC}"
echo "=================================="
echo "Without Xcode, you can still:"
echo ""
echo "• Read the UI code to understand structure"
echo "• Modify component properties"
echo "• Test logic in isolated Swift files"
echo "• Use online Swift playgrounds"
echo ""

# Provide immediate testing options
echo -e "${YELLOW}🚀 IMMEDIATE TESTING OPTIONS:${NC}"
echo "============================="
echo ""

if [ "$HAS_SWIFT" = true ]; then
    echo -e "${GREEN}1. Test Swift compilation:${NC}"
    echo "   cd FlowViz && swift -frontend -parse *.swift"
    echo ""
fi

echo -e "${GREEN}2. Explore code structure:${NC}"
echo "   find FlowViz -name '*.swift' -exec basename {} \\; | sort"
echo ""

echo -e "${GREEN}3. Check key components:${NC}"
echo "   cat FlowViz/ContentView.swift | grep -A5 -B5 'struct ContentView'"
echo ""

if [ "$HAS_PYTHON" = true ]; then
    echo -e "${GREEN}4. Start web server to browse code:${NC}"
    echo "   python3 -m http.server 8000 &"
    echo "   open http://localhost:8000"
    echo ""
fi

echo -e "${BLUE}📚 Understanding the UI Without Running It:${NC}"
echo "=========================================="
echo ""
echo "Key files to examine:"
echo "• FlowViz/ContentView.swift - Main UI layout"
echo "• FlowViz/ViewModel.swift - App logic and state"
echo "• FlowViz/UX/ControlsPanel.swift - Interactive controls"
echo "• FlowViz/UX/HUD.swift - Heads-up display"
echo "• FlowViz/Renderer/MetalRenderer.swift - GPU rendering"
echo ""

echo -e "${YELLOW}💡 RECOMMENDATION:${NC}"
echo "=================="
echo "For the best FlowViz experience, install Xcode from the Mac App Store."
echo "It's free and provides the full SwiftUI development environment."
echo ""
echo "Alternative: Try online Swift playgrounds at:"
echo "• https://swiftfiddle.com"
echo "• https://online.swiftplayground.run"
echo ""

# Ask what they want to do
echo -e "${BLUE}❓ What would you like to do?${NC}"
echo "1. Install Xcode (recommended)"
echo "2. Browse code with web server"
echo "3. Test Swift compilation"
echo "4. Just explore the code files"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo -e "${GREEN}Opening Mac App Store to install Xcode...${NC}"
        open "macappstore://apps.apple.com/app/xcode/id497799835"
        ;;
    2)
        if [ "$HAS_PYTHON" = true ]; then
            echo -e "${GREEN}Starting web server...${NC}"
            python3 -m http.server 8000 &
            sleep 2
            open "http://localhost:8000"
            echo "Web server running at http://localhost:8000"
            echo "Press Ctrl+C to stop the server"
        else
            echo -e "${RED}Python not available${NC}"
        fi
        ;;
    3)
        if [ "$HAS_SWIFT" = true ]; then
            echo -e "${GREEN}Testing Swift compilation...${NC}"
            cd FlowViz
            for file in *.swift; do
                echo "Checking $file..."
                swift -frontend -parse "$file" && echo "✅ $file syntax OK" || echo "❌ $file has issues"
            done
        else
            echo -e "${RED}Swift not available${NC}"
        fi
        ;;
    4)
        echo -e "${GREEN}Key Swift files in the project:${NC}"
        find FlowViz -name "*.swift" | while read file; do
            lines=$(wc -l < "$file")
            echo "📄 $file ($lines lines)"
        done
        ;;
    *)
        echo -e "${YELLOW}No problem! You can always install Xcode later.${NC}"
        ;;
esac

echo ""
echo -e "${GREEN}🎉 Thanks for exploring FlowViz!${NC}"
