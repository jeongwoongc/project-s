#!/bin/bash
# Convert PyTorch model weights to Core ML format for use in FlowViz

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/model"
DATA_DIR="$PROJECT_ROOT/data/weights"
PYTHON_ENV="${PYTHON_ENV:-python3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python environment has required packages
check_python_deps() {
    log_info "Checking Python dependencies..."
    
    required_packages=("torch" "coremltools" "numpy")
    missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! $PYTHON_ENV -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        log_error "Missing required Python packages: ${missing_packages[*]}"
        log_info "Install them with: pip install ${missing_packages[*]}"
        exit 1
    fi
    
    log_success "All required Python packages are installed"
}

# Convert a single model
convert_model() {
    local model_path="$1"
    local model_type="$2"
    local output_name="$3"
    
    if [ ! -f "$model_path" ]; then
        log_warning "Model file not found: $model_path"
        return 1
    fi
    
    log_info "Converting $model_type model: $(basename "$model_path")"
    
    local output_path="$DATA_DIR/${output_name}.mlmodel"
    
    # Run the conversion
    cd "$MODEL_DIR"
    $PYTHON_ENV export_coreml.py \
        --model_path "$model_path" \
        --output_path "$output_path" \
        --model_type "$model_type" \
        --validate
    
    if [ $? -eq 0 ] && [ -f "$output_path" ]; then
        local file_size=$(du -h "$output_path" | cut -f1)
        log_success "Converted to Core ML: $output_path ($file_size)"
        return 0
    else
        log_error "Failed to convert $model_type model"
        return 1
    fi
}

# Find and convert all available models
convert_all_models() {
    log_info "Searching for trained models in $PROJECT_ROOT..."
    
    # Common model file patterns
    local model_patterns=(
        "best_model.pt"
        "checkpoint_*.pt"
        "*_flow_matching_*.pt"
        "*_velocity_field_*.pt"
        "*_unet_*.pt"
    )
    
    local models_found=0
    local models_converted=0
    
    # Search for Flow Matching models
    for pattern in "best_model.pt" "*flow_matching*.pt"; do
        for model_file in $(find "$PROJECT_ROOT" -name "$pattern" -type f 2>/dev/null); do
            models_found=$((models_found + 1))
            if convert_model "$model_file" "flow_matching" "cfm_$(basename "$model_file" .pt)"; then
                models_converted=$((models_converted + 1))
            fi
        done
    done
    
    # Search for Velocity Field models
    for pattern in "*velocity_field*.pt"; do
        for model_file in $(find "$PROJECT_ROOT" -name "$pattern" -type f 2>/dev/null); do
            models_found=$((models_found + 1))
            if convert_model "$model_file" "velocity_field" "velocity_$(basename "$model_file" .pt)"; then
                models_converted=$((models_converted + 1))
            fi
        done
    done
    
    # Search for U-Net models
    for pattern in "*unet*.pt"; do
        for model_file in $(find "$PROJECT_ROOT" -name "$pattern" -type f 2>/dev/null); do
            models_found=$((models_found + 1))
            if convert_model "$model_file" "unet" "unet_$(basename "$model_file" .pt)"; then
                models_converted=$((models_converted + 1))
            fi
        done
    done
    
    if [ $models_found -eq 0 ]; then
        log_warning "No trained models found. Train models first using:"
        log_info "  cd $MODEL_DIR && python train_cfm_2d.py --config configs/default.json"
    else
        log_success "Converted $models_converted out of $models_found models found"
    fi
}

# Create a sample training configuration if it doesn't exist
create_sample_config() {
    local config_dir="$MODEL_DIR/configs"
    local config_file="$config_dir/default.json"
    
    if [ ! -f "$config_file" ]; then
        log_info "Creating sample training configuration..."
        
        mkdir -p "$config_dir"
        
        cat > "$config_file" << 'EOF'
{
    "model_type": "flow_matching",
    "input_dim": 2,
    "hidden_dim": 256,
    "num_layers": 6,
    "time_embedding_dim": 128,
    "num_heads": 8,
    "dropout": 0.1,
    
    "train_samples": 5000,
    "val_samples": 1000,
    "batch_size": 32,
    "num_workers": 4,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    
    "grid_size": 64,
    "num_obstacles_range": [0, 5],
    
    "velocity_weight": 1.0,
    "consistency_weight": 0.1,
    "boundary_weight": 0.5,
    "obstacle_weight": 1.0,
    "divergence_weight": 0.1,
    
    "output_dir": "../data/checkpoints",
    "save_every": 20,
    "vis_every": 20
}
EOF
        
        log_success "Created sample config: $config_file"
    fi
}

# Train a sample model if no models exist
train_sample_model() {
    log_info "No models found. Would you like to train a sample model? (y/N)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "Training sample Flow Matching model..."
        
        cd "$MODEL_DIR"
        create_sample_config
        
        # Train with reduced epochs for quick testing
        $PYTHON_ENV train_cfm_2d.py \
            --config configs/default.json \
            --wandb
        
        if [ $? -eq 0 ]; then
            log_success "Sample model training completed"
            # Try converting again
            convert_all_models
        else
            log_error "Sample model training failed"
        fi
    fi
}

# Clean up old Core ML models
clean_old_models() {
    if [ "$1" = "--clean" ]; then
        log_info "Cleaning old Core ML models..."
        rm -f "$DATA_DIR"/*.mlmodel
        log_success "Cleaned old models"
    fi
}

# Main function
main() {
    log_info "FlowViz Model Weight Conversion Script"
    log_info "======================================"
    
    # Handle command line arguments
    case "${1:-}" in
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Convert PyTorch model weights to Core ML format for FlowViz"
            echo ""
            echo "Options:"
            echo "  --clean         Remove old Core ML models before conversion"
            echo "  --train         Train sample models if none exist"
            echo "  --check         Only check dependencies"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  PYTHON_ENV      Python executable to use (default: python3)"
            exit 0
            ;;
        --check)
            check_python_deps
            exit 0
            ;;
        --clean)
            clean_old_models --clean
            ;;
    esac
    
    # Create output directory
    mkdir -p "$DATA_DIR"
    
    # Check dependencies
    check_python_deps
    
    # Clean if requested
    clean_old_models "$1"
    
    # Convert existing models
    convert_all_models
    
    # Offer to train if no models found and --train flag is used
    if [ "$1" = "--train" ] && [ $(find "$DATA_DIR" -name "*.mlmodel" | wc -l) -eq 0 ]; then
        train_sample_model
    fi
    
    # Summary
    local mlmodel_count=$(find "$DATA_DIR" -name "*.mlmodel" | wc -l)
    
    if [ $mlmodel_count -gt 0 ]; then
        log_success "Conversion complete! Found $mlmodel_count Core ML models:"
        find "$DATA_DIR" -name "*.mlmodel" -exec basename {} \; | sort
    else
        log_warning "No Core ML models were created."
        log_info "To train models, run:"
        log_info "  cd $MODEL_DIR && python train_cfm_2d.py --config configs/default.json"
        log_info "Then run this script again to convert them."
    fi
}

# Run main function with all arguments
main "$@"
