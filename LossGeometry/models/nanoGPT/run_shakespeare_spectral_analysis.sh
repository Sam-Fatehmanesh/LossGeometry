#!/bin/bash
# Run the complete pipeline for spectral analysis of nanoGPT on text data
# Now using the AnalysisPlotter from plot_utils.py for visualization

# Default parameters - can be overridden with command line arguments
NUM_RUNS=3
OUT_DIR="out-spectral"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --num_runs|--num_runs=*)
      if [[ $key == *=* ]]; then
        NUM_RUNS="${key#*=}"
        shift
      else
        NUM_RUNS="$2"
        shift 2
      fi
      ;;
    --out_dir|--out_dir=*)
      if [[ $key == *=* ]]; then
        OUT_DIR="${key#*=}"
        shift
      else
        OUT_DIR="$2"
        shift 2
      fi
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Define directories
SPECTRAL_DIR="$OUT_DIR/spectral"
FIGURE_DIR="$OUT_DIR/figures"
DATA_DIR="data/pile"

# Ensure output directories exist
mkdir -p $OUT_DIR
mkdir -p $SPECTRAL_DIR
mkdir -p $FIGURE_DIR
mkdir -p $DATA_DIR

echo "=== Using output directory: $OUT_DIR ==="

# Step 1: Prepare the text data
echo "=== Step 1: Preparing the text data ==="
python prepare_pile.py --data_dir $DATA_DIR

# Step 2: Train nanoGPT with spectral analysis, using multiple runs
echo "=== Step 2: Training nanoGPT with spectral analysis ($NUM_RUNS runs) ==="
python train_spectral.py config/train_pile_spectral.py --num_runs=$NUM_RUNS --out_dir=$OUT_DIR

# Step 3: Analyze and visualize spectral results using plot_utils.py's AnalysisPlotter
echo "=== Step 3: Analyzing and visualizing spectral results using plot_utils.py ==="
echo "This now uses the AnalysisPlotter class for more consistent visualizations"
# Use --top_dir to directly point to the out-spectral directory where HDF5 files are stored
python analyze_spectral_results.py --top_dir $OUT_DIR --output_dir $FIGURE_DIR

echo "=== Complete! ==="
echo "Results saved to $OUT_DIR"
echo "Figures saved to $FIGURE_DIR"
echo "Analysis performed across $NUM_RUNS runs for more robust results"
echo "All visualizations generated using the AnalysisPlotter from plot_utils.py for consistent formatting" 