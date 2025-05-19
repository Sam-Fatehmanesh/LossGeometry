#!/bin/bash
# Run the complete pipeline for spectral analysis of nanoGPT on text data

# Define directories
OUT_DIR="out-spectral"
SPECTRAL_DIR="$OUT_DIR/spectral"
FIGURE_DIR="$OUT_DIR/figures"
DATA_DIR="data/pile"

# Ensure output directories exist
mkdir -p $OUT_DIR
mkdir -p $SPECTRAL_DIR
mkdir -p $FIGURE_DIR
mkdir -p $DATA_DIR

# Step 1: Prepare the text data
echo "=== Step 1: Preparing the text data ==="
python prepare_pile.py --data_dir $DATA_DIR

# Step 2: Train nanoGPT with spectral analysis
echo "=== Step 2: Training nanoGPT with spectral analysis ==="
python train_spectral.py config/train_pile_spectral.py

# Step 3: Analyze and visualize spectral results
echo "=== Step 3: Analyzing and visualizing spectral results ==="
python analyze_spectral_results.py --results_dir $SPECTRAL_DIR --output_dir $FIGURE_DIR

echo "=== Complete! ==="
echo "Results saved to $OUT_DIR"
echo "Figures saved to $FIGURE_DIR" 