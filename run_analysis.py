#!/usr/bin/env python3
"""
Run script for LossGeometry analysis.
This script runs the LossGeometry analysis with the specified parameters.
"""

import os
import sys
import argparse
from LossGeometry.main import train_and_analyze

def main():
    """Main entry point for running LossGeometry analysis"""
    parser = argparse.ArgumentParser(description='Loss Geometry Analysis')
    
    # Model parameters
    parser.add_argument('--input_size', type=int, default=784, help='Input size')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden layer size')
    parser.add_argument('--output_size', type=int, default=10, help='Output size')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers with square dimensions')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--log_every_n_batches', type=int, default=200, help='Frequency of analysis calculation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_runs', '-n', type=int, default=1, help='Number of training runs to average results over')
    
    # Analysis parameters
    parser.add_argument('--analyze_W', action='store_true', help='Analyze weight matrices (default)')
    parser.add_argument('--analyze_delta_W', action='store_true', help='Analyze weight update matrices')
    parser.add_argument('--analyze_spectral_density', action='store_true', help='Analyze spectral density (default)')
    parser.add_argument('--analyze_level_spacing', action='store_true', help='Analyze level spacing')
    parser.add_argument('--analyze_singular_values', action='store_true', help='Analyze singular values')
    
    # Output parameters
    parser.add_argument('--experiment_dir', type=str, default='experiments', help='Base directory for experiments')
    
    args = parser.parse_args()
    
    # Set defaults for analysis types if none specified
    if not args.analyze_W and not args.analyze_delta_W:
        args.analyze_W = True
    if not args.analyze_spectral_density and not args.analyze_level_spacing and not args.analyze_singular_values:
        args.analyze_spectral_density = True
        args.analyze_singular_values = True
    
    # Run the analysis
    print("Starting LossGeometry analysis...")
    if args.num_runs > 1:
        print(f"Running {args.num_runs} training runs and averaging results.")
    output_dir, h5_path = train_and_analyze(args)
    print(f"Analysis complete. Results saved to: {output_dir}")
    print(f"Data saved to: {h5_path}")
    
    # Print some example usage help
    if args.num_runs == 1:
        print("\nTip: To run multiple training runs and average results, use:")
        print("    ./run_analysis.py -n 5")
        print("    ./run_analysis.py --num_runs 5")

if __name__ == "__main__":
    sys.exit(main()) 