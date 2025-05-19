#!/usr/bin/env python
"""
Test script for the modified analyze_spectral_results.py using plot_utils.py
This script can be used to test the visualization pipeline without running the full training.
"""

import os
import sys
import argparse
import glob

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

def find_h5_files(directory):
    """Find all H5 files in a directory (recursively)"""
    h5_pattern = os.path.join(directory, "**", "*_analysis_data.h5")
    return glob.glob(h5_pattern, recursive=True)

def main(args):
    """Main function to test the spectral plotter"""
    # Find H5 files if not specifically provided
    if not args.h5_file:
        if not os.path.exists(args.results_dir):
            print(f"Results directory does not exist: {args.results_dir}")
            return
            
        h5_files = find_h5_files(args.results_dir)
        if not h5_files:
            print(f"No H5 files found in {args.results_dir}")
            return
            
        # Use the most recent H5 file
        h5_files.sort(key=os.path.getmtime, reverse=True)
        h5_file = h5_files[0]
        print(f"Using most recent H5 file: {h5_file}")
    else:
        h5_file = args.h5_file
        if not os.path.exists(h5_file):
            print(f"H5 file does not exist: {h5_file}")
            return
    
    # Make sure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")
    
    # Run the analyze_spectral_results.py script using subprocess
    import subprocess
    cmd = [
        "python", "analyze_spectral_results.py",
        "--h5_file", h5_file,
        "--output_dir", args.output_dir
    ]
    
    if args.selected_layers:
        cmd.extend(["--selected_layers"] + args.selected_layers)
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    # Check if the command was successful
    if result.returncode == 0:
        print(f"Success! Check the output directory: {args.output_dir}")
    else:
        print(f"Error! Command failed with return code {result.returncode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the spectral plotter using analyze_spectral_results.py")
    parser.add_argument('--results_dir', type=str, default='out-spectral/spectral',
                      help='Directory containing spectral analysis results')
    parser.add_argument('--h5_file', type=str, default=None,
                      help='Direct path to HDF5 file with spectral analysis results')
    parser.add_argument('--output_dir', type=str, default='test_figures',
                      help='Directory to save plots and analysis results')
    parser.add_argument('--selected_layers', type=str, nargs='+', default=None,
                      help='Only analyze these specific layers (optional)')
    args = parser.parse_args()
    
    main(args) 