#!/usr/bin/env python3
"""
Aggregate spectral analysis results from multiple runs.

This script finds all HDF5 files in the output directory, loads them,
aggregates the data, and saves the result to a new HDF5 file.

Example:
    $ python aggregate_all_spectral_runs.py --input_dir out-spectral --output_dir out-spectral
"""

import os
import sys
import argparse
import numpy as np
import glob
from datetime import datetime
import h5py
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from LossGeometry.utils.io_utils import load_analysis_data, save_analysis_data
from LossGeometry.analysis.spectral_analysis import SpectralAnalyzer

def find_all_h5_files(input_dir):
    """Find all HDF5 files in the input directory (recursively)
    
    Args:
        input_dir: Directory to search for HDF5 files
        
    Returns:
        List of paths to HDF5 files
    """
    h5_pattern = os.path.join(input_dir, "**", "*_analysis_data.h5")
    h5_files = glob.glob(h5_pattern, recursive=True)
    
    # Skip files with "aggregated" in the name
    h5_files = [f for f in h5_files if "aggregated" not in f]
    
    # Sort by run number if possible
    def extract_run_number(filename):
        try:
            # Extract run number from filenames like "20250520_172737_run227_final_analysis_data.h5"
            basename = os.path.basename(filename)
            parts = basename.split('_')
            for i, part in enumerate(parts):
                if part.startswith('run') and i+1 < len(parts):
                    return int(part[3:])  # Extract the number after "run"
        except (ValueError, IndexError):
            pass
        return 0  # Default if no run number found
    
    h5_files.sort(key=extract_run_number)
    return h5_files

def aggregate_spectral_data(h5_files):
    """Aggregate data from multiple HDF5 files
    
    Args:
        h5_files: List of paths to HDF5 files
        
    Returns:
        SpectralAnalyzer object with aggregated data
    """
    print(f"Aggregating data from {len(h5_files)} files...")
    
    # Create a new analyzer for aggregated results
    aggregated_analyzer = SpectralAnalyzer(
        analyze_W=True,
        analyze_delta_W=True,
        analyze_spectral_density=True,
        analyze_level_spacing=False,
        analyze_singular_values=True
    )
    
    # Load the first file to get the target layers
    if not h5_files:
        raise ValueError("No HDF5 files found")
    
    first_data = load_analysis_data(h5_files[0])
    target_layers = list(first_data['results'].keys())
    aggregated_analyzer.initialize_layer_stats(target_layers)
    
    # Copy batch info from first run
    aggregated_analyzer.stats['batch'] = first_data.get('batch', [])
    aggregated_analyzer.stats['batch_numbers'] = first_data.get('batch_numbers', [])
    
    # Set description
    if 'matrix_type' in first_data:
        aggregated_analyzer.stats['matrix_type'] = first_data['matrix_type']
    if 'analysis_type' in first_data:
        aggregated_analyzer.stats['analysis_type'] = first_data['analysis_type']
    
    # Create a mapping for each layer with all eigenvalues and singular values per batch
    all_eigenvalues = {layer: {} for layer in target_layers}
    all_singular_values = {layer: {} for layer in target_layers}
    all_losses = []
    
    # Load and process each file
    for h5_file in tqdm(h5_files, desc="Loading files"):
        try:
            data = load_analysis_data(h5_file)
            
            # Collect loss values if they exist
            if 'loss_values' in data and data['loss_values']:
                all_losses.append(data['loss_values'])
            
            # Process each layer
            for layer_name in target_layers:
                if layer_name not in data['results']:
                    continue
                
                layer_data = data['results'][layer_name]
                
                # Collect eigenvalues
                if 'eigenvalues_list' in layer_data and layer_data['eigenvalues_list']:
                    for i, eigs in enumerate(layer_data['eigenvalues_list']):
                        if eigs is not None and len(eigs) > 0:
                            if i not in all_eigenvalues[layer_name]:
                                all_eigenvalues[layer_name][i] = []
                            all_eigenvalues[layer_name][i].append(eigs)
                
                # Collect singular values
                if 'singular_values_list' in layer_data and layer_data['singular_values_list']:
                    for i, svs in enumerate(layer_data['singular_values_list']):
                        if svs is not None and len(svs) > 0:
                            if i not in all_singular_values[layer_name]:
                                all_singular_values[layer_name][i] = []
                            all_singular_values[layer_name][i].append(svs)
        except Exception as e:
            print(f"Error processing file {h5_file}: {e}")
    
    # Average loss values across runs
    if all_losses:
        # Find the minimum length
        min_length = min(len(losses) for losses in all_losses)
        # Truncate all loss arrays to the same length
        truncated_losses = [losses[:min_length] for losses in all_losses]
        # Compute average
        avg_losses = np.mean(truncated_losses, axis=0)
        aggregated_analyzer.stats['loss_values'] = avg_losses.tolist()
    
    # Aggregate eigenvalues for each layer
    for layer_name in target_layers:
        # Aggregate eigenvalues
        eigenvalues_list = []
        for batch in sorted(all_eigenvalues[layer_name].keys()):
            batch_eigenvalues = all_eigenvalues[layer_name][batch]
            if batch_eigenvalues:
                # Concatenate eigenvalues from all runs at this batch point
                all_eigenvalues_batch = np.concatenate(batch_eigenvalues)
                eigenvalues_list.append(all_eigenvalues_batch)
        
        if eigenvalues_list:
            aggregated_analyzer.stats['results'][layer_name]['eigenvalues_list'] = eigenvalues_list
            aggregated_analyzer.stats['results'][layer_name]['last_eigenvalues'] = eigenvalues_list[-1] if eigenvalues_list else None
        
        # Aggregate singular values
        sv_list = []
        for batch in sorted(all_singular_values[layer_name].keys()):
            batch_sv = all_singular_values[layer_name][batch]
            if batch_sv:
                # Concatenate singular values from all runs at this batch point
                all_sv_batch = np.concatenate(batch_sv)
                sv_list.append(all_sv_batch)
        
        if sv_list:
            aggregated_analyzer.stats['results'][layer_name]['singular_values_list'] = sv_list
            aggregated_analyzer.stats['results'][layer_name]['last_singular_values'] = sv_list[-1] if sv_list else None
    
    return aggregated_analyzer

def save_aggregated_data(aggregated_analyzer, output_dir, num_runs):
    """Save aggregated data to a new HDF5 file
    
    Args:
        aggregated_analyzer: SpectralAnalyzer object with aggregated data
        output_dir: Directory to save the aggregated data
        num_runs: Number of runs that were aggregated
        
    Returns:
        Path to the saved HDF5 file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregated_dir = os.path.join(output_dir, f"{timestamp}_aggregated_{num_runs}_runs")
    os.makedirs(aggregated_dir, exist_ok=True)
    
    # Create a dummy model object with required attributes
    class DummyModel:
        def __init__(self):
            self.input_size = 512
            self.hidden_size = 512
            self.output_size = 512
            self.num_hidden_layers = 6
        
        def get_parameter(self, layer_name):
            # Return a dummy parameter
            return torch.zeros((512, 512))
    
    dummy_model = DummyModel()
    
    # Save the aggregated data
    h5_path = save_analysis_data(aggregated_analyzer, dummy_model, output_dir, f"{timestamp}_aggregated_{num_runs}_runs")
    
    return h5_path

def run_analysis(h5_path, output_dir):
    """Run the analysis script on the aggregated data
    
    Args:
        h5_path: Path to the aggregated HDF5 file
        output_dir: Directory to save the analysis results
    """
    import subprocess
    
    cmd = [
        "python", "analyze_spectral_results.py",
        "--h5_file", h5_path,
        "--output_dir", output_dir
    ]
    
    print(f"Running analysis command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Aggregate spectral analysis results from multiple runs")
    parser.add_argument("--input_dir", type=str, default="out-spectral",
                        help="Directory containing run directories with HDF5 files")
    parser.add_argument("--output_dir", type=str, default="out-spectral",
                        help="Directory to save aggregated data and analysis results")
    parser.add_argument("--figures_dir", type=str, default="figures_1000_runs",
                        help="Directory to save analysis figures")
    parser.add_argument("--run_analysis", action="store_true", 
                        help="Run analysis script on aggregated data")
    args = parser.parse_args()
    
    # Find all HDF5 files
    h5_files = find_all_h5_files(args.input_dir)
    print(f"Found {len(h5_files)} HDF5 files")
    
    if not h5_files:
        print("No HDF5 files found. Exiting.")
        return
    
    # Aggregate data
    aggregated_analyzer = aggregate_spectral_data(h5_files)
    
    # Save aggregated data
    h5_path = save_aggregated_data(aggregated_analyzer, args.output_dir, len(h5_files))
    print(f"Saved aggregated data to {h5_path}")
    
    # Run analysis if requested
    if args.run_analysis:
        figures_dir = os.path.join(args.output_dir, args.figures_dir)
        run_analysis(h5_path, figures_dir)
        print(f"Analysis results saved to {figures_dir}")
    
    print("Done!")

if __name__ == "__main__":
    import torch  # Needed for the dummy model
    main() 

# cd LossGeometry/models/nanoGPT && python aggregate_all_spectral_runs.py --input_dir out-spectral --output_dir out-spectral --figures_dir figures_1000_runs --run_analysis