"""
Analyze spectral properties collected during nanoGPT training on the Pile dataset.

This script generates visualizations and analyses of eigenvalue and singular value
distributions for the weight matrices of the nanoGPT model.

Examples:
    $ python analyze_spectral_results.py --results_dir out-spectral/spectral --output_dir figures
    $ python analyze_spectral_results.py --h5_file out-spectral/20240515_123456/20240515_123456_analysis_data.h5 --output_dir figures
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from tqdm import tqdm
import json
from pathlib import Path
import h5py
import glob
from datetime import datetime

# Set style for plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Add parent directory to path for io_utils import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from LossGeometry.utils.io_utils import load_analysis_data
from LossGeometry.visualization.plot_utils import AnalysisPlotter

# JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def find_latest_results(results_dir):
    """Find the latest spectral analysis results file"""
    # First check for PT files
    result_files = [f for f in os.listdir(results_dir) if f.startswith('spectral_stats_') and f.endswith('.pt')]
    if len(result_files) > 0:
        # If we have a final file, use that
        if 'spectral_stats_final.pt' in result_files:
            return os.path.join(results_dir, 'spectral_stats_final.pt'), 'pt'
        
        # Otherwise sort by iteration number and take the latest
        iter_nums = [int(f.split('_')[-1].split('.')[0]) for f in result_files if f != 'spectral_stats_final.pt']
        if iter_nums:
            latest_iter = max(iter_nums)
            return os.path.join(results_dir, f'spectral_stats_{latest_iter}.pt'), 'pt'
    
    # Then check for HDF5 files
    h5_pattern = os.path.join(results_dir, "**", "*_analysis_data.h5")
    h5_files = glob.glob(h5_pattern, recursive=True)
    
    if h5_files:
        # Sort by modification time (newest first)
        h5_files.sort(key=os.path.getmtime, reverse=True)
        return h5_files[0], 'h5'
    
    return None, None

def plot_eigenvalue_histograms(stats, output_dir, last_n=1, selected_layers=None):
    """
    Plot histograms of eigenvalues at different training stages
    
    Args:
        stats: The loaded spectral analysis results
        output_dir: Directory to save the plots
        last_n: Number of snapshots to plot (from the end of training)
        selected_layers: List of layer names to plot (if None, plot all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If selected_layers is None, use all layers
    if selected_layers is None:
        selected_layers = list(stats['results'].keys())
    
    for layer_name in selected_layers:
        layer_stats = stats['results'].get(layer_name, None)
        if layer_stats is None or 'eigenvalues_list' not in layer_stats:
            continue  # Skip layers without eigenvalue analysis
        
        eigenvalues_list = layer_stats['eigenvalues_list']
        if not eigenvalues_list:
            continue
        
        # Get snapshots to plot
        if len(eigenvalues_list) <= last_n:
            snapshots = range(len(eigenvalues_list))
        else:
            snapshots = range(len(eigenvalues_list) - last_n, len(eigenvalues_list))
        
        for snapshot_idx in snapshots:
            batch_num = stats['batch'][snapshot_idx] if snapshot_idx < len(stats['batch']) else "unknown"
            eigenvalues = eigenvalues_list[snapshot_idx]
            
            # Skip if we don't have eigenvalues
            if eigenvalues is None or len(eigenvalues) == 0:
                continue
            
            # Create the histogram
            plt.figure(figsize=(10, 6))
            
            # If we have lots of eigenvalues, plot a histogram
            if len(eigenvalues) > 30:
                sns.histplot(eigenvalues, bins=50, kde=True)
            else:
                # For smaller matrices, just plot the individual eigenvalues
                sns.kdeplot(eigenvalues)
                plt.scatter(eigenvalues, np.zeros_like(eigenvalues), marker='o', s=50, color='red', alpha=0.5)
            
            plt.title(f"Eigenvalue Distribution - {layer_name}\nBatch {batch_num}")
            plt.xlabel("Eigenvalue")
            plt.ylabel("Density")
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
            plt.savefig(os.path.join(output_dir, f"eig_dist_{safe_layer_name}_batch_{batch_num}.png"), dpi=300, bbox_inches='tight')
            plt.close()

def plot_singular_value_histograms(stats, output_dir, last_n=1, selected_layers=None):
    """
    Plot histograms of singular values at different training stages
    
    Args:
        stats: The loaded spectral analysis results
        output_dir: Directory to save the plots
        last_n: Number of snapshots to plot (from the end of training)
        selected_layers: List of layer names to plot (if None, plot all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If selected_layers is None, use all layers
    if selected_layers is None:
        selected_layers = list(stats['results'].keys())
    
    for layer_name in selected_layers:
        layer_stats = stats['results'].get(layer_name, None)
        if layer_stats is None or 'singular_values_list' not in layer_stats:
            continue  # Skip layers without singular value analysis
        
        singular_values_list = layer_stats['singular_values_list']
        if not singular_values_list:
            continue
        
        # Get snapshots to plot
        if len(singular_values_list) <= last_n:
            snapshots = range(len(singular_values_list))
        else:
            snapshots = range(len(singular_values_list) - last_n, len(singular_values_list))
        
        for snapshot_idx in snapshots:
            batch_num = stats['batch'][snapshot_idx] if snapshot_idx < len(stats['batch']) else "unknown"
            singular_values = singular_values_list[snapshot_idx]
            
            # Skip if we don't have singular values
            if singular_values is None or len(singular_values) == 0:
                continue
            
            # Create the histogram
            plt.figure(figsize=(10, 6))
            
            # If we have lots of singular values, plot a histogram
            if len(singular_values) > 30:
                sns.histplot(singular_values, bins=50, kde=True)
            else:
                # For smaller matrices, just plot the individual singular values
                sns.kdeplot(singular_values)
                plt.scatter(singular_values, np.zeros_like(singular_values), marker='o', s=50, color='red', alpha=0.5)
            
            plt.title(f"Singular Value Distribution - {layer_name}\nBatch {batch_num}")
            plt.xlabel("Singular Value")
            plt.ylabel("Density")
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
            plt.savefig(os.path.join(output_dir, f"sv_dist_{safe_layer_name}_batch_{batch_num}.png"), dpi=300, bbox_inches='tight')
            plt.close()

def plot_spectral_properties_over_time(stats, output_dir, selected_layers=None):
    """
    Plot how spectral properties evolve over training iterations
    
    Args:
        stats: The loaded spectral analysis results
        output_dir: Directory to save the plots
        selected_layers: List of layer names to plot (if None, plot all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get batch numbers for x-axis
    batch_numbers = stats['batch']
    if not batch_numbers:
        print("No batch numbers found in stats.")
        return
    
    # If selected_layers is None, use all layers
    if selected_layers is None:
        selected_layers = list(stats['results'].keys())
    
    # Plot loss over time if available
    if stats['loss_values'] and stats['batch_numbers']:
        plt.figure(figsize=(10, 6))
        plt.plot(stats['batch_numbers'], stats['loss_values'], 'b-', linewidth=2)
        plt.title("Loss During Training")
        plt.xlabel("Batch Number")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "loss_over_time.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create plots for:
    # 1. Max eigenvalue over time
    # 2. Mean eigenvalue over time
    # 3. Max singular value over time
    # 4. "Effective rank" over time (sum of singular values / max singular value)
    
    # Initialize dictionaries to store metrics for each layer
    max_eigenvalues = {}
    mean_eigenvalues = {}
    max_singular_values = {}
    effective_ranks = {}
    
    # Calculate metrics for each layer
    for layer_name in selected_layers:
        layer_stats = stats['results'].get(layer_name, None)
        if layer_stats is None:
            continue
        
        # Process eigenvalues if available
        if 'eigenvalues_list' in layer_stats and layer_stats['eigenvalues_list']:
            max_eig = []
            mean_eig = []
            for eigs in layer_stats['eigenvalues_list']:
                if eigs is not None and len(eigs) > 0:
                    max_eig.append(np.max(np.abs(eigs)))
                    mean_eig.append(np.mean(np.abs(eigs)))
                else:
                    max_eig.append(np.nan)
                    mean_eig.append(np.nan)
            
            # Only add if we have at least one valid value
            if not all(np.isnan(v) for v in max_eig):
                max_eigenvalues[layer_name] = max_eig
            if not all(np.isnan(v) for v in mean_eig):
                mean_eigenvalues[layer_name] = mean_eig
        
        # Process singular values if available
        if 'singular_values_list' in layer_stats and layer_stats['singular_values_list']:
            max_sv = []
            eff_rank = []
            for svs in layer_stats['singular_values_list']:
                if svs is not None and len(svs) > 0:
                    max_sv.append(np.max(svs))
                    # Calculate effective rank: sum(s_i) / s_1
                    # This is a simplification; sometimes trace(S) / ||S|| is used
                    eff_rank.append(np.sum(svs) / (np.max(svs) * len(svs)))
                else:
                    max_sv.append(np.nan)
                    eff_rank.append(np.nan)
            
            # Only add if we have at least one valid value
            if not all(np.isnan(v) for v in max_sv):
                max_singular_values[layer_name] = max_sv
            if not all(np.isnan(v) for v in eff_rank):
                effective_ranks[layer_name] = eff_rank
    
    # Plot max eigenvalues over time
    if max_eigenvalues:
        plt.figure(figsize=(12, 8))
        for layer_name, values in max_eigenvalues.items():
            # Only plot if we have enough values matching the batch numbers
            if len(values) == len(batch_numbers):
                # Convert list to string to avoid matplotlib warning
                label = '.'.join(layer_name.split('.')[-2:])
                plt.plot(batch_numbers, values, label=label)
        plt.title("Maximum Absolute Eigenvalue During Training")
        plt.xlabel("Batch Number")
        plt.ylabel("Max |Eigenvalue|")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "max_eigenvalue_over_time.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot mean eigenvalues over time
    if mean_eigenvalues:
        plt.figure(figsize=(12, 8))
        for layer_name, values in mean_eigenvalues.items():
            if len(values) == len(batch_numbers):
                # Convert list to string to avoid matplotlib warning
                label = '.'.join(layer_name.split('.')[-2:])
                plt.plot(batch_numbers, values, label=label)
        plt.title("Mean Absolute Eigenvalue During Training")
        plt.xlabel("Batch Number")
        plt.ylabel("Mean |Eigenvalue|")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mean_eigenvalue_over_time.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot max singular values over time
    if max_singular_values:
        plt.figure(figsize=(12, 8))
        for layer_name, values in max_singular_values.items():
            if len(values) == len(batch_numbers):
                # Convert list to string to avoid matplotlib warning
                label = '.'.join(layer_name.split('.')[-2:])
                plt.plot(batch_numbers, values, label=label)
        plt.title("Maximum Singular Value During Training")
        plt.xlabel("Batch Number")
        plt.ylabel("Max Singular Value")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "max_singular_value_over_time.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot effective rank over time
    if effective_ranks:
        plt.figure(figsize=(12, 8))
        for layer_name, values in effective_ranks.items():
            if len(values) == len(batch_numbers):
                # Convert list to string to avoid matplotlib warning
                label = '.'.join(layer_name.split('.')[-2:])
                plt.plot(batch_numbers, values, label=label)
        plt.title("Effective Rank During Training (Normalized)")
        plt.xlabel("Batch Number")
        plt.ylabel("Effective Rank")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "effective_rank_over_time.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    return {
        'max_eigenvalues': max_eigenvalues,
        'mean_eigenvalues': mean_eigenvalues,
        'max_singular_values': max_singular_values,
        'effective_ranks': effective_ranks
    }

def analyze_weight_delta_differences(stats, output_dir, selected_layers=None):
    """
    Compare spectral properties between weight matrices (W) and weight updates (ΔW)
    
    Args:
        stats: The loaded spectral analysis results
        output_dir: Directory to save the plots
        selected_layers: List of layer names to analyze (if None, analyze all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If selected_layers is None, use all layers
    if selected_layers is None:
        selected_layers = list(stats['results'].keys())
    
    # Check if we have both W and ΔW analysis
    if stats['matrix_type'] != 'W' and stats['matrix_type'] != 'DeltaW':
        print("Cannot perform W vs ΔW comparison: stats don't contain both matrix types")
        return
    
    # We'll compare:
    # 1. Eigenvalue distributions for W vs ΔW (for the latest snapshot)
    # 2. Singular value distributions for W vs ΔW (for the latest snapshot)
    
    # Determine the latest snapshot
    if not stats['batch']:
        print("No batch information found in stats.")
        return
    
    latest_idx = len(stats['batch']) - 1
    latest_batch = stats['batch'][latest_idx]
    
    for layer_name in selected_layers:
        layer_stats = stats['results'].get(layer_name, None)
        if layer_stats is None:
            continue
        
        # Compare eigenvalues if available
        if 'eigenvalues_list' in layer_stats and layer_stats['eigenvalues_list'] and latest_idx < len(layer_stats['eigenvalues_list']):
            eigenvalues = layer_stats['eigenvalues_list'][latest_idx]
            
            if eigenvalues is not None and len(eigenvalues) > 0:
                plt.figure(figsize=(10, 6))
                sns.histplot(eigenvalues, bins=50, kde=True, label=f"{stats['matrix_type']} Eigenvalues")
                plt.title(f"Eigenvalue Distribution - {layer_name}\nBatch {latest_batch}")
                plt.xlabel("Eigenvalue")
                plt.ylabel("Density")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
                plt.savefig(os.path.join(output_dir, f"eig_comparison_{safe_layer_name}_batch_{latest_batch}.png"), dpi=300, bbox_inches='tight')
                plt.close()
        
        # Compare singular values if available
        if 'singular_values_list' in layer_stats and layer_stats['singular_values_list'] and latest_idx < len(layer_stats['singular_values_list']):
            singular_values = layer_stats['singular_values_list'][latest_idx]
            
            if singular_values is not None and len(singular_values) > 0:
                plt.figure(figsize=(10, 6))
                sns.histplot(singular_values, bins=50, kde=True, label=f"{stats['matrix_type']} Singular Values")
                plt.title(f"Singular Value Distribution - {layer_name}\nBatch {latest_batch}")
                plt.xlabel("Singular Value")
                plt.ylabel("Density")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
                plt.savefig(os.path.join(output_dir, f"sv_comparison_{safe_layer_name}_batch_{latest_batch}.png"), dpi=300, bbox_inches='tight')
                plt.close()

def generate_summary_report(stats, metrics, output_dir):
    """
    Generate a summary report of the spectral analysis results
    
    Args:
        stats: The loaded spectral analysis results
        metrics: The metrics calculated by plot_spectral_properties_over_time
        output_dir: Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary report
    report = {
        'training_summary': {
            'total_batches': len(stats['batch']),
            'final_batch': int(stats['batch'][-1]) if stats['batch'] else None,
            'matrix_type': stats['matrix_type'],
            'analysis_type': stats['analysis_type'],
        },
        'layers_analyzed': list(stats['results'].keys()),
        'key_metrics': {}
    }
    
    # Add loss information if available
    if stats['loss_values'] and stats['batch_numbers']:
        report['training_summary']['initial_loss'] = float(stats['loss_values'][0]) if stats['loss_values'] else None
        report['training_summary']['final_loss'] = float(stats['loss_values'][-1]) if stats['loss_values'] else None
    
    # Add layer-specific metrics
    for layer_name in stats['results'].keys():
        layer_metrics = {}
        
        # Add eigenvalue metrics if available
        if layer_name in metrics.get('max_eigenvalues', {}):
            values = metrics['max_eigenvalues'][layer_name]
            layer_metrics['max_eigenvalue_initial'] = float(values[0]) if values else None
            layer_metrics['max_eigenvalue_final'] = float(values[-1]) if values else None
            if values and values[0] != 0:
                change_pct = ((values[-1] - values[0]) / values[0] * 100)
                layer_metrics['max_eigenvalue_change_pct'] = float(change_pct)
        
        # Add singular value metrics if available
        if layer_name in metrics.get('max_singular_values', {}):
            values = metrics['max_singular_values'][layer_name]
            layer_metrics['max_singular_value_initial'] = float(values[0]) if values else None
            layer_metrics['max_singular_value_final'] = float(values[-1]) if values else None
            if values and values[0] != 0:
                change_pct = ((values[-1] - values[0]) / values[0] * 100)
                layer_metrics['max_singular_value_change_pct'] = float(change_pct)
        
        # Add effective rank metrics if available
        if layer_name in metrics.get('effective_ranks', {}):
            values = metrics['effective_ranks'][layer_name]
            layer_metrics['effective_rank_initial'] = float(values[0]) if values else None
            layer_metrics['effective_rank_final'] = float(values[-1]) if values else None
            if values and values[0] != 0:
                change_pct = ((values[-1] - values[0]) / values[0] * 100)
                layer_metrics['effective_rank_change_pct'] = float(change_pct)
        
        report['key_metrics'][layer_name] = layer_metrics
    
    # Save the report as JSON
    with open(os.path.join(output_dir, 'spectral_analysis_summary.json'), 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    # Also generate a simple text summary
    with open(os.path.join(output_dir, 'spectral_analysis_summary.txt'), 'w') as f:
        f.write("SPECTRAL ANALYSIS SUMMARY\n")
        f.write("========================\n\n")
        
        f.write("TRAINING INFORMATION\n")
        f.write(f"Total batches analyzed: {report['training_summary']['total_batches']}\n")
        f.write(f"Final batch: {report['training_summary']['final_batch']}\n")
        f.write(f"Matrix type: {report['training_summary']['matrix_type']}\n")
        f.write(f"Analysis type: {report['training_summary']['analysis_type']}\n")
        
        if 'initial_loss' in report['training_summary'] and 'final_loss' in report['training_summary']:
            initial_loss = report['training_summary']['initial_loss']
            final_loss = report['training_summary']['final_loss']
            f.write(f"Initial loss: {initial_loss:.4f}\n")
            f.write(f"Final loss: {final_loss:.4f}\n")
            if initial_loss:
                loss_change = (final_loss - initial_loss) / initial_loss * 100
                f.write(f"Loss change: {loss_change:.2f}%\n")
        
        f.write("\nLAYER METRICS\n")
        for layer_name, metrics in report['key_metrics'].items():
            f.write(f"\n{layer_name}:\n")
            for metric_name, value in metrics.items():
                if value is not None:
                    if 'pct' in metric_name:
                        f.write(f"  {metric_name}: {value:.2f}%\n")
                    else:
                        f.write(f"  {metric_name}: {value:.6f}\n")

def main(args):
    """
    Main function for analyzing spectral results.
    
    Args:
        args: Command line arguments
    """
    # Generate timestamp for unique output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the AnalysisPlotter
    plotter = AnalysisPlotter(args.output_dir, timestamp, dpi=300)
    
    # Load the spectral analysis data
    if args.h5_file:
        print(f"Loading data from H5 file: {args.h5_file}")
        data = load_analysis_data(args.h5_file)
    else:
        # Find the latest results file
        latest_file, file_type = find_latest_results(args.results_dir)
        if latest_file is None:
            print(f"No results found in directory: {args.results_dir}")
            return
        
        print(f"Loading data from file: {latest_file} (type: {file_type})")
        if file_type == 'pt':
            data = torch.load(latest_file)
        else:  # h5
            data = load_analysis_data(latest_file)
    
    # Extract data for plotting
    batch_numbers = data.get('batch_numbers', data.get('batch', []))
    loss_values = data.get('loss_values', [])
    
    # Plot loss if available
    if loss_values and batch_numbers:
        plotter.plot_loss(batch_numbers, loss_values, plot_title="nanoGPT Training Loss")
    
    # Selected layers filtering
    all_layers = list(data['results'].keys())
    selected_layers = args.selected_layers if args.selected_layers else all_layers
    
    # Plot spectral analysis for each selected layer
    for layer_name in selected_layers:
        if layer_name not in data['results']:
            print(f"Layer '{layer_name}' not found in results. Available layers: {all_layers}")
            continue
            
        layer_stats = data['results'][layer_name]
        
        # Extract layer shape information
        layer_shape = None
        if 'shape' in layer_stats:
            # If shape is stored directly
            layer_shape = layer_stats['shape']
        elif 'eigenvalues_list' in layer_stats and layer_stats['eigenvalues_list']:
            # Estimate shape from eigenvalue count for square matrices
            for eig_array in layer_stats['eigenvalues_list']:
                if eig_array is not None and len(eig_array) > 0:
                    n = len(eig_array)
                    layer_shape = (n, n)  # Assume square
                    break
        
        # Convert shape to tuple if it's a string representation
        if isinstance(layer_shape, str) and layer_shape.startswith('(') and layer_shape.endswith(')'):
            try:
                # Parse string like "(512, 512)" into tuple (512, 512)
                shape_str = layer_shape.strip('() ')
                shape_parts = [int(x.strip()) for x in shape_str.split(',')]
                if len(shape_parts) >= 2:
                    layer_shape = tuple(shape_parts)
            except (ValueError, SyntaxError):
                # If parsing fails, keep as string
                pass
        
        print(f"Processing layer: {layer_name}, Shape: {layer_shape}")
        
        # Matrix description for plots
        matrix_description = f"Weight Matrix - {layer_name}"
        
        # Plot eigenvalue distributions if available
        if 'eigenvalues_list' in layer_stats and layer_stats['eigenvalues_list']:
            plotter.plot_spectral_density(
                layer_name, 
                layer_shape, 
                layer_stats['eigenvalues_list'], 
                batch_numbers, 
                matrix_description
            )
        
        # Plot singular value distributions if available
        if 'singular_values_list' in layer_stats and layer_stats['singular_values_list']:
            plotter.plot_singular_values(
                layer_name, 
                layer_shape, 
                layer_stats['singular_values_list'], 
                batch_numbers, 
                matrix_description
            )
        
        # Plot level spacing if available
        level_spacing_std_key = next((k for k in ['level_spacing_std_list', 'std_dev_norm_spacing_list'] 
                                     if k in layer_stats), None)
        level_spacing_last_key = next((k for k in ['level_spacing_last', 'last_normalized_spacings'] 
                                      if k in layer_stats), None)
        
        if level_spacing_std_key and level_spacing_last_key and layer_stats[level_spacing_std_key]:
            plotter.plot_level_spacing(
                layer_name,
                layer_shape,
                layer_stats[level_spacing_std_key],
                layer_stats[level_spacing_last_key],
                batch_numbers,
                matrix_description
            )
    
    print(f"All plots have been saved to {args.output_dir}")
    
    # Generate a summary report
    metrics = {}
    metrics_file = os.path.join(args.output_dir, f"{timestamp}_metrics_summary.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, cls=NumpyEncoder, indent=2)
    
    print(f"Summary metrics saved to {metrics_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze spectral properties of nanoGPT model")
    parser.add_argument('--results_dir', type=str, default='out-spectral/spectral',
                        help='Directory containing spectral analysis results')
    parser.add_argument('--h5_file', type=str, default=None,
                        help='Direct path to HDF5 file with spectral analysis results')
    parser.add_argument('--output_dir', type=str, default='figures',
                        help='Directory to save plots and analysis results')
    parser.add_argument('--selected_layers', type=str, nargs='+', default=None,
                        help='Only analyze these specific layers (optional)')
    args = parser.parse_args()
    
    main(args) 