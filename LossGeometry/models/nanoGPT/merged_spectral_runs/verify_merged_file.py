#!/usr/bin/env python3
"""
Quick verification script to examine the merged H5 file structure
"""

import h5py
import numpy as np
import os

def print_h5_structure(group, prefix="", max_datasets=5):
    """Print the structure of an H5 group"""
    for i, (key, item) in enumerate(group.items()):
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/")
            if len(prefix) < 20:  # Limit depth to avoid too much output
                print_h5_structure(item, prefix + "  ", max_datasets)
        elif isinstance(item, h5py.Dataset):
            if i < max_datasets:  # Limit number of datasets shown per group
                print(f"{prefix}{key}: {item.shape}, {item.dtype}")
            elif i == max_datasets:
                print(f"{prefix}... and {len(group) - max_datasets} more datasets")

def verify_merged_file():
    merged_file = "/workspace/LossGeometry/LossGeometry/models/nanoGPT/merged_spectral_runs/20250526_151326_merged_all_spectral_runs_analysis_data.h5"
    
    print(f"=== Verifying merged file ===")
    print(f"File: {os.path.basename(merged_file)}")
    print(f"Size: {os.path.getsize(merged_file) / (1024*1024):.2f} MB")
    
    with h5py.File(merged_file, 'r') as f:
        print(f"\nTop-level structure:")
        print_h5_structure(f)
        
        # Check some specific data
        print(f"\n=== Quick Data Sample ===")
        if 'loss_values' in f:
            loss_data = f['loss_values'][:]
            print(f"Loss values summary:")
            print(f"  - Shape: {loss_data.shape}")
            print(f"  - Range: {loss_data.min():.6f} to {loss_data.max():.6f}")
            print(f"  - Mean: {loss_data.mean():.6f}")
            print(f"  - First few values: {loss_data[:5]}")
            print(f"  - Last few values: {loss_data[-5:]}")
        
        # Check layer data
        if 'layers' in f:
            layers = f['layers']
            print(f"\nLayers found: {list(layers.keys())}")
            
            # Check first layer
            first_layer_name = list(layers.keys())[0]
            first_layer = layers[first_layer_name]
            print(f"\nFirst layer '{first_layer_name}' contains:")
            for analysis_type in first_layer.keys():
                print(f"  - {analysis_type}/")
                if analysis_type in first_layer:
                    datasets = list(first_layer[analysis_type].keys())
                    run_counts = {}
                    for dataset in datasets:
                        if '_run' in dataset:
                            run_id = dataset.split('_run')[-1]
                            run_counts[run_id] = run_counts.get(run_id, 0) + 1
                    print(f"    Datasets per run: {run_counts}")
        
        print(f"\n✅ Verification complete!")

if __name__ == "__main__":
    verify_merged_file() 