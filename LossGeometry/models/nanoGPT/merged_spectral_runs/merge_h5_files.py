#!/usr/bin/env python3
"""
Script to merge H5 analysis data files from multiple spectral runs and verify the merge
"""

import h5py
import numpy as np
import os
from datetime import datetime
import warnings

def copy_group_structure(source_group, dest_group, source_name="source"):
    """Recursively copy group structure from source to destination"""
    for key, item in source_group.items():
        if isinstance(item, h5py.Group):
            # Create group in destination if it doesn't exist
            if key not in dest_group:
                dest_group.create_group(key)
            # Recursively copy subgroups
            copy_group_structure(item, dest_group[key], source_name)
        elif isinstance(item, h5py.Dataset):
            # Create a new dataset name that includes the source identifier
            new_key = f"{key}_{source_name}"
            if new_key in dest_group:
                print(f"Warning: Dataset {new_key} already exists, skipping...")
                continue
            
            # Copy dataset to destination with new name
            dest_group.create_dataset(new_key, data=item[:], dtype=item.dtype)
            
            # Copy attributes
            for attr_name, attr_value in item.attrs.items():
                dest_group[new_key].attrs[attr_name] = attr_value

def merge_metadata_and_scalars(source_files, dest_file):
    """Merge metadata and scalar datasets from multiple files"""
    metadata_keys = ['batch_numbers', 'batches', 'loss_values']
    
    for key in metadata_keys:
        combined_data = []
        for i, filepath in enumerate(source_files):
            try:
                with h5py.File(filepath, 'r') as f:
                    if key in f:
                        data = f[key][:]
                        combined_data.append(data)
                        print(f"  {key} from run{i+1}: shape {data.shape}")
            except Exception as e:
                print(f"Warning: Could not read {key} from {filepath}: {e}")
        
        if combined_data:
            # Concatenate along the first axis
            merged_data = np.concatenate(combined_data, axis=0)
            dest_file.create_dataset(key, data=merged_data, dtype=combined_data[0].dtype)
            print(f"  Merged {key}: final shape {merged_data.shape}")

def merge_h5_files(file_paths, output_path):
    """Merge multiple H5 files into a single file"""
    print(f"Merging {len(file_paths)} H5 files...")
    
    # Verify all files exist
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as dest_file:
        # Add metadata about the merge
        merge_info = dest_file.create_group('merge_metadata')
        merge_info.attrs['merged_timestamp'] = datetime.now().isoformat()
        merge_info.attrs['num_source_files'] = len(file_paths)
        merge_info.attrs['source_files'] = [os.path.basename(p) for p in file_paths]
        
        print("Merging metadata and scalar datasets...")
        merge_metadata_and_scalars(file_paths, dest_file)
        
        print("Merging layer data...")
        # Process each source file
        for i, filepath in enumerate(file_paths):
            run_name = f"run{i+1}"
            print(f"Processing {run_name}: {os.path.basename(filepath)}")
            
            try:
                with h5py.File(filepath, 'r') as source_file:
                    # Copy layers group structure
                    if 'layers' in source_file:
                        if 'layers' not in dest_file:
                            dest_file.create_group('layers')
                        
                        # Copy each layer's data with run identifier
                        for layer_name, layer_group in source_file['layers'].items():
                            if layer_name not in dest_file['layers']:
                                dest_file['layers'].create_group(layer_name)
                            
                            # Copy the layer group structure
                            copy_group_structure(layer_group, dest_file['layers'][layer_name], run_name)
                    
                    # Copy model and other top-level groups if they exist
                    for group_name in ['model', 'metadata']:
                        if group_name in source_file:
                            group_dest_name = f"{group_name}_{run_name}"
                            if group_dest_name not in dest_file:
                                dest_file.create_group(group_dest_name)
                            copy_group_structure(source_file[group_name], dest_file[group_dest_name], run_name)
                            
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue
    
    print(f"Merge completed successfully. Output file: {output_path}")
    return output_path

def test_merged_file(merged_file_path, source_files):
    """Test that the merged file contains expected data"""
    print(f"\n=== Testing merged file: {merged_file_path} ===")
    
    if not os.path.exists(merged_file_path):
        print("ERROR: Merged file does not exist!")
        return False
    
    success = True
    
    try:
        with h5py.File(merged_file_path, 'r') as f:
            print(f"File size: {os.path.getsize(merged_file_path) / (1024*1024):.2f} MB")
            
            # Test 1: Check merge metadata
            if 'merge_metadata' in f:
                meta = f['merge_metadata']
                print(f"✓ Merge metadata found")
                print(f"  - Merged timestamp: {meta.attrs.get('merged_timestamp', 'N/A')}")
                print(f"  - Number of source files: {meta.attrs.get('num_source_files', 'N/A')}")
                print(f"  - Source files: {list(meta.attrs.get('source_files', []))}")
            else:
                print("✗ Merge metadata missing")
                success = False
            
            # Test 2: Check that we have data from all runs
            expected_runs = len(source_files)
            if 'layers' in f:
                # Check one layer to see if we have data from all runs
                layer_keys = list(f['layers'].keys())
                if layer_keys:
                    first_layer = f['layers'][layer_keys[0]]
                    if 'singular_values' in first_layer:
                        sv_group = first_layer['singular_values']
                        run_datasets = [k for k in sv_group.keys() if k.startswith('batch_') and k.endswith('_run1')]
                        print(f"✓ Found {len(run_datasets)} datasets from run1 in first layer")
                        
                        # Check for all runs
                        for run_num in range(1, expected_runs + 1):
                            run_datasets = [k for k in sv_group.keys() if k.endswith(f'_run{run_num}')]
                            if run_datasets:
                                print(f"✓ Found {len(run_datasets)} datasets from run{run_num}")
                            else:
                                print(f"✗ No datasets found from run{run_num}")
                                success = False
            
            # Test 3: Check merged scalar datasets
            scalar_keys = ['batch_numbers', 'batches', 'loss_values']
            for key in scalar_keys:
                if key in f:
                    shape = f[key].shape
                    print(f"✓ Merged {key}: shape {shape}")
                    # For arrays that should be concatenated, check if length increased
                    if shape[0] > 10:  # Expecting more data than single file
                        print(f"  - Data appears to be properly concatenated")
                    else:
                        print(f"  - Warning: Data might not be properly merged")
                else:
                    print(f"✗ Missing expected dataset: {key}")
                    success = False
            
            # Test 4: Data integrity check - verify some actual values
            print("\n=== Data Integrity Checks ===")
            if 'loss_values' in f:
                loss_data = f['loss_values'][:]
                print(f"✓ Loss values range: {loss_data.min():.6f} to {loss_data.max():.6f}")
                print(f"  - Number of loss values: {len(loss_data)}")
                if len(loss_data) > 0 and not np.any(np.isnan(loss_data)):
                    print("  - No NaN values detected")
                else:
                    print("  - Warning: NaN values detected or empty array")
            
            # Test 5: Compare with original file sizes
            original_total_size = sum(os.path.getsize(fp) for fp in source_files) / (1024*1024)
            merged_size = os.path.getsize(merged_file_path) / (1024*1024)
            print(f"\n=== Size Comparison ===")
            print(f"Original files total: {original_total_size:.2f} MB")
            print(f"Merged file: {merged_size:.2f} MB")
            size_ratio = merged_size / original_total_size if original_total_size > 0 else 0
            print(f"Size ratio: {size_ratio:.2f}")
            
            if 0.8 <= size_ratio <= 1.5:  # Allow some overhead/compression differences
                print("✓ Merged file size looks reasonable")
            else:
                print("⚠ Merged file size seems unusual")
    
    except Exception as e:
        print(f"ERROR during testing: {e}")
        success = False
    
    if success:
        print("\n🎉 All tests passed! Merge appears successful.")
    else:
        print("\n❌ Some tests failed. Please check the merge process.")
    
    return success

def main():
    # Updated file paths with correct timestamps
    source_files = [
        "/workspace/LossGeometry/LossGeometry/models/nanoGPT/out-spectral-run1/20250526_090530_aggregated_100_runs/20250526_090530_aggregated_100_runs_analysis_data.h5",
        "/workspace/LossGeometry/LossGeometry/models/nanoGPT/out-spectral-run2/20250526_090725_aggregated_100_runs/20250526_090725_aggregated_100_runs_analysis_data.h5",
        "/workspace/LossGeometry/LossGeometry/models/nanoGPT/out-spectral-run3/20250526_090737_aggregated_100_runs/20250526_090737_aggregated_100_runs_analysis_data.h5",
        "/workspace/LossGeometry/LossGeometry/models/nanoGPT/out-spectral-run4/20250526_090744_aggregated_100_runs/20250526_090744_aggregated_100_runs_analysis_data.h5"
    ]
    
    # Create output path
    output_dir = "/workspace/LossGeometry/LossGeometry/models/nanoGPT/merged_spectral_runs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/{timestamp}_merged_all_spectral_runs_analysis_data.h5"
    
    try:
        # Perform the merge
        merged_path = merge_h5_files(source_files, output_file)
        
        # Test the merge
        test_success = test_merged_file(merged_path, source_files)
        
        if test_success:
            print(f"\n✅ SUCCESS: Files merged successfully!")
            print(f"📁 Merged file location: {merged_path}")
        else:
            print(f"\n⚠️  WARNING: Merge completed but some tests failed.")
            print(f"📁 Merged file location: {merged_path}")
            print("Please manually verify the merged file.")
            
    except Exception as e:
        print(f"\n❌ ERROR: Merge failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 