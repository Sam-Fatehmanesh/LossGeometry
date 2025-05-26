# Merged Spectral Analysis Data

## Overview
This directory contains the merged spectral analysis data from 4 independent nanoGPT training runs.

## Files
- **`20250526_151326_merged_all_spectral_runs_analysis_data.h5`**: The merged file containing all spectral analysis data
- **`merge_h5_files.py`**: Script used to perform the merge
- **`verify_merged_file.py`**: Script to verify the merged file structure

## Merged File Structure

The merged file contains data from 4 spectral runs, with the following structure:

### Top-level datasets:
- `batch_numbers`: (36,) - Concatenated batch numbers from all runs
- `batches`: (44,) - Concatenated batch data from all runs  
- `loss_values`: (36,) - Concatenated loss values from all runs

### Layer data:
Each transformer layer contains spectral analysis data organized as:
```
layers/
├── transformer.h.0.attn.c_attn.weight/
│   └── singular_values/
│       ├── batch_0_run1: (25600,) float32
│       ├── batch_0_run2: (25600,) float32
│       ├── batch_0_run3: (25600,) float32
│       ├── batch_0_run4: (25600,) float32
│       └── ... (11 batches × 4 runs = 44 datasets per layer/analysis)
├── transformer.h.0.attn.c_proj.weight/
│   ├── eigenvalues/ (44 datasets)
│   └── singular_values/ (44 datasets)
└── ... (17 layers total)
```

### Metadata:
- `merge_metadata/`: Information about the merge process
- `metadata_run1/`, `metadata_run2/`, etc.: Original metadata from each run
- `model_run1/`, `model_run2/`, etc.: Model information from each run

## Usage Example

```python
import h5py
import numpy as np

# Open the merged file
with h5py.File('20250526_151326_merged_all_spectral_runs_analysis_data.h5', 'r') as f:
    # Get loss values from all runs
    all_losses = f['loss_values'][:]
    print(f"Combined loss values: {all_losses.shape}")
    
    # Get singular values for a specific layer and batch from run 1
    layer_name = 'transformer.h.0.attn.c_attn.weight'
    sv_run1_batch0 = f[f'layers/{layer_name}/singular_values/batch_0_run1'][:]
    
    # Get data from all runs for comparison
    for run in range(1, 5):
        sv_data = f[f'layers/{layer_name}/singular_values/batch_0_run{run}'][:]
        print(f"Run {run} - batch 0 singular values shape: {sv_data.shape}")
```

## Verification Results

✅ **Merge successful!**
- Original files total: 90.72 MB
- Merged file: 90.61 MB  
- Size ratio: 1.00 (efficient merge)
- All 4 runs present with 11 datasets each per layer/analysis
- Loss values range: 8.20 to 10.57
- No NaN values detected
- 17 transformer layers with spectral analysis data

## Data Integrity
- Each run contributes 11 batch datasets per analysis type
- Total datasets per layer/analysis: 44 (11 batches × 4 runs)
- All original data preserved with run identifiers
- Metadata from all runs maintained separately 