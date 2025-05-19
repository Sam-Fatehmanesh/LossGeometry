#!/usr/bin/env python
"""
Generate test data for spectral analysis visualization
"""

import os
import sys
import numpy as np
import torch
from datetime import datetime
import h5py

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

def generate_random_eigenvalues(size, batches=5):
    """Generate random eigenvalues for testing"""
    eigenvalues_list = []
    for i in range(batches):
        # Create circular law distribution for eigenvalues
        # For real matrices: real parts follow a half-circle distribution
        x = np.random.normal(0, 1, size=(size, size)) / np.sqrt(size)
        eigenvalues = np.linalg.eigvals(x)
        eigenvalues_list.append(eigenvalues)
    return eigenvalues_list

def generate_random_singular_values(size, batches=5):
    """Generate random singular values for testing"""
    singular_values_list = []
    for i in range(batches):
        # Create rectangular random matrix
        x = np.random.normal(0, 1, size=(size, size)) / np.sqrt(size)
        # Get singular values
        singular_values = np.linalg.svd(x, compute_uv=False)
        singular_values_list.append(singular_values)
    return singular_values_list

def generate_level_spacing_data(size, batches=5):
    """Generate random level spacing data for testing"""
    std_dev_list = []
    last_spacings = None
    for i in range(batches):
        # Create random matrix
        x = np.random.normal(0, 1, size=(size, size)) / np.sqrt(size)
        # Make it symmetric for real eigenvalues
        x = 0.5 * (x + x.T)
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(x)
        # Sort eigenvalues
        eigenvalues = np.sort(eigenvalues)
        # Calculate spacings
        spacings = eigenvalues[1:] - eigenvalues[:-1]
        # Normalize spacings
        mean_spacing = np.mean(spacings)
        if mean_spacing > 0:
            normalized_spacings = spacings / mean_spacing
            std_dev = np.std(normalized_spacings)
            std_dev_list.append(std_dev)
            last_spacings = normalized_spacings
    return std_dev_list, last_spacings

def create_test_data():
    """Create test data for spectral analysis"""
    # Parameters
    batch_numbers = list(range(0, 2000, 400))  # 5 batches
    loss_values = [4.0, 3.5, 3.0, 2.8, 2.5]    # Decreasing loss
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create dictionary for PT file
    data = {
        'batch': batch_numbers,
        'batch_numbers': batch_numbers,
        'loss_values': loss_values,
        'matrix_type': 'W',
        'analysis_type': 'spectral',
        'results': {}
    }
    
    # Add data for three layers
    layer_params = [
        ('transformer.h.0.mlp.c_fc.weight', (512, 2048)),
        ('transformer.h.1.attn.c_proj.weight', (512, 512)),
        ('transformer.h.2.mlp.c_proj.weight', (2048, 512))
    ]
    
    for layer_name, shape in layer_params:
        m, n = shape
        data['results'][layer_name] = {
            'shape': shape,
            'eigenvalues_list': generate_random_eigenvalues(min(m, n), len(batch_numbers)),
            'singular_values_list': generate_random_singular_values(min(m, n), len(batch_numbers)),
            'level_spacing_std_list': generate_level_spacing_data(min(m, n), len(batch_numbers))[0],
            'level_spacing_last': generate_level_spacing_data(min(m, n), len(batch_numbers))[1]
        }
    
    # Save as PT file
    pt_file = os.path.join('out-spectral/spectral', 'spectral_stats_final.pt')
    torch.save(data, pt_file)
    print(f"Saved test data to: {pt_file}")
    
    # Also save as H5 file
    h5_dir = os.path.join('out-spectral', timestamp)
    os.makedirs(h5_dir, exist_ok=True)
    h5_file = os.path.join(h5_dir, f"{timestamp}_analysis_data.h5")
    
    with h5py.File(h5_file, 'w') as f:
        # Save metadata
        meta_group = f.create_group('metadata')
        meta_group.attrs['timestamp'] = timestamp
        meta_group.attrs['matrix_type'] = data['matrix_type']
        meta_group.attrs['analysis_type'] = data['analysis_type']
        
        # Save batch and loss data
        f.create_dataset('batch_numbers', data=np.array(data['batch_numbers']))
        f.create_dataset('batches', data=np.array(data['batch']))
        f.create_dataset('loss_values', data=np.array(data['loss_values']))
        
        # Save layer results
        layers_group = f.create_group('layers')
        for layer_name, layer_data in data['results'].items():
            layer_group = layers_group.create_group(layer_name)
            
            # Save shape
            layer_group.attrs['shape'] = str(layer_data['shape'])
            
            # Save eigenvalues
            eigen_group = layer_group.create_group('eigenvalues')
            for i, eig_array in enumerate(layer_data['eigenvalues_list']):
                eigen_group.create_dataset(f'batch_{i}', data=eig_array)
            
            # Save singular values
            sv_group = layer_group.create_group('singular_values')
            for i, sv_array in enumerate(layer_data['singular_values_list']):
                sv_group.create_dataset(f'batch_{i}', data=sv_array)
            
            # Save level spacing data
            spacing_group = layer_group.create_group('spacing')
            spacing_group.create_dataset('std_dev_list', data=np.array(layer_data['level_spacing_std_list']))
            spacing_group.create_dataset('last_spacings', data=layer_data['level_spacing_last'])
    
    print(f"Saved test data to: {h5_file}")
    return pt_file, h5_file

if __name__ == "__main__":
    create_test_data() 