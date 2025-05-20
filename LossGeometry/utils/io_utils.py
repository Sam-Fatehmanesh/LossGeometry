import h5py
import os
import numpy as np
import json
from datetime import datetime

def save_analysis_data(analyzer, model, experiment_dir, timestamp=None, num_runs=1):
    """
    Save analysis data to HDF5 format
    
    Args:
        analyzer: The SpectralAnalyzer object with collected data
        model: The model being analyzed
        experiment_dir (str): Base directory for experiments
        timestamp (str, optional): Timestamp for the experiment, if None, will generate one
        num_runs (int): Number of runs the results are averaged over
        
    Returns:
        str: Path to the saved file
    """
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment directory
    experiment_path = os.path.join(experiment_dir, timestamp)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok=True)
        print(f"Created experiment directory: {experiment_path}")
    
    # Path for the HDF5 file
    h5_path = os.path.join(experiment_path, f"{timestamp}_analysis_data.h5")
    
    # Get analysis stats
    stats = analyzer.get_stats()
    
    # Save data to HDF5
    with h5py.File(h5_path, 'w') as f:
        # Save metadata
        meta_group = f.create_group('metadata')
        meta_group.attrs['timestamp'] = timestamp
        meta_group.attrs['matrix_type'] = stats['matrix_type']
        meta_group.attrs['analysis_type'] = stats['analysis_type']
        meta_group.attrs['matrix_description'] = analyzer.matrix_description
        meta_group.attrs['num_runs'] = num_runs
        
        # Save model info as attributes
        model_group = f.create_group('model')
        model_group.attrs['input_size'] = model.input_size
        model_group.attrs['hidden_size'] = model.hidden_size
        model_group.attrs['output_size'] = model.output_size
        model_group.attrs['num_hidden_layers'] = model.num_hidden_layers
        
        # Save batch and loss data
        f.create_dataset('batch_numbers', data=np.array(stats['batch_numbers']))
        f.create_dataset('batches', data=np.array(stats['batch']))
        f.create_dataset('loss_values', data=np.array(stats['loss_values']))
        
        # Save layer results
        layers_group = f.create_group('layers')
        for layer_name, layer_data in stats['results'].items():
            layer_group = layers_group.create_group(layer_name)
            
            # Save shape
            target_param = model.get_parameter(layer_name)
            layer_group.attrs['shape'] = str(target_param.shape)
            
            # Save eigenvalues if they exist
            if 'eigenvalues_list' in layer_data and layer_data['eigenvalues_list']:
                eigen_group = layer_group.create_group('eigenvalues')
                for i, eig_array in enumerate(layer_data['eigenvalues_list']):
                    if eig_array is not None:
                        eigen_group.create_dataset(f'batch_{i}', data=eig_array)
            
            # Save level spacing data if it exists
            if 'std_dev_norm_spacing_list' in layer_data and layer_data['std_dev_norm_spacing_list']:
                spacing_group = layer_group.create_group('spacing')
                spacing_group.create_dataset('std_dev_list', data=np.array(layer_data['std_dev_norm_spacing_list']))
                if layer_data['last_normalized_spacings'] is not None:
                    spacing_group.create_dataset('last_spacings', data=layer_data['last_normalized_spacings'])
            
            # Save singular values if they exist
            if 'singular_values_list' in layer_data and layer_data['singular_values_list']:
                sv_group = layer_group.create_group('singular_values')
                for i, sv_array in enumerate(layer_data['singular_values_list']):
                    if sv_array is not None:
                        sv_group.create_dataset(f'batch_{i}', data=sv_array)
    
    print(f"Analysis data saved to {h5_path}")
    return h5_path

def load_analysis_data(h5_path):
    """
    Load analysis data from HDF5 file
    
    Args:
        h5_path (str): Path to the HDF5 file
        
    Returns:
        dict: Analysis data
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"File not found: {h5_path}")
    
    data = {
        'metadata': {},
        'model': {},
        'batch_numbers': [],
        'batch': [],  # Use 'batch' to match old stats format
        'loss_values': [],
        'results': {}
    }
    
    with h5py.File(h5_path, 'r') as f:
        # Load metadata
        if 'metadata' in f:
            for key, value in f['metadata'].attrs.items():
                data['metadata'][key] = value
            # Copy metadata keys to top level for compatibility
            if 'matrix_type' in data['metadata']:
                data['matrix_type'] = data['metadata']['matrix_type']
            if 'analysis_type' in data['metadata']:
                data['analysis_type'] = data['metadata']['analysis_type']
            # Default to 1 run if not found
            if 'num_runs' not in data['metadata']:
                data['metadata']['num_runs'] = 1
        
        # Load model info
        if 'model' in f:
            for key, value in f['model'].attrs.items():
                data['model'][key] = value
        
        # Load batch and loss data
        if 'batch_numbers' in f:
            data['batch_numbers'] = f['batch_numbers'][()].tolist()
        if 'batches' in f:
            data['batch'] = f['batches'][()].tolist()
        if 'loss_values' in f:
            data['loss_values'] = f['loss_values'][()].tolist()
        
        # Load layer results
        if 'layers' in f:
            for layer_name in f['layers']:
                layer_group = f['layers'][layer_name]
                
                # Safely parse shape without eval
                shape = None
                if 'shape' in layer_group.attrs:
                    shape_str = layer_group.attrs['shape']
                    # Try to parse a tuple like "(512, 512)" safely
                    if shape_str.startswith('(') and shape_str.endswith(')'):
                        try:
                            # Remove parentheses and split by comma
                            dims = shape_str.strip('()').split(',')
                            # Convert each dimension to int
                            shape = tuple(int(dim.strip()) for dim in dims if dim.strip())
                        except (ValueError, SyntaxError):
                            # If parsing fails, store as string
                            shape = shape_str
                
                data['results'][layer_name] = {
                    'shape': shape
                }
                
                # Load eigenvalues
                if 'eigenvalues' in layer_group:
                    eigen_group = layer_group['eigenvalues']
                    eigenvalues_list = []
                    for i in range(len(eigen_group)):
                        batch_key = f'batch_{i}'
                        if batch_key in eigen_group:
                            eigenvalues_list.append(eigen_group[batch_key][()].tolist())
                    data['results'][layer_name]['eigenvalues_list'] = eigenvalues_list
                    if eigenvalues_list:
                        data['results'][layer_name]['last_eigenvalues'] = eigenvalues_list[-1]
                
                # Load level spacing data
                if 'spacing' in layer_group:
                    spacing_group = layer_group['spacing']
                    if 'std_dev_list' in spacing_group:
                        data['results'][layer_name]['std_dev_norm_spacing_list'] = spacing_group['std_dev_list'][()].tolist()
                    if 'last_spacings' in spacing_group:
                        data['results'][layer_name]['last_normalized_spacings'] = spacing_group['last_spacings'][()].tolist()
                
                # Load singular values
                if 'singular_values' in layer_group:
                    sv_group = layer_group['singular_values']
                    sv_list = []
                    for i in range(len(sv_group)):
                        batch_key = f'batch_{i}'
                        if batch_key in sv_group:
                            sv_list.append(sv_group[batch_key][()].tolist())
                    data['results'][layer_name]['singular_values_list'] = sv_list
                    if sv_list:
                        data['results'][layer_name]['last_singular_values'] = sv_list[-1]
    
    print(f"Loaded analysis data from {h5_path}")
    return data

def get_experiment_dir(base_dir="experiments"):
    """
    Get experiment directory, creating it if it doesn't exist
    
    Args:
        base_dir (str): Base directory for experiments
        
    Returns:
        str: Path to the experiment directory
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        print(f"Created experiment directory: {base_dir}")
    return base_dir 