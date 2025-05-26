import h5py
import os
import numpy as np
import json
import re
from datetime import datetime
import torch

def save_analysis_data(analyzer, model, experiment_dir, timestamp=None, num_runs=1, 
                       learning_rate=0.01, batch_size=64, num_epochs=1, 
                       log_every_n_batches=200, parallel_runs=1,
                       analyze_W=True, analyze_delta_W=False, analyze_singular_values=True,
                       disable_gradient_noise=False, experiment_dir_name='experiments',
                       optimizer='sgd', momentum=0.9, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    """
    Save analysis data to HDF5 format
    
    Args:
        analyzer: The SpectralAnalyzer object with collected data
        model: The model being analyzed
        experiment_dir (str): Base directory for experiments
        timestamp (str, optional): Timestamp for the experiment, if None, will generate one
        num_runs (int): Number of runs the results are averaged over
        learning_rate (float): Learning rate used for training
        batch_size (int): Batch size used for training
        num_epochs (int): Number of epochs trained
        log_every_n_batches (int): Frequency of analysis calculation
        parallel_runs (int): Number of parallel runs executed
        analyze_W (bool): Whether weight matrices were analyzed
        analyze_delta_W (bool): Whether weight update matrices were analyzed
        analyze_singular_values (bool): Whether singular values were analyzed
        disable_gradient_noise (bool): Whether gradient noise calculation was disabled
        experiment_dir_name (str): Name of the base experiment directory
        optimizer (str): Optimizer type ('sgd' or 'adam')
        momentum (float): Momentum factor for SGD optimizer
        beta1 (float): Beta1 parameter for Adam optimizer
        beta2 (float): Beta2 parameter for Adam optimizer
        eps (float): Epsilon parameter for Adam optimizer
        weight_decay (float): Weight decay (L2 penalty) for both optimizers
        
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
        meta_group.attrs['timestamp'] = timestamp.encode('utf-8')
        meta_group.attrs['matrix_type'] = stats['matrix_type'].encode('utf-8')
        meta_group.attrs['analysis_type'] = stats['analysis_type'].encode('utf-8')
        meta_group.attrs['matrix_description'] = analyzer.matrix_description.encode('utf-8')
        meta_group.attrs['num_runs'] = num_runs
        
        # Training parameters
        training_group = f.create_group('training_parameters')
        training_group.attrs['learning_rate'] = learning_rate
        training_group.attrs['batch_size'] = batch_size
        training_group.attrs['num_epochs'] = num_epochs
        training_group.attrs['log_every_n_batches'] = log_every_n_batches
        training_group.attrs['parallel_runs'] = parallel_runs
        
        # Optimizer parameters
        training_group.attrs['optimizer'] = optimizer.encode('utf-8')
        training_group.attrs['momentum'] = momentum
        training_group.attrs['beta1'] = beta1
        training_group.attrs['beta2'] = beta2
        training_group.attrs['eps'] = eps
        training_group.attrs['weight_decay'] = weight_decay
        
        # Analysis parameters
        analysis_group = f.create_group('analysis_parameters')
        analysis_group.attrs['analyze_W'] = analyze_W
        analysis_group.attrs['analyze_delta_W'] = analyze_delta_W
        analysis_group.attrs['analyze_singular_values'] = analyze_singular_values
        analysis_group.attrs['disable_gradient_noise'] = disable_gradient_noise
        
        # System and experiment info
        system_group = f.create_group('system_info')
        system_group.attrs['experiment_dir_name'] = experiment_dir_name.encode('utf-8')
        system_group.attrs['pytorch_version'] = (torch.__version__ if 'torch' in globals() else 'unknown').encode('utf-8')
        system_group.attrs['device_available'] = ('cuda' if torch.cuda.is_available() else 'cpu').encode('utf-8')
        if torch.cuda.is_available():
            system_group.attrs['cuda_device_count'] = torch.cuda.device_count()
            system_group.attrs['cuda_device_name'] = torch.cuda.get_device_name(0).encode('utf-8')
        else:
            system_group.attrs['cuda_device_count'] = 0
            system_group.attrs['cuda_device_name'] = 'N/A'.encode('utf-8')
        
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
        
        # Save loss gradients if available
        if 'loss_gradients' in stats and len(stats['loss_gradients']) > 0:
            f.create_dataset('loss_gradients', data=np.array(stats['loss_gradients']))
        
        # Save gradient noise if available
        if 'gradient_noise' in stats and len(stats['gradient_noise']) > 0:
            f.create_dataset('gradient_noise', data=np.array(stats['gradient_noise']))
        
        # Save layer results
        layers_group = f.create_group('layers')
        for layer_name, layer_data in stats['results'].items():
            layer_group = layers_group.create_group(layer_name)
            
            # Save shape
            target_param = model.get_parameter(layer_name)
            # Save as a tuple representation directly instead of torch.Size string
            shape_tuple = tuple(target_param.shape)
            layer_group.attrs['shape'] = str(shape_tuple)
            
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
            
            # Save loss gradients for this layer if they exist
            if 'loss_gradients' in layer_data and layer_data['loss_gradients']:
                layer_group.create_dataset('loss_gradients', data=np.array(layer_data['loss_gradients']))
            
            # Save gradient noise for this layer if it exists
            if 'gradient_noise' in layer_data and layer_data['gradient_noise']:
                layer_group.create_dataset('gradient_noise', data=np.array(layer_data['gradient_noise']))
    
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
        'training_parameters': {},
        'analysis_parameters': {},
        'system_info': {},
        'model': {},
        'batch_numbers': [],
        'batches': [],
        'loss_values': [],
        'results': {}
    }
    
    with h5py.File(h5_path, 'r') as f:
        # Load metadata
        if 'metadata' in f:
            for key, value in f['metadata'].attrs.items():
                # Decode byte strings back to regular strings
                if isinstance(value, bytes):
                    data['metadata'][key] = value.decode('utf-8')
                else:
                    data['metadata'][key] = value
            # Default to 1 run if not found
            if 'num_runs' not in data['metadata']:
                data['metadata']['num_runs'] = 1
        
        # Load training parameters
        if 'training_parameters' in f:
            for key, value in f['training_parameters'].attrs.items():
                # Decode byte strings back to regular strings
                if isinstance(value, bytes):
                    data['training_parameters'][key] = value.decode('utf-8')
                else:
                    data['training_parameters'][key] = value
        else:
            # For backward compatibility, check if learning_rate is in metadata
            if 'learning_rate' in data['metadata']:
                data['training_parameters']['learning_rate'] = data['metadata']['learning_rate']
            # Set defaults for missing parameters
            data['training_parameters'].setdefault('learning_rate', 0.01)
            data['training_parameters'].setdefault('batch_size', 64)
            data['training_parameters'].setdefault('num_epochs', 1)
            data['training_parameters'].setdefault('log_every_n_batches', 200)
            data['training_parameters'].setdefault('parallel_runs', 1)
        
        # Set defaults for optimizer parameters (for backward compatibility)
        data['training_parameters'].setdefault('optimizer', 'sgd')
        data['training_parameters'].setdefault('momentum', 0.9)
        data['training_parameters'].setdefault('beta1', 0.9)
        data['training_parameters'].setdefault('beta2', 0.999)
        data['training_parameters'].setdefault('eps', 1e-8)
        data['training_parameters'].setdefault('weight_decay', 0.0)
        
        # Load analysis parameters
        if 'analysis_parameters' in f:
            for key, value in f['analysis_parameters'].attrs.items():
                # Decode byte strings back to regular strings
                if isinstance(value, bytes):
                    data['analysis_parameters'][key] = value.decode('utf-8')
                else:
                    data['analysis_parameters'][key] = value
        else:
            # Set defaults for missing parameters
            data['analysis_parameters'].setdefault('analyze_W', True)
            data['analysis_parameters'].setdefault('analyze_delta_W', False)
            data['analysis_parameters'].setdefault('analyze_singular_values', True)
            data['analysis_parameters'].setdefault('disable_gradient_noise', False)
        
        # Load system info
        if 'system_info' in f:
            for key, value in f['system_info'].attrs.items():
                # Decode byte strings back to regular strings
                if isinstance(value, bytes):
                    data['system_info'][key] = value.decode('utf-8')
                else:
                    data['system_info'][key] = value
        else:
            # Set defaults for missing parameters
            data['system_info'].setdefault('experiment_dir_name', 'experiments')
            data['system_info'].setdefault('pytorch_version', 'unknown')
            data['system_info'].setdefault('device_available', 'unknown')
            data['system_info'].setdefault('cuda_device_count', 0)
            data['system_info'].setdefault('cuda_device_name', 'N/A')
        
        # Load model info
        if 'model' in f:
            for key, value in f['model'].attrs.items():
                # Decode byte strings back to regular strings
                if isinstance(value, bytes):
                    data['model'][key] = value.decode('utf-8')
                else:
                    data['model'][key] = value
        
        # Load batch and loss data
        if 'batch_numbers' in f:
            data['batch_numbers'] = f['batch_numbers'][()]
        if 'batches' in f:
            data['batches'] = f['batches'][()]
        if 'loss_values' in f:
            data['loss_values'] = f['loss_values'][()]
        
        # Load loss gradients if available
        if 'loss_gradients' in f:
            data['loss_gradients'] = f['loss_gradients'][()]
        else:
            data['loss_gradients'] = []
        
        # Load gradient noise if available
        if 'gradient_noise' in f:
            data['gradient_noise'] = f['gradient_noise'][()]
        else:
            data['gradient_noise'] = []
        
        # Load layer results
        if 'layers' in f:
            for layer_name in f['layers']:
                layer_group = f['layers'][layer_name]
                
                # Safely parse the shape string without using eval()
                shape = None
                if 'shape' in layer_group.attrs:
                    shape_str = layer_group.attrs['shape']
                    # Handle torch.Size format or tuple format
                    if 'torch.Size' in shape_str:
                        # Extract numbers from torch.Size([dim1, dim2]) format
                        nums = re.findall(r'\d+', shape_str)
                        shape = tuple(int(num) for num in nums)
                    else:
                        # Should be a string representation of a tuple like "(dim1, dim2)"
                        try:
                            shape = eval(shape_str)  # Safe to eval a tuple string
                        except:
                            # Fallback to regex extraction if eval fails
                            nums = re.findall(r'\d+', shape_str)
                            shape = tuple(int(num) for num in nums)
                
                data['results'][layer_name] = {'shape': shape}
                
                # Load eigenvalues
                if 'eigenvalues' in layer_group:
                    eigen_group = layer_group['eigenvalues']
                    eigenvalues_list = []
                    for i in range(len(eigen_group)):
                        batch_key = f'batch_{i}'
                        if batch_key in eigen_group:
                            eigenvalues_list.append(eigen_group[batch_key][()])
                    data['results'][layer_name]['eigenvalues_list'] = eigenvalues_list
                    if eigenvalues_list:
                        data['results'][layer_name]['last_eigenvalues'] = eigenvalues_list[-1]
                
                # Load level spacing data
                if 'spacing' in layer_group:
                    spacing_group = layer_group['spacing']
                    if 'std_dev_list' in spacing_group:
                        data['results'][layer_name]['std_dev_norm_spacing_list'] = spacing_group['std_dev_list'][()]
                    if 'last_spacings' in spacing_group:
                        data['results'][layer_name]['last_normalized_spacings'] = spacing_group['last_spacings'][()]
                
                # Load singular values
                if 'singular_values' in layer_group:
                    sv_group = layer_group['singular_values']
                    sv_list = []
                    for i in range(len(sv_group)):
                        batch_key = f'batch_{i}'
                        if batch_key in sv_group:
                            sv_list.append(sv_group[batch_key][()])
                    data['results'][layer_name]['singular_values_list'] = sv_list
                    if sv_list:
                        data['results'][layer_name]['last_singular_values'] = sv_list[-1]
                
                # Load loss gradients for this layer if they exist
                if 'loss_gradients' in layer_group:
                    data['results'][layer_name]['loss_gradients'] = layer_group['loss_gradients'][()]
                else:
                    data['results'][layer_name]['loss_gradients'] = []
                
                # Load gradient noise for this layer if it exists
                if 'gradient_noise' in layer_group:
                    data['results'][layer_name]['gradient_noise'] = layer_group['gradient_noise'][()]
                else:
                    data['results'][layer_name]['gradient_noise'] = []
    
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