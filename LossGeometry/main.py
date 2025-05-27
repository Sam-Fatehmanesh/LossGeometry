import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
import warnings
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp
import h5py

from LossGeometry.models.mlp import SimpleMLP
from LossGeometry.datasets.mnist_dataset import load_mnist
from LossGeometry.analysis.spectral_analysis import SpectralAnalyzer
from LossGeometry.visualization.plot_utils import AnalysisPlotter
from LossGeometry.utils.io_utils import save_analysis_data, load_analysis_data, get_experiment_dir

def parse_args():
    """Parse command line arguments"""
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
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of training runs to average results over')
    parser.add_argument('--parallel_runs', type=int, default=1, 
                        help='Maximum number of runs to execute in parallel (default: 1)')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], 
                        help='Optimizer to use: sgd (SGD with momentum) or adam (Adam)')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='Momentum factor for SGD optimizer (default: 0.9)')
    parser.add_argument('--beta1', type=float, default=0.9, 
                        help='Beta1 parameter for Adam optimizer (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, 
                        help='Beta2 parameter for Adam optimizer (default: 0.999)')
    parser.add_argument('--eps', type=float, default=1e-8, 
                        help='Epsilon parameter for Adam optimizer (default: 1e-8)')
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='Weight decay (L2 penalty) for both optimizers (default: 0.0)')
    
    # Analysis parameters
    parser.add_argument('--analyze_W', action='store_true', help='Analyze weight matrices (default)')
    parser.add_argument('--analyze_delta_W', action='store_true', help='Analyze weight update matrices')
    parser.add_argument('--analyze_singular_values', action='store_true', help='Analyze singular values (default)')
    parser.add_argument('--disable_gradient_noise', action='store_true', help='Disable gradient noise calculation (faster training)')
    
    # Output parameters
    parser.add_argument('--experiment_dir', type=str, default='experiments', help='Base directory for experiments')
    
    # Replotting from existing H5 file
    parser.add_argument('--replot_h5', type=str, default=None, 
                        help='Path to an existing H5 file to load and replot (skips training)')
    
    args = parser.parse_args()
    
    # Set defaults for analysis types if none specified
    if not args.analyze_W and not args.analyze_delta_W:
        args.analyze_W = True
    if not args.analyze_singular_values:
        args.analyze_singular_values = True
    
    return args

# Encapsulate one training run to be executed in a worker process
def _run_training(run_config):
    """Execute a single training run
    
    Args:
        run_config (dict): Dictionary containing run configuration
        
    Returns:
        analyzer (SpectralAnalyzer): Analyzer containing results from this run
    """
    run_idx = run_config['run_idx']
    args = run_config['args']
    target_layers = run_config['target_layers']
    
    # For GPU allocation in multi-process environment
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{run_idx % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    print(f"Starting run {run_idx+1} on device {device}")
    
    # Initialize model fresh for this run
    model = SimpleMLP(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        num_hidden_layers=args.num_hidden_layers
    )
    model.to(device)
    
    # Setup optimizer and loss
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay
        )
        print(f"Using Adam optimizer with lr={args.learning_rate}, betas=({args.beta1}, {args.beta2}), eps={args.eps}, weight_decay={args.weight_decay}")
    else:  # SGD
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.learning_rate, 
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        print(f"Using SGD optimizer with lr={args.learning_rate}, momentum={args.momentum}, weight_decay={args.weight_decay}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Load MNIST dataset
    train_loader = load_mnist(batch_size=args.batch_size)
    
    # Initialize spectral analyzer for this run
    analyzer = SpectralAnalyzer(
        analyze_W=args.analyze_W,
        analyze_delta_W=args.analyze_delta_W,
        analyze_singular_values=args.analyze_singular_values
    )
    analyzer.initialize_layer_stats(target_layers)
    
    # Training loop
    current_batch = 0
    start_time = time.time()
    
    # Run training without nested progress bars in worker processes
    for epoch in range(args.num_epochs):
        epoch_loss_sum = 0.0
        num_batches_in_epoch = 0
        last_batch_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Standard training step
            model.train()
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Calculate gradients
            
            # Perform analysis
            if current_batch % args.log_every_n_batches == 0:
                # Only pass data_loader and criterion if gradient noise is enabled
                if args.disable_gradient_noise:
                    analyzer.analyze_batch(model, optimizer, current_batch, loss_tensor=loss)
                else:
                    analyzer.analyze_batch(model, optimizer, current_batch, loss_tensor=loss, 
                                         data_loader=train_loader, criterion=criterion,
                                         current_inputs=inputs, current_labels=labels)
            
            # Optimizer step
            optimizer.step()
            
            # Post-step
            current_batch += 1
            num_batches_in_epoch += 1
            epoch_loss_sum += loss.item()
            last_batch_loss = loss.item()
            
            # Track loss for plotting
            analyzer.track_loss(loss.item(), current_batch)
    
    total_duration = time.time() - start_time
    print(f"Run {run_idx+1} completed in {total_duration:.2f}s. Final loss: {last_batch_loss:.4f}")
    
    return analyzer

def train_and_analyze(args):
    """Train the model and perform spectral analysis"""
    # Ensure proper multiprocessing start method for CUDA
    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment directory
    experiment_dir = get_experiment_dir(args.experiment_dir)
    output_dir = os.path.join(experiment_dir, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created experiment directory: {output_dir}")
    
    # Setup device for reference model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get a reference model to determine target layers
    reference_model = SimpleMLP(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        num_hidden_layers=args.num_hidden_layers
    )
    
    # Get target layers for analysis
    target_layers = reference_model.get_target_layers()
    
    # Verify target layer dimensions
    layer_shapes = {}
    for target_layer_name in target_layers:
        try:
            target_param = reference_model.get_parameter(target_layer_name)
            layer_shapes[target_layer_name] = target_param.shape
            print(f"Layer for analysis: '{target_layer_name}' with size {target_param.shape[0]}x{target_param.shape[1]}")
        except AttributeError:
            print(f"WARNING: Target layer '{target_layer_name}' not found in the model.")
            
    # Dictionary to collect all runs' results
    all_runs_data = {
        'loss_values': [],
        'batch_numbers': [],
        'batches': []
    }
    
    # Initialize dictionaries to store accumulated singular values
    accumulated_singular_values = {}
    for layer_name in target_layers:
        accumulated_singular_values[layer_name] = {}
    
    # List to store analyzers from each run
    analyzers = []
    
    # Disable warnings during analysis
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Print analysis configuration
    print(f"\nStarting training on {device} for {args.num_epochs} epochs.")
    print(f"Analyzing layers with {args.hidden_size} hidden units, {args.num_hidden_layers} hidden layers")
    print(f"Calculation Frequency: Every {args.log_every_n_batches} batches.")
    
    matrix_type = "Weight (W)" if args.analyze_W else "Weight Update (ΔW)"
    analysis_types = []
    if args.analyze_singular_values:
        analysis_types.append("Singular Values")
    
    analysis_desc = ", ".join(analysis_types)
    print(f"Analyzing: {matrix_type} matrices for {analysis_desc}")
    
    # Determine whether to run in parallel or sequentially
    if args.num_runs > 1:
        print(f"Performing {args.num_runs} runs, with up to {min(args.parallel_runs, args.num_runs)} in parallel.")
        
        # Use parallel execution if requested
        if args.parallel_runs > 1:
            max_workers = min(args.parallel_runs, args.num_runs)
            print(f"Using {max_workers} parallel workers")
            
            # Create run configurations for each process
            run_configs = [
                {
                    'run_idx': run_idx,
                    'args': args,
                    'target_layers': target_layers
                } 
                for run_idx in range(args.num_runs)
            ]
            
            # Create and update progress bar
            progress_bar = tqdm(total=args.num_runs, desc="Training Runs", position=0, 
                               leave=True, colour='blue', ncols=100)
            
            # Execute runs in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_run_idx = {
                    executor.submit(_run_training, config): config['run_idx'] 
                    for config in run_configs
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_run_idx):
                    run_idx = future_to_run_idx[future]
                    try:
                        analyzer = future.result()
                        analyzers.append(analyzer)
                        
                        # Update progress
                        progress_bar.update(1)
                        progress_bar.set_postfix({"run": f"{run_idx+1}/{args.num_runs}"})
                        
                    except Exception as exc:
                        print(f'Run {run_idx+1} generated an exception: {exc}')
                        
            # Close progress bar
            progress_bar.close()
            
        else:
            # Fallback to sequential runs if parallel_runs=1
            run_loop = tqdm(range(args.num_runs), desc="Training Runs", position=0, 
                         leave=True, colour='blue', ncols=100)
            
            for run_idx in run_loop:
                run_loop.set_description(f"Run {run_idx+1}/{args.num_runs}")
                
                # Run training sequentially using the same function
                run_config = {
                    'run_idx': run_idx,
                    'args': args,
                    'target_layers': target_layers
                }
                analyzer = _run_training(run_config)
                analyzers.append(analyzer)
    else:
        # Just one run
        run_config = {
            'run_idx': 0,
            'args': args,
            'target_layers': target_layers
        }
        analyzer = _run_training(run_config)
        analyzers.append(analyzer)
    
    # Collect all runs' results
    for i, analyzer in enumerate(analyzers):
        all_runs_data['loss_values'].append(analyzer.stats['loss_values'])
        all_runs_data['batch_numbers'].append(analyzer.stats['batch_numbers'])
        all_runs_data['batches'].append(analyzer.stats['batch'])
        
        # Collect singular values
        for layer_name in target_layers:
            if (layer_name in analyzer.stats['results'] and 
                'singular_values_list' in analyzer.stats['results'][layer_name]):
                sv_list = analyzer.stats['results'][layer_name]['singular_values_list']
                for j, sv in enumerate(sv_list):
                    batch_key = analyzer.stats['batch'][j] if j < len(analyzer.stats['batch']) else j
                    if batch_key not in accumulated_singular_values[layer_name]:
                        accumulated_singular_values[layer_name][batch_key] = []
                    accumulated_singular_values[layer_name][batch_key].append(sv)
    
    # Re-enable warnings after analysis
    warnings.filterwarnings("default", category=RuntimeWarning)
    
    # Create a new analyzer for the aggregated results
    aggregated_analyzer = SpectralAnalyzer(
        analyze_W=args.analyze_W,
        analyze_delta_W=args.analyze_delta_W,
        analyze_singular_values=args.analyze_singular_values
    )
    aggregated_analyzer.initialize_layer_stats(target_layers)
    
    # Compute average loss
    if args.num_runs > 1:
        print("Aggregating results from all runs...")
        # For simplicity, use the batches from the first run
        aggregated_analyzer.stats['batch'] = analyzers[0].stats['batch']
        aggregated_analyzer.stats['batch_numbers'] = analyzers[0].stats['batch_numbers']
        
        # Average losses across runs
        avg_losses = np.mean([run_losses for run_losses in all_runs_data['loss_values']], axis=0)
        aggregated_analyzer.stats['loss_values'] = avg_losses.tolist()
        
        # Average loss gradients across runs if available
        all_loss_gradients = []
        for analyzer in analyzers:
            if 'loss_gradients' in analyzer.stats and len(analyzer.stats['loss_gradients']) > 0:
                all_loss_gradients.append(analyzer.stats['loss_gradients'])
        
        if all_loss_gradients:
            avg_loss_gradients = np.mean(all_loss_gradients, axis=0)
            aggregated_analyzer.stats['loss_gradients'] = avg_loss_gradients.tolist()
        
        # Process singular values
        for layer_name in target_layers:
            # Create empty result structure for this layer
            if layer_name not in aggregated_analyzer.stats['results']:
                aggregated_analyzer.stats['results'][layer_name] = {}
            
            # Average loss gradients for this layer across runs
            layer_loss_gradients = []
            layer_gradient_noise = []
            for analyzer in analyzers:
                if (layer_name in analyzer.stats['results'] and 
                    'loss_gradients' in analyzer.stats['results'][layer_name] and
                    len(analyzer.stats['results'][layer_name]['loss_gradients']) > 0):
                    layer_loss_gradients.append(analyzer.stats['results'][layer_name]['loss_gradients'])
                
                # Also collect gradient noise if available
                if (layer_name in analyzer.stats['results'] and 
                    'gradient_noise' in analyzer.stats['results'][layer_name] and
                    len(analyzer.stats['results'][layer_name]['gradient_noise']) > 0):
                    layer_gradient_noise.append(analyzer.stats['results'][layer_name]['gradient_noise'])
            
            if layer_loss_gradients:
                avg_layer_gradients = np.mean(layer_loss_gradients, axis=0)
                aggregated_analyzer.stats['results'][layer_name]['loss_gradients'] = avg_layer_gradients.tolist()
            else:
                aggregated_analyzer.stats['results'][layer_name]['loss_gradients'] = []
            
            if layer_gradient_noise:
                avg_layer_noise = np.mean(layer_gradient_noise, axis=0)
                aggregated_analyzer.stats['results'][layer_name]['gradient_noise'] = avg_layer_noise.tolist()
            else:
                aggregated_analyzer.stats['results'][layer_name]['gradient_noise'] = []
            
            # Average singular values
            if 'singular_values_list' in analyzer.stats['results'][layer_name]:
                sv_list = []
                for batch in sorted(accumulated_singular_values[layer_name].keys()):
                    if accumulated_singular_values[layer_name][batch]:
                        # For each batch, get the singular values from each run
                        batch_sv = accumulated_singular_values[layer_name][batch]
                        
                        if batch_sv and len(batch_sv) > 0:
                            # Flatten all singular values from all runs at this batch
                            all_sv = np.concatenate(batch_sv)
                            sv_list.append(all_sv)
                
                aggregated_analyzer.stats['results'][layer_name]['singular_values_list'] = sv_list
                if sv_list:
                    aggregated_analyzer.stats['results'][layer_name]['last_singular_values'] = sv_list[-1]
    else:
        # If only one run, use the first analyzer's data directly
        aggregated_analyzer = analyzers[0]
    
    # Save the aggregated data
    h5_path = save_analysis_data(
        aggregated_analyzer, reference_model, experiment_dir, timestamp, 
        num_runs=args.num_runs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        log_every_n_batches=args.log_every_n_batches,
        parallel_runs=args.parallel_runs,
        analyze_W=args.analyze_W,
        analyze_delta_W=args.analyze_delta_W,
        analyze_singular_values=args.analyze_singular_values,
        disable_gradient_noise=args.disable_gradient_noise,
        experiment_dir_name=args.experiment_dir,
        optimizer=args.optimizer,
        momentum=args.momentum,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    # Generate plots
    print("\nGenerating plots...")
    plotter = AnalysisPlotter(output_dir, timestamp)
    
    # Create metadata summary from the saved data
    saved_data = load_analysis_data(h5_path)
    plotter.create_metadata_summary(saved_data)
    
    # Plot loss
    if args.num_runs > 1:
        plotter.plot_loss(aggregated_analyzer.stats['batch_numbers'], 
                          aggregated_analyzer.stats['loss_values'],
                          plot_title=f"MNIST Training Loss (Averaged over {args.num_runs} runs)")
    else:
        plotter.plot_loss(aggregated_analyzer.stats['batch_numbers'], 
                          aggregated_analyzer.stats['loss_values'])
    
    # Plot analysis results for each layer
    for layer_name in target_layers:
        layer_shape = layer_shapes.get(layer_name, None)
        layer_results = aggregated_analyzer.stats['results'][layer_name]
        
        # Plot singular values if analyzed
        if 'singular_values_list' in layer_results:
            plotter.plot_singular_values(
                layer_name, 
                layer_shape, 
                layer_results['singular_values_list'], 
                aggregated_analyzer.stats['batch'], 
                aggregated_analyzer.matrix_description,
                runs=args.num_runs,
                learning_rate=args.learning_rate
            )
            
            # Also create overlay plots showing evolution across batches (if enabled)
            if not getattr(args, 'no_overlay_plots', False):
                plotter.plot_singular_values_overlay(
                    layer_name, 
                    layer_shape, 
                    layer_results['singular_values_list'], 
                    aggregated_analyzer.stats['batch'], 
                    aggregated_analyzer.matrix_description,
                    runs=args.num_runs,
                    learning_rate=args.learning_rate
                )
            
            # Plot singular value dynamics terms if we have the necessary data
            if ('loss_gradients' in layer_results and 
                len(layer_results['loss_gradients']) > 0 and
                len(layer_results['singular_values_list']) > 0):
                
                # Get gradient noise data if available
                gradient_noise = layer_results.get('gradient_noise', None)
                
                plotter.plot_singular_value_dynamics_terms(
                    layer_name,
                    layer_shape,
                    layer_results['singular_values_list'],
                    layer_results['loss_gradients'],
                    aggregated_analyzer.stats['batch'],
                    aggregated_analyzer.matrix_description,
                    runs=args.num_runs,
                    gradient_noise=gradient_noise
            )
    
    print("\nAnalysis complete. Results saved to:", output_dir)
    return output_dir, h5_path

def replot_from_h5(h5_path, experiment_dir):
    """
    Load data from an existing H5 file and regenerate plots
    
    Args:
        h5_path (str): Path to the H5 file to load
        experiment_dir (str): Base directory for experiments
        
    Returns:
        tuple: (output_dir, h5_path)
    """
    print(f"Loading data from {h5_path} to regenerate plots...")
    
    # Generate a new timestamp for the replotting output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(experiment_dir, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created new output directory: {output_dir}")
    
    # Load data from H5 file
    data = load_analysis_data(h5_path)
    
    # Create a reference model to get layer shapes (needed for some plots)
    reference_model = SimpleMLP(
        input_size=data['model'].get('input_size', 784),
        hidden_size=data['model'].get('hidden_size', 1024),
        output_size=data['model'].get('output_size', 10),
        num_hidden_layers=data['model'].get('num_hidden_layers', 2)
    )
    
    # Get target layers and their shapes
    target_layers = [layer_name for layer_name in data['results'].keys()]
    layer_shapes = {}
    for layer_name in target_layers:
        # Try to get shape from the data, or from the reference model
        layer_shapes[layer_name] = data['results'][layer_name].get('shape')
        if not layer_shapes[layer_name]:
            try:
                target_param = reference_model.get_parameter(layer_name)
                layer_shapes[layer_name] = target_param.shape
            except AttributeError:
                print(f"WARNING: Unable to determine shape for layer '{layer_name}'")
                layer_shapes[layer_name] = None
    
    # Save a new copy of the H5 file with the new timestamp
    new_h5_path = os.path.join(output_dir, f"{timestamp}_analysis_data.h5")
    with h5py.File(h5_path, 'r') as src:
        with h5py.File(new_h5_path, 'w') as dst:
            # Copy all groups and datasets
            for key in src.keys():
                src.copy(src[key], dst, key)
            
            # Update the timestamp in metadata
            if 'metadata' in dst:
                dst['metadata'].attrs['timestamp'] = timestamp
                dst['metadata'].attrs['replotted_from'] = os.path.basename(h5_path)
    
    print(f"Created new H5 file at {new_h5_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    plotter = AnalysisPlotter(output_dir, timestamp)
    
    # Create metadata summary
    plotter.create_metadata_summary(data)
    
    # Get num_runs
    num_runs = data['metadata'].get('num_runs', 1)
    
    # Plot loss
    if len(data['loss_values']) > 0:
        if num_runs > 1:
            plotter.plot_loss(data['batch_numbers'], 
                              data['loss_values'],
                              plot_title=f"MNIST Training Loss (Averaged over {num_runs} runs)")
        else:
            plotter.plot_loss(data['batch_numbers'], 
                              data['loss_values'])
    
    # Plot analysis results for each layer
    for layer_name in target_layers:
        layer_shape = layer_shapes.get(layer_name)
        layer_results = data['results'][layer_name]
        
        # Matrix description from metadata
        matrix_description = data['metadata'].get('matrix_description', 
                                                 "Centered & Normalized Weight (W' / (std(W') * sqrt(max(N))))")
        
        # Plot singular values if analyzed
        if 'singular_values_list' in layer_results and layer_results['singular_values_list']:
            # Get learning rate from saved data, default to 0.01 if not available
            learning_rate = data['training_parameters'].get('learning_rate', 0.01)
            
            plotter.plot_singular_values(
                layer_name, 
                layer_shape, 
                layer_results['singular_values_list'], 
                data['batches'], 
                matrix_description,
                runs=num_runs,
                learning_rate=learning_rate
            )
            
            # Also create overlay plots showing evolution across batches
            plotter.plot_singular_values_overlay(
                layer_name, 
                layer_shape, 
                layer_results['singular_values_list'], 
                data['batches'], 
                matrix_description,
                runs=num_runs,
                learning_rate=learning_rate
            )
            
            # Plot singular value dynamics terms if we have the necessary data
            if ('loss_gradients' in layer_results and 
                len(layer_results['loss_gradients']) > 0 and
                len(layer_results['singular_values_list']) > 0):
                
                # Get gradient noise data if available
                gradient_noise = layer_results.get('gradient_noise', None)
                
                plotter.plot_singular_value_dynamics_terms(
                    layer_name,
                    layer_shape,
                    layer_results['singular_values_list'],
                    layer_results['loss_gradients'],
                    data['batches'],
                    matrix_description,
                    runs=num_runs,
                    gradient_noise=gradient_noise
            )
    
    print("\nReplotting complete. New plots saved to:", output_dir)
    return output_dir, new_h5_path

def main():
    """Main entry point for running the analysis directly"""
    args = parse_args()
    
    # Check if we're replotting from an existing H5 file
    if args.replot_h5:
        if not os.path.exists(args.replot_h5):
            print(f"ERROR: H5 file not found: {args.replot_h5}")
            return 1
        output_dir, h5_path = replot_from_h5(args.replot_h5, args.experiment_dir)
    else:
        # Run normal training and analysis
        output_dir, h5_path = train_and_analyze(args)
    
    print(f"Analysis complete. Results saved to: {output_dir}")
    print(f"Data saved to: {h5_path}")
    return 0  # Success exit code

if __name__ == "__main__":
    main() 