import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
import warnings

from LossGeometry.models.mlp import SimpleMLP
from LossGeometry.datasets.mnist_dataset import load_mnist
from LossGeometry.analysis.spectral_analysis import SpectralAnalyzer
from LossGeometry.visualization.plot_utils import AnalysisPlotter
from LossGeometry.utils.io_utils import save_analysis_data, get_experiment_dir

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
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    # Analysis parameters
    parser.add_argument('--analyze_W', action='store_true', help='Analyze weight matrices (default)')
    parser.add_argument('--analyze_delta_W', action='store_true', help='Analyze weight update matrices')
    parser.add_argument('--analyze_spectral_density', action='store_true', help='Analyze spectral density (default)')
    parser.add_argument('--analyze_level_spacing', action='store_true', help='Analyze level spacing')
    parser.add_argument('--analyze_singular_values', action='store_true', help='Analyze singular values')
    
    # Output parameters
    parser.add_argument('--experiment_dir', type=str, default='experiments', help='Base directory for experiments')
    
    args = parser.parse_args()
    
    # Set defaults for analysis types if none specified
    if not args.analyze_W and not args.analyze_delta_W:
        args.analyze_W = True
    if not args.analyze_spectral_density and not args.analyze_level_spacing and not args.analyze_singular_values:
        args.analyze_spectral_density = True
        args.analyze_singular_values = True
    
    return args

def train_and_analyze(args):
    """Train the model and perform spectral analysis"""
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment directory
    experiment_dir = get_experiment_dir(args.experiment_dir)
    output_dir = os.path.join(experiment_dir, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created experiment directory: {output_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = SimpleMLP(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        num_hidden_layers=args.num_hidden_layers
    )
    model.to(device)
    
    # Get target layers for analysis
    target_layers = model.get_target_layers()
    
    # Verify target layer dimensions
    layer_shapes = {}
    for target_layer_name in target_layers:
        try:
            target_param = model.get_parameter(target_layer_name)
            layer_shapes[target_layer_name] = target_param.shape
            print(f"Layer for analysis: '{target_layer_name}' with size {target_param.shape[0]}x{target_param.shape[1]}")
        except AttributeError:
            print(f"WARNING: Target layer '{target_layer_name}' not found in the model.")
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Load MNIST dataset
    train_loader = load_mnist(batch_size=args.batch_size)
    
    # Initialize spectral analyzer
    analyzer = SpectralAnalyzer(
        analyze_W=args.analyze_W,
        analyze_delta_W=args.analyze_delta_W,
        analyze_spectral_density=args.analyze_spectral_density,
        analyze_level_spacing=args.analyze_level_spacing,
        analyze_singular_values=args.analyze_singular_values
    )
    analyzer.initialize_layer_stats(target_layers)
    
    # Print analysis configuration
    print(f"\nStarting training on {device} for {args.num_epochs} epochs.")
    print(f"Analyzing: {analyzer.stats['analysis_type']} of {analyzer.matrix_description}")
    if args.analyze_singular_values:
        print(f"Additionally analyzing: Singular Value Distributions")
    print(f"Calculation Frequency: Every {args.log_every_n_batches} batches.")
    
    # Disable warnings during analysis
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Training loop
    current_batch = 0
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
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
                calc_start_time = time.time()
                print(f"\n--- Analyzing Batch {current_batch} (Epoch {epoch+1}) ---")
                
                analyzer.analyze_batch(model, optimizer, current_batch)
                
                calc_duration = time.time() - calc_start_time
            
            # Optimizer step
            optimizer.step()
            
            # Post-step
            current_batch += 1
            num_batches_in_epoch += 1
            epoch_loss_sum += loss.item()
            last_batch_loss = loss.item()
            
            # Track loss for plotting
            analyzer.track_loss(loss.item(), current_batch)
        
        # End of epoch
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        print(f"\nEpoch {epoch+1}/{args.num_epochs} completed in {epoch_duration:.2f}s. Avg Loss: {avg_epoch_loss:.4f}. (Last batch loss: {last_batch_loss:.4f})")
    
    total_duration = time.time() - start_time
    print(f"\nTraining finished in {total_duration:.2f}s")
    
    # Save the data
    h5_path = save_analysis_data(analyzer, model, experiment_dir, timestamp)
    
    # Generate plots
    print("\nGenerating plots...")
    plotter = AnalysisPlotter(output_dir, timestamp)
    
    # Plot loss
    plotter.plot_loss(analyzer.stats['batch_numbers'], analyzer.stats['loss_values'])
    
    # Plot analysis results for each layer
    for layer_name in target_layers:
        layer_shape = layer_shapes.get(layer_name, None)
        layer_results = analyzer.stats['results'][layer_name]
        
        # Plot spectral density if analyzed
        if args.analyze_spectral_density and 'eigenvalues_list' in layer_results:
            plotter.plot_spectral_density(
                layer_name, 
                layer_shape, 
                layer_results['eigenvalues_list'], 
                analyzer.stats['batch'], 
                analyzer.matrix_description
            )
        
        # Plot level spacing if analyzed
        if args.analyze_level_spacing and 'std_dev_norm_spacing_list' in layer_results:
            plotter.plot_level_spacing(
                layer_name, 
                layer_shape, 
                layer_results['std_dev_norm_spacing_list'], 
                layer_results.get('last_normalized_spacings', None), 
                analyzer.stats['batch'], 
                analyzer.matrix_description
            )
        
        # Plot singular values if analyzed
        if args.analyze_singular_values and 'singular_values_list' in layer_results:
            plotter.plot_singular_values(
                layer_name, 
                layer_shape, 
                layer_results['singular_values_list'], 
                analyzer.stats['batch'], 
                analyzer.matrix_description
            )
    
    # Re-enable warnings after analysis
    warnings.filterwarnings("default", category=RuntimeWarning)
    
    print("\nAnalysis complete. Results saved to:", output_dir)
    return output_dir, h5_path

def main():
    """Main entry point for running the analysis directly"""
    args = parse_args()
    output_dir, h5_path = train_and_analyze(args)
    print(f"Analysis complete. Results saved to: {output_dir}")
    print(f"Data saved to: {h5_path}")
    return 0  # Success exit code

if __name__ == "__main__":
    main() 