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

from LossGeometry.models.mlp import SimpleMLP
from LossGeometry.datasets.mnist_dataset import load_mnist
from LossGeometry.datasets.cifar100_dataset import load_cifar100
from LossGeometry.analysis.spectral_analysis import SpectralAnalyzer
from LossGeometry.visualization.plot_utils import AnalysisPlotter
from LossGeometry.utils.io_utils import save_analysis_data, get_experiment_dir
from LossGeometry.models.resnet import CustomResNet18
from LossGeometry.models.vit import CustomViT

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
    parser.add_argument('--num_runs', type=int, default=1, help='Number of training runs to average results over')
    
    # Analysis parameters
    parser.add_argument('--analyze_W', action='store_true', help='Analyze weight matrices (default)')
    parser.add_argument('--analyze_delta_W', action='store_true', help='Analyze weight update matrices')
    parser.add_argument('--analyze_spectral_density', action='store_true', help='Analyze spectral density (default)')
    parser.add_argument('--analyze_level_spacing', action='store_true', help='Analyze level spacing')
    parser.add_argument('--analyze_singular_values', action='store_true', help='Analyze singular values')
    
    # Output parameters
    parser.add_argument('--experiment_dir', type=str, default='experiments', help='Base directory for experiments')
    parser.add_argument('--experiment_name', type=str, default=None, help='Name of the experiment subdirectory')
    # Model selection
    parser.add_argument('--model', type=str, choices=['mlp','resnet18','vit'], default='mlp',
                        help='Which model to use: mlp, resnet18, or vit')
    # ResNet-specific parameters
    parser.add_argument('--resnet_input_channels', type=int, default=1,
                        help='Number of input channels for ResNet')
    parser.add_argument('--resnet_init_conv', action='store_true',
                        help='Apply Gaussian init to ResNet convolutional layers')
    # ViT-specific parameters
    parser.add_argument('--vit_image_size', type=int, default=28, help='Image size for ViT patches')
    parser.add_argument('--vit_patch_size', type=int, default=7, help='Patch size for ViT')
    parser.add_argument('--vit_embed_dim', type=int, default=64, help='Embedding dimension for ViT')
    parser.add_argument('--vit_depth', type=int, default=2, help='Number of Transformer encoder layers for ViT')
    parser.add_argument('--vit_num_heads', type=int, default=4, help='Number of attention heads in ViT')
    parser.add_argument('--vit_mlp_ratio', type=float, default=2.0, help='Ratio of MLP hidden dim to embed_dim in ViT')
    parser.add_argument('--vit_input_channels', type=int, default=1, help='Number of input channels for ViT')
    parser.add_argument('--vit_init_fc', action='store_true', help='Gaussian init for ViT classification head')
    # Dataset selection
    parser.add_argument('--dataset', type=str, choices=['mnist','cifar100'], default='mnist',
                        help='Which dataset to use for training')
    
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
    
    # Determine experiment directory
    base_experiment_dir = get_experiment_dir(args.experiment_dir)
    if args.experiment_name:
        base_experiment_dir = os.path.join(base_experiment_dir, args.experiment_name)
        os.makedirs(base_experiment_dir, exist_ok=True)
        print(f"Created experiment subdirectory: {base_experiment_dir}")
    experiment_dir = base_experiment_dir
    # Create timestamped directory
    output_dir = os.path.join(experiment_dir, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created experiment directory: {output_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get a reference model to determine target layers
    if args.model == 'mlp':
        reference_model = SimpleMLP(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            num_hidden_layers=args.num_hidden_layers
        )
    elif args.model == 'resnet18':
        reference_model = CustomResNet18(
            num_classes=args.output_size,
            input_channels=args.resnet_input_channels,
            gaussian_init_fc=True,
            gaussian_init_conv=args.resnet_init_conv
        )
    elif args.model == 'vit':
        reference_model = CustomViT(
            num_classes=args.output_size,
            image_size=args.vit_image_size,
            patch_size=args.vit_patch_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_num_heads,
            mlp_ratio=args.vit_mlp_ratio,
            input_channels=args.vit_input_channels,
            gaussian_init_fc=args.vit_init_fc
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
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
    
    # Initialize dictionaries to store accumulated eigenvalues and singular values
    accumulated_eigenvalues = {}
    accumulated_singular_values = {}
    for layer_name in target_layers:
        accumulated_eigenvalues[layer_name] = {}
        accumulated_singular_values[layer_name] = {}
    
    # List to store analyzers from each run
    analyzers = []
    
    # Disable warnings during analysis
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Print analysis configuration
    print(f"\nStarting training on {device} for {args.num_epochs} epochs.")
    print(f"Analyzing layers with {args.hidden_size} hidden units, {args.num_hidden_layers} hidden layers")
    print(f"Calculation Frequency: Every {args.log_every_n_batches} batches.")
    if args.num_runs > 1:
        print(f"Performing {args.num_runs} runs and averaging results.")
    
    # Run training N times
    run_loop = tqdm(range(args.num_runs), desc="Training Runs", position=0, 
                    leave=True, colour='blue', ncols=100)
    
    for run_idx in run_loop:
        run_loop.set_description(f"Run {run_idx+1}/{args.num_runs}")
        
        # Initialize model fresh for this run
        if args.model == 'mlp':
            model = SimpleMLP(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                output_size=args.output_size,
                num_hidden_layers=args.num_hidden_layers
            )
        elif args.model == 'resnet18':
            model = CustomResNet18(
                num_classes=args.output_size,
                input_channels=args.resnet_input_channels,
                gaussian_init_fc=True,
                gaussian_init_conv=args.resnet_init_conv
            )
        elif args.model == 'vit':
            model = CustomViT(
                num_classes=args.output_size,
                image_size=args.vit_image_size,
                patch_size=args.vit_patch_size,
                embed_dim=args.vit_embed_dim,
                depth=args.vit_depth,
                num_heads=args.vit_num_heads,
                mlp_ratio=args.vit_mlp_ratio,
                input_channels=args.vit_input_channels,
                gaussian_init_fc=args.vit_init_fc
            )
        else:
            raise ValueError(f"Unknown model type: {args.model}")
        model.to(device)
        
        # Setup optimizer and loss
        #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Load dataset
        if args.dataset == 'mnist':
            train_loader = load_mnist(batch_size=args.batch_size)
        elif args.dataset == 'cifar100':
            train_loader = load_cifar100(batch_size=args.batch_size)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        # Initialize spectral analyzer for this run
        analyzer = SpectralAnalyzer(
            analyze_W=args.analyze_W,
            analyze_delta_W=args.analyze_delta_W,
            analyze_spectral_density=args.analyze_spectral_density,
            analyze_level_spacing=args.analyze_level_spacing,
            analyze_singular_values=args.analyze_singular_values
        )
        analyzer.initialize_layer_stats(target_layers)
        analyzers.append(analyzer)
        
        # Print analysis configuration (only for first run)
        if run_idx == 0:
            matrix_type = "Weight (W)" if args.analyze_W else "Weight Update (ΔW)"
            analysis_types = []
            if args.analyze_spectral_density:
                analysis_types.append("Spectral Density")
            if args.analyze_level_spacing:
                analysis_types.append("Level Spacing")
            if args.analyze_singular_values:
                analysis_types.append("Singular Values")
            
            analysis_desc = ", ".join(analysis_types)
            run_loop.write(f"Analyzing: {matrix_type} matrices for {analysis_desc}")
        
        # Training loop
        current_batch = 0
        start_time = time.time()
        
        # Create progress bars for epochs
        epoch_loop = tqdm(range(args.num_epochs), desc=f"Run {run_idx+1} Epochs", 
                          position=1, leave=False, colour='green', ncols=100)
        
        for epoch in epoch_loop:
            epoch_loss_sum = 0.0
            num_batches_in_epoch = 0
            last_batch_loss = 0.0
            
            # Estimate total batches for progress bar
            estimated_batches = len(train_loader)
            
            # Create progress bar for batches
            batch_loop = tqdm(enumerate(train_loader), desc=f"Run {run_idx+1} Epoch {epoch+1}", 
                              total=estimated_batches, position=2, leave=False, 
                              colour='yellow', ncols=100)
            
            for batch_idx, (inputs, labels) in batch_loop:
                # Standard training step
                model.train()
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()  # Calculate gradients
                
                # Perform analysis
                if current_batch % args.log_every_n_batches == 0:
                    batch_loop.set_postfix({"analyzing": True, "loss": loss.item()})
                    analyzer.analyze_batch(model, optimizer, current_batch)
                else:
                    batch_loop.set_postfix({"loss": loss.item()})
                
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
            avg_epoch_loss = epoch_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
            epoch_loop.set_postfix({"avg_loss": f"{avg_epoch_loss:.4f}"})
            
        total_duration = time.time() - start_time
        run_loop.write(f"Run {run_idx+1} completed in {total_duration:.2f}s. Final loss: {last_batch_loss:.4f}")
        
        # Collect this run's results
        all_runs_data['loss_values'].append(analyzer.stats['loss_values'])
        all_runs_data['batch_numbers'].append(analyzer.stats['batch_numbers'])
        all_runs_data['batches'].append(analyzer.stats['batch'])
        
        # Collect eigenvalues and singular values
        for layer_name in target_layers:
            # For eigenvalues
            if args.analyze_spectral_density and 'eigenvalues_list' in analyzer.stats['results'][layer_name]:
                eigenvalues_list = analyzer.stats['results'][layer_name]['eigenvalues_list']
                for i, eigenvalues in enumerate(eigenvalues_list):
                    batch_key = analyzer.stats['batch'][i] if i < len(analyzer.stats['batch']) else i
                    if batch_key not in accumulated_eigenvalues[layer_name]:
                        accumulated_eigenvalues[layer_name][batch_key] = []
                    accumulated_eigenvalues[layer_name][batch_key].append(eigenvalues)
            
            # For singular values
            if args.analyze_singular_values and 'singular_values_list' in analyzer.stats['results'][layer_name]:
                sv_list = analyzer.stats['results'][layer_name]['singular_values_list']
                for i, sv in enumerate(sv_list):
                    batch_key = analyzer.stats['batch'][i] if i < len(analyzer.stats['batch']) else i
                    if batch_key not in accumulated_singular_values[layer_name]:
                        accumulated_singular_values[layer_name][batch_key] = []
                    accumulated_singular_values[layer_name][batch_key].append(sv)
    
    # Re-enable warnings after analysis
    warnings.filterwarnings("default", category=RuntimeWarning)
    
    # Create a new analyzer for the aggregated results
    aggregated_analyzer = SpectralAnalyzer(
        analyze_W=args.analyze_W,
        analyze_delta_W=args.analyze_delta_W,
        analyze_spectral_density=args.analyze_spectral_density,
        analyze_level_spacing=args.analyze_level_spacing,
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
        
        # Process eigenvalues and singular values
        for layer_name in target_layers:
            # Create empty result structure for this layer
            if layer_name not in aggregated_analyzer.stats['results']:
                aggregated_analyzer.stats['results'][layer_name] = {}
            
            # Average eigenvalues
            if args.analyze_spectral_density:
                eigenvalues_list = []
                for batch in sorted(accumulated_eigenvalues[layer_name].keys()):
                    if accumulated_eigenvalues[layer_name][batch]:
                        # For each batch, get the eigenvalues from each run
                        batch_eigenvalues = accumulated_eigenvalues[layer_name][batch]
                        
                        # Convert to histograms and average the distributions
                        # This is a simplified approach - more sophisticated approaches could be used
                        if batch_eigenvalues and len(batch_eigenvalues) > 0:
                            # Flatten all eigenvalues from all runs at this batch
                            all_eigenvalues = np.concatenate(batch_eigenvalues)
                            eigenvalues_list.append(all_eigenvalues)
                
                aggregated_analyzer.stats['results'][layer_name]['eigenvalues_list'] = eigenvalues_list
                if eigenvalues_list:
                    aggregated_analyzer.stats['results'][layer_name]['last_eigenvalues'] = eigenvalues_list[-1]
            
            # Average singular values
            if args.analyze_singular_values:
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
    h5_path = save_analysis_data(aggregated_analyzer, reference_model, experiment_dir, timestamp, num_runs=args.num_runs)
    
    # Generate plots
    print("\nGenerating plots...")
    plotter = AnalysisPlotter(output_dir, timestamp)
    
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
        
        # Plot spectral density if analyzed
        if args.analyze_spectral_density and 'eigenvalues_list' in layer_results:
            plotter.plot_spectral_density(
                layer_name, 
                layer_shape, 
                layer_results['eigenvalues_list'], 
                aggregated_analyzer.stats['batch'], 
                aggregated_analyzer.matrix_description,
                runs=args.num_runs
            )
        
        # Plot level spacing if analyzed
        if args.analyze_level_spacing and 'std_dev_norm_spacing_list' in layer_results:
            plotter.plot_level_spacing(
                layer_name, 
                layer_shape, 
                layer_results['std_dev_norm_spacing_list'], 
                layer_results.get('last_normalized_spacings', None), 
                aggregated_analyzer.stats['batch'], 
                aggregated_analyzer.matrix_description,
                runs=args.num_runs
            )
        
        # Plot singular values if analyzed
        if args.analyze_singular_values and 'singular_values_list' in layer_results:
            plotter.plot_singular_values(
                layer_name, 
                layer_shape, 
                layer_results['singular_values_list'], 
                aggregated_analyzer.stats['batch'], 
                aggregated_analyzer.matrix_description,
                runs=args.num_runs,
                num_epochs=args.num_epochs
            )
    
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