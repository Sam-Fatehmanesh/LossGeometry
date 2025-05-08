# -*- coding: utf-8 -*-
# CIFAR-10 Spectral Analysis
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import warnings
import math
import os
from datetime import datetime

# --- Noise Injection Functions ---
def inject_no_noise(param):
    """No noise injection"""
    return

def inject_standard_noise(param, magnitude):
    """Inject standard Gaussian noise"""
    # Calculate standard deviation of the weights
    weight_std = torch.std(param.data).item()
    # Scale noise by 0.4 * standard deviation of weights
    scaled_magnitude = magnitude * weight_std
    noise = scaled_magnitude * torch.randn_like(param.data)
    param.data.add_(noise)
    
def inject_symmetric_noise(param, magnitude):
    """Inject symmetric noise: 0.5 * (N + N.T)"""
    # Calculate standard deviation of the weights
    weight_std = torch.std(param.data).item()
    # Scale noise by 0.4 * standard deviation of weights
    scaled_magnitude = magnitude * weight_std
    noise = scaled_magnitude * torch.randn_like(param.data)
    symmetric_noise = 0.5 * (noise + noise.t())
    param.data.add_(symmetric_noise)
    
def inject_antisymmetric_noise(param, magnitude):
    """Inject antisymmetric noise: 0.5 * (N - N.T)"""
    # Calculate standard deviation of the weights
    weight_std = torch.std(param.data).item()
    # Scale noise by 0.4 * standard deviation of weights
    scaled_magnitude = magnitude * weight_std
    noise = scaled_magnitude * torch.randn_like(param.data)
    antisymmetric_noise = 0.5 * (noise - noise.t())
    param.data.add_(antisymmetric_noise)

# --- Configuration ---
num_epochs = 1
log_every_n_batches = 200 # Frequency of analysis calculation
target_layer_name = 'fc2.weight' # Target the SQUARE hidden layer weight
batch_size = 64

# --- Noise Injection Options ---
noise_types = ['none', 'standard', 'symmetric', 'antisymmetric']  # List of noise types to run
noise_magnitude = 0.4  # Magnitude as a factor of weight standard deviation

# --- Select Matrix Type (Choose ONE) ---
analyze_W = True
analyze_delta_W = False # Note: DeltaW will be based on the weights
# --- End Matrix Type ---

# --- Select Analysis Type (Choose ONE) ---
analyze_spectral_density = True
analyze_level_spacing = False
# --- End Analysis Type ---

# --- Main Function ---
def run_experiment(noise_type, noise_magnitude):
    print(f"\n{'='*80}")
    print(f"Starting experiment with noise type: {noise_type}, magnitude: {noise_magnitude}")
    print(f"{'='*80}")
    
    # --- Sanity Checks ---
    if analyze_W and analyze_delta_W:
        raise ValueError("Please select only ONE matrix type (analyze_W or analyze_delta_W).")
    if not analyze_W and not analyze_delta_W:
        raise ValueError("Please select a matrix type to analyze (analyze_W or analyze_delta_W).")
    if analyze_spectral_density and analyze_level_spacing:
        raise ValueError("Please select only ONE analysis type (analyze_spectral_density or analyze_level_spacing).")
    if not analyze_spectral_density and not analyze_level_spacing:
        raise ValueError("Please select an analysis type (analyze_spectral_density or analyze_level_spacing).")
    if not isinstance(target_layer_name, str) or not target_layer_name:
         raise ValueError("target_layer_name must be a non-empty string.")

    # --- CIFAR-10 Data Loading ---
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 normalization values
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Dataset loaded.")

    # --- Model Definition ---
    class SimpleMLP(nn.Module):
        def __init__(self, input_size=3072, hidden_size1=1024, square_hidden_size=1024, output_size=10):
            super().__init__()
            # Ensure dimensions allow the target layer to be square if needed
            if target_layer_name == 'fc2.weight':
                hidden_size1 = square_hidden_size # Input to fc2 must match its output size

            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, square_hidden_size) # Target square layer
            self.fc3 = nn.Linear(square_hidden_size, output_size)
            self.relu = nn.ReLU()

            if square_hidden_size <= 0:
                 raise ValueError(f"square_hidden_size must be positive, got {square_hidden_size}")

        def forward(self, x):
            x = x.view(x.size(0), -1) # Flatten CIFAR-10 image from [B, 3, 32, 32] to [B, 3072]
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # --- Model Setup ---
    hidden_size_fc2 = 1024 # Ensure this layer is square
    model = SimpleMLP(input_size=3072, square_hidden_size=hidden_size_fc2)  # 3072 = 3 channels * 32 * 32 for CIFAR-10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Verify target layer dimensions after model creation
    target_param_shape = None
    try:
        target_param_check = model.get_parameter(target_layer_name)
        target_param_shape = target_param_check.shape
        if len(target_param_shape) != 2 or target_param_shape[0] != target_param_shape[1]:
            # Raise error because the antisymmetric modification requires a square matrix
            raise ValueError(f"CRITICAL ERROR: Target layer '{target_layer_name}' is NOT SQUARE ({target_param_shape}). Antisymmetric noise injection requires a square matrix.")
        else:
            print(f"Model created. Target layer for analysis: '{target_layer_name}' with size {target_param_shape[0]}x{target_param_shape[1]}")
    except AttributeError:
        raise ValueError(f"Target layer '{target_layer_name}' not found in the model.")

    # Select the appropriate noise injection function based on noise_type
    noise_function = None
    if noise_type == 'none':
        noise_function = inject_no_noise
    elif noise_type == 'standard':
        noise_function = inject_standard_noise
    elif noise_type == 'symmetric':
        noise_function = inject_symmetric_noise
    elif noise_type == 'antisymmetric':
        noise_function = inject_antisymmetric_noise
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # --- Optimizer and Loss ---
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.01) # Example for SGD
    criterion = nn.CrossEntropyLoss()

    # --- Data Storage ---
    analysis_stats = {
        'batch': [],
        'matrix_type': 'W' if analyze_W else 'DeltaW',
        'analysis_type': 'Density' if analyze_spectral_density else 'Spacing',
        'loss_values': [], # Track loss per batch
        'batch_numbers': [], # Corresponding batch numbers for loss values
        'results': {}, # Will be populated based on analysis type
        'noise_type': noise_type # Add noise type for plot titles
    }

    # Initialize results structure based on analysis type
    if analyze_spectral_density:
        analysis_stats['results']['eigenvalues_list'] = []
        analysis_stats['results']['last_eigenvalues'] = None
        if analyze_W:
            matrix_desc = "Centered & Normalized Weight (W' / (std(W') * sqrt(N)))"
        else: # analyze_delta_W
            matrix_desc = f"Normalized Weight Update (ΔW / (std(ΔW) * sqrt(N)))"
    elif analyze_level_spacing:
        analysis_stats['results']['std_dev_norm_spacing_list'] = []
        analysis_stats['results']['last_normalized_spacings'] = None
        matrix_desc = f"{'Weight Matrix (W)' if analyze_W else 'Weight Update Matrix (ΔW)'}"

    analysis_stats['matrix_description'] = matrix_desc # Store for plotting titles

    # --- Training Loop ---
    current_batch = 0
    start_time = time.time()
    print(f"\nStarting CIFAR-10 training on {device} for {num_epochs} epochs.")
    print(f"Noise Type: {noise_type}, Magnitude: {noise_magnitude}")
    print(f"Analyzing: {analysis_stats['analysis_type']} of {analysis_stats['matrix_description']}")
    print(f"Calculation Frequency: Every {log_every_n_batches} batches.")

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss_sum = 0.0
        num_batches_in_epoch = 0
        last_batch_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # --- Standard Training Step ---
            model.train()
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # Calculate gradients for ALL parameters
            
            # Apply optimizer step
            optimizer.step()
            
            # --- Inject Noise After Optimization ---
            try:
                target_param = model.get_parameter(target_layer_name)
                if target_param.shape[0] == target_param.shape[1]: # Check if square
                    # Apply the selected noise function
                    if noise_function is not None:
                        if noise_type == 'none':
                            noise_function(target_param)
                        else:
                            noise_function(target_param, noise_magnitude)
                else:
                    print(f"  WARNING: Target layer '{target_layer_name}' is not square. Cannot apply noise injection. Skipping.")
            except AttributeError:
                print(f"  ERROR: Could not find target layer '{target_layer_name}' during noise injection.")
            except Exception as e:
                print(f"  ERROR during noise injection: {e}")

            # --- Perform Analysis Calculation ---
            if (current_batch % log_every_n_batches == 0):
                calc_start_time = time.time()
                print(f"\n--- Analyzing Batch {current_batch} (Epoch {epoch+1}) ---")
                with torch.no_grad():
                    matrix = None
                    N = target_param_shape[0] # Use shape determined at setup

                    try:
                        target_param = model.get_parameter(target_layer_name)
                        # N check already done at setup, assume it's square here

                        # --- Get the raw matrix ---
                        if analyze_W:
                            matrix = target_param.data.cpu().numpy()
                        elif analyze_delta_W:
                            if target_param.grad is None:
                                print(f"  Skipping ΔW analysis: Gradient is None.")
                                matrix = None # Skip analysis
                            else:
                                grad_W_tensor = target_param.grad
                                lr = optimizer.param_groups[0]['lr']
                                # Delta W = -lr * gradient
                                delta_W_tensor = -lr * grad_W_tensor
                                matrix = delta_W_tensor.cpu().numpy()
                                print(f"  Calculated ΔW matrix (shape {matrix.shape})")


                        # --- Calculate Eigenvalues and Perform Chosen Analysis ---
                        if matrix is not None:
                            # --- Check if matrix is effectively zero ---
                            matrix_std = np.std(matrix)
                            if matrix_std < 1e-15:
                                 print(f"  Skipping analysis: Matrix standard deviation is near zero ({matrix_std:.2e}). Matrix is likely zero or constant.")
                                 matrix = None # Prevent further analysis on effectively zero matrix

                        if matrix is not None:
                            try:
                                eigenvalues_complex = np.linalg.eigvals(matrix)
                            except np.linalg.LinAlgError:
                                print(f"  Error: Eigenvalue computation failed for batch {current_batch}. Skipping analysis.")
                                matrix = None # Skip rest of analysis for this batch

                        if matrix is not None:
                             # Proceed with analysis using eigenvalues_complex
                            if analyze_spectral_density:
                                # --- Density Analysis: Normalize matrix THEN compute eigenvalues ---
                                matrix_norm = None
                                if analyze_W:
                                    W_centered = matrix - np.mean(matrix)
                                    sigma_W_centered = np.std(W_centered)
                                    # print(f"  Density(W): Mean(W)={np.mean(matrix):.3g}, Std(W')={sigma_W_centered:.3g}")
                                    if sigma_W_centered > 1e-15:
                                        matrix_norm = W_centered / (sigma_W_centered * np.sqrt(N))
                                    else: print("  Skipping Density(W): Std dev near zero.")
                                else: # analyze_delta_W
                                    sigma_delta_w = np.std(matrix) # Std of delta_W
                                    # print(f"  Density(ΔW): Std(ΔW)={sigma_delta_w:.3g}")
                                    if sigma_delta_w > 1e-15:
                                        matrix_norm = matrix / (sigma_delta_w * np.sqrt(N))
                                    else: print("  Skipping Density(ΔW): Std dev near zero.")

                                if matrix_norm is not None:
                                    try:
                                        norm_eigenvalues_complex = np.linalg.eigvals(matrix_norm)
                                        norm_eigenvalues = np.real(norm_eigenvalues_complex)
                                        print(f"  Calculated {len(norm_eigenvalues)} normalized eigenvalues.")
                                        analysis_stats['results']['eigenvalues_list'].append(norm_eigenvalues)
                                        analysis_stats['results']['last_eigenvalues'] = norm_eigenvalues
                                        analysis_stats['batch'].append(current_batch) # Add batch only on success
                                    except np.linalg.LinAlgError:
                                         print(f"  Error: Eigenvalue computation failed for *normalized* matrix. Skipping density analysis.")

                            elif analyze_level_spacing:
                                # --- Spacing Analysis: Use raw eigenvalues ---
                                eigenvalues = np.real(eigenvalues_complex) # Use real part for spacing
                                print(f"  Calculated {len(eigenvalues)} raw eigenvalues for spacing analysis.")

                                if len(eigenvalues) >= 2:
                                    sorted_eigenvalues = np.sort(eigenvalues)
                                    spacings = np.diff(sorted_eigenvalues)

                                    if len(spacings) > 0:
                                        # Filter out tiny/zero spacings BEFORE calculating mean
                                        spacings_filtered = spacings[spacings > 1e-10]
                                        if len(spacings_filtered) > 0:
                                            mean_spacing = np.mean(spacings_filtered)
                                            # print(f"  Mean spacing (s>1e-10): {mean_spacing:.4g}")

                                            if mean_spacing > 1e-10:
                                                normalized_spacings = spacings / mean_spacing # Normalize original spacings
                                                # Filter *again* for stats/plotting if needed (e.g., std dev of s>0)
                                                norm_spacings_positive = normalized_spacings[normalized_spacings > 1e-10]

                                                if len(norm_spacings_positive) > 0:
                                                    std_dev_norm = np.std(norm_spacings_positive)
                                                    print(f"  Std Dev of Normalized Spacings (s > 1e-10): {std_dev_norm:.4f}")
                                                    analysis_stats['results']['std_dev_norm_spacing_list'].append(std_dev_norm)
                                                    analysis_stats['results']['last_normalized_spacings'] = normalized_spacings # Store all norm. spacings
                                                    analysis_stats['batch'].append(current_batch) # Add batch only on success
                                                else:
                                                    print("  No positive normalized spacings found for std dev calculation.")
                                            else:
                                                 print("  Mean spacing is too small for normalization.")
                                        else:
                                            print("  No spacings > 1e-10 found.")
                                    else:
                                        print("  Not enough spacings calculated.")
                                else:
                                    print(f"  Not enough eigenvalues ({len(eigenvalues)}) for spacing analysis.")

                    except AttributeError:
                        print(f"  Error: Layer '{target_layer_name}' not found during analysis.")
                    except ValueError as ve:
                        print(f"  Error during analysis setup: {ve}.")
                    except Exception as e:
                        print(f"  An unexpected error occurred during analysis: {e}")

                calc_duration = time.time() - calc_start_time
                # print(f"  Analysis Calculation Time: {calc_duration:.2f}s")

            # --- Post-Step ---
            current_batch += 1
            num_batches_in_epoch += 1
            epoch_loss_sum += loss.item()
            last_batch_loss = loss.item()
            
            # Track loss for plotting
            analysis_stats['loss_values'].append(loss.item())
            analysis_stats['batch_numbers'].append(current_batch)

        # --- End of Epoch ---
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f}s. Avg Loss: {avg_epoch_loss:.4f}. (Last batch loss: {last_batch_loss:.4f})")

    total_duration = time.time() - start_time
    print(f"\nTraining finished in {total_duration:.2f}s")

    return analysis_stats

# --- Plotting Functions ---
def plot_results(analysis_stats, plots_dir, timestamp):
    """Plot results for a single experiment"""
    print("\nGenerating plots...")
    
    noise_type = analysis_stats['noise_type']
    matrix_title_str = analysis_stats['matrix_description']
    target_param_shape = None
    try:
        # Get shape from the first batch entry if available
        if 'eigenvalues_list' in analysis_stats['results'] and len(analysis_stats['results']['eigenvalues_list']) > 0:
            target_param_shape = (len(analysis_stats['results']['eigenvalues_list'][0]), len(analysis_stats['results']['eigenvalues_list'][0]))
        else:
            # Use a default value if unavailable
            target_param_shape = (1024, 1024)  # Default from the model setup
    except:
        target_param_shape = (1024, 1024)  # Fallback

    layer_info_str = f"Layer '{target_layer_name}', N={target_param_shape[0]}"
    
    # Create noise type string for titles
    noise_title = f"Noise Type: {noise_type.capitalize()}"
    if noise_type != 'none':
        noise_title += f", Magnitude: {noise_magnitude}"

    # --- Plot Loss Per Batch ---
    if len(analysis_stats['loss_values']) > 0:
        print("\nPlotting loss per batch...")
        plt.figure(figsize=(10, 6))
        plt.plot(analysis_stats['batch_numbers'], analysis_stats['loss_values'], 'b-')
        plt.xlabel('Batch Number')
        plt.ylabel('Loss')
        plt.title(f'CIFAR-10 Training Loss\n{noise_title}')
        plt.grid(True)
        
        # Add moving average to smooth the curve
        if len(analysis_stats['loss_values']) > 20:
            window_size = min(20, len(analysis_stats['loss_values']) // 5)
            moving_avg = np.convolve(analysis_stats['loss_values'], 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            # Plot moving average starting at the window_size-1 point
            plt.plot(analysis_stats['batch_numbers'][window_size-1:window_size-1+len(moving_avg)], 
                    moving_avg, 'r-', linewidth=2, alpha=0.7, 
                    label=f'Moving Avg (window={window_size})')
            plt.legend()
        
        # Save loss plot
        loss_plot_filename = f"{timestamp}_{noise_type}_cifar10_loss_curve.png"
        loss_plot_path = os.path.join(plots_dir, loss_plot_filename)
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.tight_layout()
        print(f"Saved loss plot to: {loss_plot_path}")
        plt.close()

    num_calculations = len(analysis_stats['batch'])

    if num_calculations == 0:
        print("No analysis calculations were successfully completed. Cannot generate plots.")
        return

    if analyze_spectral_density:
        # --- Plot Spectral Density Evolution ---
        print(f"Plotting Spectral Density evolution for {matrix_title_str}...")
        results_data = analysis_stats['results']['eigenvalues_list']
        batch_numbers = analysis_stats['batch']

        num_plots_aim = 6
        num_plots_to_show = min(num_calculations, num_plots_aim)
        if num_plots_to_show <= 0: 
            print("No density data points available.")
            return
            
        ncols = math.ceil(math.sqrt(num_plots_to_show * 1.2))
        nrows = math.ceil(num_plots_to_show / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5), squeeze=False)
        axes = axes.flatten()

        if num_plots_to_show == 1: indices_to_plot = [0]
        else:
            # Ensure indices are unique and cover the range reasonably
            indices_to_plot = np.linspace(0, num_calculations - 1, num_plots_to_show, dtype=int)
            indices_to_plot = np.unique(indices_to_plot)
            num_plots_to_show = len(indices_to_plot) # Update actual number of plots

        print(f"Plotting {num_plots_to_show} density snapshots at batches: {[batch_numbers[i] for i in indices_to_plot]}")

        # Determine common plot range based on selected snapshots
        max_R_fit = 0.0
        all_eigenvalues_list = []
        valid_indices = []
        for idx in indices_to_plot:
             if idx < len(results_data) and results_data[idx] is not None and len(results_data[idx]) > 0:
                  all_eigenvalues_list.append(results_data[idx])
                  sigma_empirical = np.std(results_data[idx])
                  if np.isfinite(sigma_empirical) and sigma_empirical > 0:
                       max_R_fit = max(max_R_fit, sigma_empirical)
                  valid_indices.append(idx) # Keep track of indices with valid data

        # Recalculate plot layout if some data was invalid
        num_plots_to_show = len(valid_indices)
        if num_plots_to_show <= 0:
             print("No valid density data points available for plotting.")
             return

        common_plot_range = (-2.5, 2.5) # Default
        if all_eigenvalues_list:
             all_selected_eigenvalues = np.concatenate(all_eigenvalues_list)
             data_min = np.min(all_selected_eigenvalues)
             data_max = np.max(all_selected_eigenvalues)
             # Wigner semicircle extends from -2R to 2R. Use max_R_fit for scale.
             radius_scale = max(max_R_fit, 1.0) # Use at least 1.0 if std is tiny
             plot_min = min(data_min * 1.1 if data_min < 0 else data_min * 0.9, -2.0 * radius_scale * 1.05, -0.1)
             plot_max = max(data_max * 1.1 if data_max > 0 else data_max * 0.9, 2.0 * radius_scale * 1.05, 0.1)
             # Ensure range is reasonable, e.g., not excessively large if data is clustered
             plot_min = max(plot_min, -5.0 * radius_scale) # Limit how far left
             plot_max = min(plot_max, 5.0 * radius_scale) # Limit how far right
             if plot_max - plot_min < 1e-6: # Avoid zero range
                  plot_min -= 0.5
                  plot_max += 0.5
             common_plot_range = (plot_min, plot_max)
             print(f"  Determined common plot range: ({common_plot_range[0]:.2f}, {common_plot_range[1]:.2f}) based on R_fit up to {max_R_fit:.2f}")

        plot_count = 0
        for i, data_idx in enumerate(valid_indices): # Iterate through valid indices only
            ax = axes[plot_count] # Use plot_count for indexing axes
            batch_num = batch_numbers[data_idx]
            eigenvalues = results_data[data_idx]

            sigma_empirical = np.std(eigenvalues)
            # Use empirical std for R_fit, ensuring it's positive for the formula
            R_fit = sigma_empirical if (np.isfinite(sigma_empirical) and sigma_empirical > 1e-9) else 1.0 # Use 1.0 as fallback

            # Wigner semicircle formula
            x_semicircle_fit = np.linspace(-2 * R_fit, 2 * R_fit, 300)
            # Prevent sqrt of negative numbers due to float precision
            sqrt_term_fit = np.sqrt(np.maximum(0, (2 * R_fit)**2 - x_semicircle_fit**2))
            # Avoid division by zero if R_fit is extremely small
            rho_semicircle_fit = (1 / (np.pi * max(R_fit**2, 1e-15) * 2.0)) * sqrt_term_fit


            num_bins = max(min(len(eigenvalues)//10, 75), 30) # Adaptive binning
            ax.hist(eigenvalues, bins=num_bins, density=True, alpha=0.75, label=f'Empirical ρ(λ)', range=common_plot_range)
            ax.plot(x_semicircle_fit, rho_semicircle_fit, 'r--', linewidth=1.5, label=f'Wigner (R={R_fit:.2f})')
            ax.set_xlabel("Normalized Eigenvalue (λ)")
            ax.set_ylabel("Density ρ(λ)")
            ax.set_title(f"Batch: {batch_num} (σ={sigma_empirical:.2f})")
            ax.legend(fontsize='small')
            ax.grid(True, alpha=0.5)
            ax.set_xlim(common_plot_range)
            ax.set_ylim(bottom=0)
            plot_count += 1

        # Turn off unused axes
        for j in range(plot_count, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f"CIFAR-10 Spectral Density Evolution\n{noise_title}\n{matrix_title_str}\n{layer_info_str}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.91]) # Adjust rect to prevent title overlap
        
        # Save plot to file
        plot_filename = f"{timestamp}_{noise_type}_spectral_density.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved spectral density plot to: {plot_path}")
        plt.close()

    elif analyze_level_spacing:
        # --- Plot Level Spacing Evolution and Final P(s) ---
        print(f"Plotting Level Spacing evolution for {matrix_title_str}...")
        std_dev_list = analysis_stats['results']['std_dev_norm_spacing_list']
        last_spacings = analysis_stats['results']['last_normalized_spacings']
        batch_numbers = analysis_stats['batch']

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Evolution of Std Dev
        ax = axes[0]
        if len(batch_numbers) == len(std_dev_list) and len(batch_numbers) > 0:
            ax.plot(batch_numbers, std_dev_list, marker='.', linestyle='-')
            ax.set_xlabel("Number of Batches Performed")
            ax.set_ylabel("Std Dev of Norm. Spacings (s > 1e-10)")
            ax.set_title(f"Evolution of Level Spacing Std Dev")
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No std dev data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Evolution of Level Spacing Std Dev")


        # Plot 2: Histogram of Final Normalized Spacings P(s)
        ax = axes[1]
        if last_spacings is not None and len(last_spacings) > 0:
            # Filter for positive spacings for log scale and comparison distributions
            positive_spacings = last_spacings[last_spacings > 1e-10]

            if len(positive_spacings) > 0:
                 # Determine appropriate bins for log scale
                min_s_val = np.min(positive_spacings)
                max_s_val = np.max(positive_spacings)
                # Handle cases where min/max are zero or equal, or non-finite
                if not np.isfinite(min_s_val) or min_s_val <= 1e-10: min_s_val = 1e-5 # Set a small positive floor
                if not np.isfinite(max_s_val) or max_s_val <= min_s_val: max_s_val = min_s_val * 1000 # Ensure max > min

                min_log_s = np.log10(min_s_val)
                max_log_s = np.log10(max_s_val)
                num_bins = 50
                # Ensure log bins are valid
                if max_log_s <= min_log_s : max_log_s = min_log_s + 3 # Ensure range if somehow still equal
                log_bins = np.logspace(min_log_s, max_log_s, num=num_bins + 1)

                ax.hist(positive_spacings, bins=log_bins, density=True, alpha=0.75, label='Empirical P(s)')
                ax.set_xscale('log') # Set X-axis to log scale

                # Plot reference distributions on the log scale
                s_min_plot, s_max_plot = ax.get_xlim()
                # Ensure limits are positive for logspace, using calculated min/max if plot limits are bad
                s_min_plot = max(s_min_plot, min_s_val * 0.9, 1e-6)
                s_max_plot = max(s_max_plot, max_s_val * 1.1)
                if s_max_plot <= s_min_plot: s_max_plot = s_min_plot * 1000 # Final safety

                s_theory_log = np.logspace(np.log10(s_min_plot), np.log10(s_max_plot), 200)

                # Theory distributions
                poisson = np.exp(-s_theory_log)
                goe = (np.pi / 2.0) * s_theory_log * np.exp(-np.pi / 4.0 * s_theory_log**2)

                ax.plot(s_theory_log, poisson, 'r--', label='Poisson ($e^{-s}$)')
                ax.plot(s_theory_log, goe, 'g--', label='GOE ($\frac{\pi}{2}s e^{-\pi s^2/4}$)')

                ax.set_xlabel("Normalized Level Spacing (s) [Log Scale]")
                ax.set_ylabel("Probability Density P(s)")
                final_batch = batch_numbers[-1] if batch_numbers else 'N/A'
                ax.set_title(f"Final P(s) at Batch {final_batch} ({len(positive_spacings)} pos. spacings)")
                ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.set_ylim(bottom=0)
                # Explicitly set xlim again after plotting theory lines
                ax.set_xlim(s_min_plot, s_max_plot)

            else:
                ax.text(0.5, 0.5, "No positive spacing data\navailable for log plot.", ha='center', va='center', transform=ax.transAxes)
                ax.set_title("Final P(s) - Log Scale Unavailable")
        else:
            ax.text(0.5, 0.5, "No final spacing data available.", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Final P(s) - Not Available")

        fig.suptitle(f"CIFAR-10 Level Spacing Analysis\n{noise_title}\n{matrix_title_str}\n{layer_info_str}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.91])
        
        # Save plot to file
        plot_filename = f"{timestamp}_{noise_type}_level_spacing.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved level spacing plot to: {plot_path}")
        plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    # Create plots directory
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}")
    
    # Generate timestamp for all experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run experiments for each noise type
    all_results = {}
    for noise_type in noise_types:
        print(f"\n\n{'='*80}")
        print(f"STARTING EXPERIMENT WITH NOISE TYPE: {noise_type.upper()}")
        print(f"{'='*80}")
        
        # Run the experiment
        results = run_experiment(noise_type, noise_magnitude)
        
        # Store results
        all_results[noise_type] = results
        
        # Generate plots
        plot_results(results, plots_dir, timestamp)
    
    print("\nAll experiments completed.")
    warnings.filterwarnings("default", category=RuntimeWarning)