# -*- coding: utf-8 -*-
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

# --- Configuration ---
num_epochs = 1
log_every_n_batches = 200 # Frequency of analysis calculation
batch_size = 64
num_hidden_layers = 2  # Number of hidden layers with square dimensions (default: 2)

# --- Select Matrix Type (Choose ONE) ---
analyze_W = True
analyze_delta_W = False # Note: If True, DeltaW will be based on the gradient
# --- End Matrix Type ---

# --- Select Analysis Type (Choose ONE or BOTH) ---
analyze_spectral_density = True
analyze_level_spacing = False
analyze_singular_values = True  # <<< NEW: Set to True to analyze singular value distributions
# --- End Analysis Type ---

# --- Sanity Checks ---
if analyze_W and analyze_delta_W:
    raise ValueError("Please select only ONE matrix type (analyze_W or analyze_delta_W).")
if not analyze_W and not analyze_delta_W:
    raise ValueError("Please select a matrix type to analyze (analyze_W or analyze_delta_W).")
if not analyze_spectral_density and not analyze_level_spacing and not analyze_singular_values:
    raise ValueError("Please select at least one analysis type (analyze_spectral_density, analyze_level_spacing, or analyze_singular_values).")
if analyze_spectral_density and analyze_level_spacing:
    raise ValueError("Please select only ONE eigenvalue analysis type (analyze_spectral_density or analyze_level_spacing).")

# --- MNIST Data Loading ---
print("Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("Dataset loaded.")

# --- Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=1024, output_size=10, num_hidden_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        
        # First layer (non-square): input_size to hidden_size
        self.fc_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        
        # Middle hidden layers (square): hidden_size to hidden_size
        for i in range(num_hidden_layers):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Last layer (non-square): hidden_size to output_size
        self.fc_layers.append(nn.Linear(hidden_size, output_size))
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image
        
        # Process through all layers except the last one with ReLU
        for i in range(len(self.fc_layers) - 1):
            x = self.relu(self.fc_layers[i](x))
        
        # Last layer without ReLU
        x = self.fc_layers[-1](x)
        return x

# --- Model Setup ---
hidden_size = 1024
model = SimpleMLP(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Define target layers for analysis ---
target_layers = []
# First layer (non-square)
target_layers.append('fc_layers.0.weight')
# Middle layers (square)
for i in range(num_hidden_layers):
    target_layers.append(f'fc_layers.{i+1}.weight')
# Last layer (non-square)
target_layers.append(f'fc_layers.{num_hidden_layers+1}.weight')

# Verify target layer dimensions
layer_shapes = {}
for target_layer_name in target_layers:
    try:
        target_param = model.get_parameter(target_layer_name)
        layer_shapes[target_layer_name] = target_param.shape
        print(f"Layer for analysis: '{target_layer_name}' with size {target_param.shape[0]}x{target_param.shape[1]}")
    except AttributeError:
        print(f"WARNING: Target layer '{target_layer_name}' not found in the model.")

# --- Optimizer and Loss ---
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# --- Data Storage ---
analysis_stats = {
    'batch': [],
    'matrix_type': 'W' if analyze_W else 'DeltaW',
    'analysis_type': 'Density' if analyze_spectral_density else 'Spacing',
    'loss_values': [], # Track loss per batch
    'batch_numbers': [], # Corresponding batch numbers for loss values
    'results': {layer: {} for layer in target_layers}
}

# Initialize results structure based on analysis type
for layer in target_layers:
    if analyze_spectral_density:
        analysis_stats['results'][layer]['eigenvalues_list'] = []
        analysis_stats['results'][layer]['last_eigenvalues'] = None
    elif analyze_level_spacing:
        analysis_stats['results'][layer]['std_dev_norm_spacing_list'] = []
        analysis_stats['results'][layer]['last_normalized_spacings'] = None
    if analyze_singular_values:  # NEW: Add singular value storage
        analysis_stats['results'][layer]['singular_values_list'] = []
        analysis_stats['results'][layer]['last_singular_values'] = None

if analyze_W:
    matrix_desc = "Centered & Normalized Weight (W' / (std(W') * sqrt(N)))"
else: # analyze_delta_W
    matrix_desc = "Normalized Weight Update (ΔW / (std(ΔW) * sqrt(N)))"

analysis_stats['matrix_description'] = matrix_desc # Store for plotting titles

# --- Training Loop ---
current_batch = 0
start_time = time.time()
print(f"\nStarting training on {device} for {num_epochs} epochs.")
print(f"Analyzing: {analysis_stats['analysis_type']} of {analysis_stats['matrix_description']}")
if analyze_singular_values:
    print(f"Additionally analyzing: Singular Value Distributions")
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

        # --- Perform Analysis Calculation ---
        if (current_batch % log_every_n_batches == 0):
            calc_start_time = time.time()
            print(f"\n--- Analyzing Batch {current_batch} (Epoch {epoch+1}) ---")
            
            # Check if we need to record this batch
            need_to_record_batch = False
            
            with torch.no_grad():
                for target_layer_name in target_layers:
                    try:
                        target_param = model.get_parameter(target_layer_name)
                        
                        # --- Get the raw matrix ---
                        if analyze_W:
                            matrix = target_param.data.cpu().numpy()
                        elif analyze_delta_W:
                            if target_param.grad is None:
                                print(f"  Skipping {target_layer_name} ΔW analysis: Gradient is None.")
                                matrix = None # Skip analysis
                            else:
                                grad_W_tensor = target_param.grad
                                lr = optimizer.param_groups[0]['lr']
                                # Delta W = -lr * gradient
                                delta_W_tensor = -lr * grad_W_tensor
                                matrix = delta_W_tensor.cpu().numpy()
                        
                        # --- Calculate Eigenvalues and Perform Chosen Analysis ---
                        if matrix is not None:
                            # --- Check if matrix is effectively zero ---
                            matrix_std = np.std(matrix)
                            if matrix_std < 1e-15:
                                print(f"  Skipping {target_layer_name} analysis: Matrix standard deviation is near zero ({matrix_std:.2e}).")
                                continue

                            try:
                                eigenvalues_complex = np.linalg.eigvals(matrix)
                            except np.linalg.LinAlgError:
                                print(f"  Error: Eigenvalue computation failed for {target_layer_name} at batch {current_batch}. Skipping analysis.")
                                continue

                            if analyze_spectral_density:
                                # --- Density Analysis: Normalize matrix THEN compute eigenvalues ---
                                matrix_norm = None
                                if analyze_W:
                                    W_centered = matrix - np.mean(matrix)
                                    sigma_W_centered = np.std(W_centered)
                                    if sigma_W_centered > 1e-15:
                                        matrix_norm = W_centered / (sigma_W_centered * np.sqrt(min(matrix.shape)))
                                    else: 
                                        print(f"  Skipping {target_layer_name} Density(W): Std dev near zero.")
                                else: # analyze_delta_W
                                    sigma_delta_w = np.std(matrix)
                                    if sigma_delta_w > 1e-15:
                                        matrix_norm = matrix / (sigma_delta_w * np.sqrt(min(matrix.shape)))
                                    else: 
                                        print(f"  Skipping {target_layer_name} Density(ΔW): Std dev near zero.")

                                if matrix_norm is not None:
                                    try:
                                        norm_eigenvalues_complex = np.linalg.eigvals(matrix_norm)
                                        norm_eigenvalues = np.real(norm_eigenvalues_complex)
                                        print(f"  Calculated {len(norm_eigenvalues)} normalized eigenvalues for {target_layer_name}.")
                                        analysis_stats['results'][target_layer_name]['eigenvalues_list'].append(norm_eigenvalues)
                                        analysis_stats['results'][target_layer_name]['last_eigenvalues'] = norm_eigenvalues
                                        need_to_record_batch = True
                                    except np.linalg.LinAlgError:
                                        print(f"  Error: Eigenvalue computation failed for *normalized* matrix of {target_layer_name}. Skipping density analysis.")


                            elif analyze_level_spacing:
                                # --- Spacing Analysis: Use raw eigenvalues ---
                                eigenvalues = np.real(eigenvalues_complex) # Use real part for spacing
                                print(f"  Calculated {len(eigenvalues)} raw eigenvalues for {target_layer_name} spacing analysis.")

                                if len(eigenvalues) >= 2:
                                    sorted_eigenvalues = np.sort(eigenvalues)
                                    spacings = np.diff(sorted_eigenvalues)

                                    if len(spacings) > 0:
                                        # Filter out tiny/zero spacings BEFORE calculating mean
                                        spacings_filtered = spacings[spacings > 1e-10]
                                        if len(spacings_filtered) > 0:
                                            mean_spacing = np.mean(spacings_filtered)

                                            if mean_spacing > 1e-10:
                                                normalized_spacings = spacings / mean_spacing # Normalize original spacings
                                                # Filter *again* for stats/plotting if needed (e.g., std dev of s>0)
                                                norm_spacings_positive = normalized_spacings[normalized_spacings > 1e-10]

                                                if len(norm_spacings_positive) > 0:
                                                    std_dev_norm = np.std(norm_spacings_positive)
                                                    print(f"  Std Dev of Normalized Spacings for {target_layer_name} (s > 1e-10): {std_dev_norm:.4f}")
                                                    analysis_stats['results'][target_layer_name]['std_dev_norm_spacing_list'].append(std_dev_norm)
                                                    analysis_stats['results'][target_layer_name]['last_normalized_spacings'] = normalized_spacings # Store all norm. spacings
                                                    need_to_record_batch = True
                                                else:
                                                    print(f"  No positive normalized spacings found for {target_layer_name} std dev calculation.")
                                            else:
                                                print(f"  Mean spacing for {target_layer_name} is too small for normalization.")
                                        else:
                                            print(f"  No spacings > 1e-10 found for {target_layer_name}.")
                                    else:
                                        print(f"  Not enough spacings calculated for {target_layer_name}.")
                                else:
                                    print(f"  Not enough eigenvalues ({len(eigenvalues)}) for {target_layer_name} spacing analysis.")
                            
                            # NEW: Compute singular values if requested    
                            if analyze_singular_values:
                                # --- Singular Value Analysis ---
                                try:
                                    # Normalize matrix to make comparisons more meaningful
                                    if analyze_W:
                                        W_centered = matrix - np.mean(matrix)
                                        sigma_W_centered = np.std(W_centered)
                                        if sigma_W_centered > 1e-15:
                                            matrix_norm = W_centered / (sigma_W_centered * np.sqrt(min(matrix.shape)))
                                        else:
                                            print(f"  Skipping {target_layer_name} SVD: Std dev near zero.")
                                            matrix_norm = None
                                    else: # analyze_delta_W
                                        sigma_delta_w = np.std(matrix)
                                        if sigma_delta_w > 1e-15:
                                            matrix_norm = matrix / (sigma_delta_w * np.sqrt(min(matrix.shape)))
                                        else:
                                            print(f"  Skipping {target_layer_name} SVD: Std dev near zero.")
                                            matrix_norm = None
                                    
                                    if matrix_norm is not None:
                                        # Compute singular values
                                        singular_values = np.linalg.svd(matrix_norm, compute_uv=False)
                                        print(f"  Calculated {len(singular_values)} normalized singular values for {target_layer_name}.")
                                        analysis_stats['results'][target_layer_name]['singular_values_list'].append(singular_values)
                                        analysis_stats['results'][target_layer_name]['last_singular_values'] = singular_values
                                        need_to_record_batch = True
                                
                                except np.linalg.LinAlgError:
                                    print(f"  Error: SVD computation failed for {target_layer_name}. Skipping singular value analysis.")
                                except Exception as e:
                                    print(f"  Error during singular value computation for {target_layer_name}: {e}")

                    except AttributeError:
                        print(f"  Error: Layer '{target_layer_name}' not found during analysis.")
                    except Exception as e:
                        print(f"  An unexpected error occurred during analysis of {target_layer_name}: {e}")
            
            # Only record the batch if at least one analysis was successful
            if need_to_record_batch:
                analysis_stats['batch'].append(current_batch)
                
            calc_duration = time.time() - calc_start_time
            # print(f"  Analysis Calculation Time: {calc_duration:.2f}s")

        # --- Optimizer Step ---
        optimizer.step()

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


# --- Plotting ---
print("\nGenerating plots...")

# Create plots directory if it doesn't exist
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created directory: {plots_dir}")

# Generate timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

num_calculations = len(analysis_stats['batch'])
matrix_title_str = analysis_stats['matrix_description']

# --- Plot Loss Per Batch ---
if len(analysis_stats['loss_values']) > 0:
    print("\nPlotting loss per batch...")
    plt.figure(figsize=(10, 6))
    plt.plot(analysis_stats['batch_numbers'], analysis_stats['loss_values'], 'b-')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('MNIST Training Loss')
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
    loss_plot_filename = f"{timestamp}_mnist_loss_curve.png"
    loss_plot_path = os.path.join(plots_dir, loss_plot_filename)
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    print(f"Saved loss plot to: {loss_plot_path}")

if num_calculations == 0:
    print("No analysis calculations were successfully completed. Cannot generate plots.")
else:
    # Create separate plots for each layer
    for target_layer_name in target_layers:
        if analyze_spectral_density:
            # --- Plot Spectral Density Evolution ---
            layer_results = analysis_stats['results'][target_layer_name]
            if 'eigenvalues_list' not in layer_results or not layer_results['eigenvalues_list']:
                print(f"No eigenvalues data for {target_layer_name}. Skipping plot.")
                continue
                
            print(f"Plotting Spectral Density evolution for {target_layer_name} {matrix_title_str}...")
            results_data = layer_results['eigenvalues_list']
            batch_numbers = analysis_stats['batch']

            num_plots_aim = 6
            num_plots_to_show = min(len(results_data), num_plots_aim)
            if num_plots_to_show <= 0: 
                print(f"No density data points available for {target_layer_name}.")
                continue
                
            ncols = math.ceil(math.sqrt(num_plots_to_show * 1.2))
            nrows = math.ceil(num_plots_to_show / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5), squeeze=False)
            axes = axes.flatten()

            if num_plots_to_show == 1: 
                indices_to_plot = [0]
            else:
                # Ensure indices are unique and cover the range reasonably
                indices_to_plot = np.linspace(0, len(results_data) - 1, num_plots_to_show, dtype=int)
                indices_to_plot = np.unique(indices_to_plot)
                num_plots_to_show = len(indices_to_plot) # Update actual number of plots

            print(f"Plotting {num_plots_to_show} density snapshots at batches: {[batch_numbers[i] for i in indices_to_plot if i < len(batch_numbers)]}")

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
                 print(f"No valid density data points available for plotting {target_layer_name}.")
                 continue

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
                batch_num = batch_numbers[data_idx] if data_idx < len(batch_numbers) else "Unknown"
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

            layer_shape = layer_shapes.get(target_layer_name, "unknown shape")
            layer_info_str = f"Layer '{target_layer_name}', Shape: {layer_shape}"
            fig.suptitle(f"Spectral Density Evolution ({analysis_stats['analysis_type']})\n{matrix_title_str}\n{layer_info_str}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.91]) # Adjust rect to prevent title overlap
            
            # Save plot to file
            plot_filename = f"{timestamp}_{target_layer_name.replace('.', '_')}_spectral_density.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved spectral density plot for {target_layer_name} to: {plot_path}")

        elif analyze_level_spacing:
            # --- Plot Level Spacing Evolution and Final P(s) ---
            layer_results = analysis_stats['results'][target_layer_name]
            if 'std_dev_norm_spacing_list' not in layer_results or not layer_results['std_dev_norm_spacing_list']:
                print(f"No level spacing data for {target_layer_name}. Skipping plot.")
                continue
                
            print(f"Plotting Level Spacing evolution for {target_layer_name} {matrix_title_str}...")
            std_dev_list = layer_results['std_dev_norm_spacing_list']
            last_spacings = layer_results['last_normalized_spacings']
            batch_numbers = analysis_stats['batch']

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Plot 1: Evolution of Std Dev
            ax = axes[0]
            if len(std_dev_list) > 0:
                # Use the batch numbers corresponding to successful analyses
                plot_batches = batch_numbers[:len(std_dev_list)]
                ax.plot(plot_batches, std_dev_list, marker='.', linestyle='-')
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

            layer_shape = layer_shapes.get(target_layer_name, "unknown shape")
            layer_info_str = f"Layer '{target_layer_name}', Shape: {layer_shape}"
            fig.suptitle(f"Level Spacing Analysis ({analysis_stats['analysis_type']})\n{matrix_title_str}\n{layer_info_str}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.91])
            
            # Save plot to file
            plot_filename = f"{timestamp}_{target_layer_name.replace('.', '_')}_level_spacing.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved level spacing plot for {target_layer_name} to: {plot_path}")

        # NEW: Plot Singular Value Distribution if analyzed
        if analyze_singular_values:
            # Check if we have singular value data for this layer
            layer_results = analysis_stats['results'][target_layer_name]
            if 'singular_values_list' not in layer_results or not layer_results['singular_values_list']:
                print(f"No singular value data for {target_layer_name}. Skipping plot.")
                continue
                
            print(f"Plotting Singular Value distribution for {target_layer_name} {matrix_title_str}...")
            results_data = layer_results['singular_values_list']
            batch_numbers = analysis_stats['batch']

            num_plots_aim = 6
            num_plots_to_show = min(len(results_data), num_plots_aim)
            if num_plots_to_show <= 0: 
                print(f"No singular value data points available for {target_layer_name}.")
                continue
                
            ncols = math.ceil(math.sqrt(num_plots_to_show * 1.2))
            nrows = math.ceil(num_plots_to_show / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5), squeeze=False)
            axes = axes.flatten()

            if num_plots_to_show == 1: 
                indices_to_plot = [0]
            else:
                # Ensure indices are unique and cover the range reasonably
                indices_to_plot = np.linspace(0, len(results_data) - 1, num_plots_to_show, dtype=int)
                indices_to_plot = np.unique(indices_to_plot)
                num_plots_to_show = len(indices_to_plot) # Update actual number of plots

            print(f"Plotting {num_plots_to_show} singular value distribution snapshots at batches: {[batch_numbers[i] for i in indices_to_plot if i < len(batch_numbers)]}")

            # Determine common plot range for all plots
            max_sv = 0.0
            all_sv_list = []
            valid_indices = []
            for idx in indices_to_plot:
                 if idx < len(results_data) and results_data[idx] is not None and len(results_data[idx]) > 0:
                      all_sv_list.append(results_data[idx])
                      max_val = np.max(results_data[idx])
                      if np.isfinite(max_val) and max_val > 0:
                           max_sv = max(max_sv, max_val)
                      valid_indices.append(idx) # Keep track of indices with valid data

            # Recalculate plot layout if some data was invalid
            num_plots_to_show = len(valid_indices)
            if num_plots_to_show <= 0:
                 print(f"No valid singular value data points available for plotting {target_layer_name}.")
                 continue

            common_plot_range = (0, 3.0) # Default
            if all_sv_list:
                 all_selected_sv = np.concatenate(all_sv_list)
                 min_sv = np.min(all_selected_sv)
                 max_sv = np.max(all_selected_sv)
                 
                 # For singular values, we typically focus on range from 0 to max
                 # with some padding
                 plot_min = 0  # Singular values are non-negative
                 plot_max = max(max_sv * 1.1, 3.0)  # Add some padding and ensure reasonable minimum range
                 common_plot_range = (plot_min, plot_max)
                 print(f"  Determined common plot range: ({common_plot_range[0]:.2f}, {common_plot_range[1]:.2f}) for singular values")

            plot_count = 0
            for i, data_idx in enumerate(valid_indices): # Iterate through valid indices only
                ax = axes[plot_count] # Use plot_count for indexing axes
                batch_num = batch_numbers[data_idx] if data_idx < len(batch_numbers) else "Unknown"
                sv_values = results_data[data_idx]

                # Calculate quarter-circle law theoretical density
                # For normalized matrices
                R = 1.0  # R=1 for quarter-circle law
                x_quarter_circle = np.linspace(0, 2 * R, 300)
                # Quarter-circle law formula
                quarter_circle_density = (2 / (np.pi * R**2)) * np.sqrt(np.maximum(0, (2*R)**2 - x_quarter_circle**2)) / 2
                
                # Plot histogram of singular values
                num_bins = max(min(len(sv_values)//10, 75), 30) # Adaptive binning
                ax.hist(sv_values, bins=num_bins, density=True, alpha=0.75, label='Empirical density', range=common_plot_range)
                
                # Plot quarter-circle law
                ax.plot(x_quarter_circle, quarter_circle_density, 'r--', linewidth=1.5, label='Quarter-Circle Law')
                
                ax.set_xlabel("Normalized Singular Value (σ)")
                ax.set_ylabel("Density")
                ax.set_title(f"Batch: {batch_num}")
                ax.legend(fontsize='small')
                ax.grid(True, alpha=0.5)
                ax.set_xlim(common_plot_range)
                ax.set_ylim(bottom=0)
                plot_count += 1

            # Turn off unused axes
            for j in range(plot_count, len(axes)):
                axes[j].axis('off')

            layer_shape = layer_shapes.get(target_layer_name, "unknown shape")
            layer_info_str = f"Layer '{target_layer_name}', Shape: {layer_shape}"
            fig.suptitle(f"Singular Value Distribution\n{matrix_title_str}\n{layer_info_str}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.91]) # Adjust rect to prevent title overlap
            
            # Save plot to file
            plot_filename = f"{timestamp}_{target_layer_name.replace('.', '_')}_singular_values.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved singular value plot for {target_layer_name} to: {plot_path}")

print("\nDone.")

warnings.filterwarnings("default", category=RuntimeWarning)