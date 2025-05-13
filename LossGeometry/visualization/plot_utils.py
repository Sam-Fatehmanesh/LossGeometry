import numpy as np
import matplotlib.pyplot as plt
import os
import math

class AnalysisPlotter:
    """
    Class for plotting spectral analysis results
    """
    def __init__(self, output_dir, timestamp, dpi=300):
        """
        Initialize the plotter
        
        Args:
            output_dir (str): Output directory for plots
            timestamp (str): Timestamp for file naming
            dpi (int): DPI for saved plots
        """
        self.output_dir = output_dir
        self.timestamp = timestamp
        self.dpi = dpi
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
    
    def plot_loss(self, batch_numbers, loss_values, plot_title=None):
        """
        Plot loss over batches
        
        Args:
            batch_numbers (list): Batch numbers
            loss_values (list): Loss values
            plot_title (str): Optional custom title for the plot
        """
        if not loss_values:
            print("No loss values to plot.")
            return
        
        print("\nPlotting loss per batch...")
        plt.figure(figsize=(10, 6))
        plt.plot(batch_numbers, loss_values, 'b-')
        plt.xlabel('Batch Number')
        plt.ylabel('Loss')
        plt.title(plot_title if plot_title else 'MNIST Training Loss')
        plt.grid(True)
        
        # Add moving average to smooth the curve
        if len(loss_values) > 20:
            window_size = min(20, len(loss_values) // 5)
            moving_avg = np.convolve(
                loss_values, 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            # Plot moving average starting at the window_size-1 point
            plt.plot(
                batch_numbers[window_size-1:window_size-1+len(moving_avg)], 
                moving_avg, 'r-', linewidth=2, alpha=0.7, 
                label=f'Moving Avg (window={window_size})'
            )
            plt.legend()
        
        # Save loss plot
        plot_filename = f"{self.timestamp}_mnist_loss_curve.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.tight_layout()
        print(f"Saved loss plot to: {plot_path}")
        plt.close()
    
    def plot_spectral_density(self, layer_name, layer_shape, results_data, batch_numbers, matrix_description, runs=1):
        """
        Plot spectral density evolution
        
        Args:
            layer_name (str): Name of the layer
            layer_shape (tuple): Shape of the layer
            results_data (list): List of eigenvalue arrays
            batch_numbers (list): Batch numbers
            matrix_description (str): Description of the matrix type
            runs (int): Number of runs the results are averaged over
        """
        if not results_data:
            print(f"No eigenvalues data for {layer_name}. Skipping plot.")
            return
        
        run_info = f" (Averaged over {runs} runs)" if runs > 1 else ""
        print(f"Plotting Spectral Density evolution for {layer_name} {matrix_description}{run_info}...")
        
        num_plots_aim = 6
        num_plots_to_show = min(len(results_data), num_plots_aim)
        if num_plots_to_show <= 0:
            print(f"No density data points available for {layer_name}.")
            return
            
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
            num_plots_to_show = len(indices_to_plot)  # Update actual number of plots

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
                  valid_indices.append(idx)  # Keep track of indices with valid data

        # Recalculate plot layout if some data was invalid
        num_plots_to_show = len(valid_indices)
        if num_plots_to_show <= 0:
             print(f"No valid density data points available for plotting {layer_name}.")
             return

        common_plot_range = (-2.5, 2.5)  # Default
        if all_eigenvalues_list:
             all_selected_eigenvalues = np.concatenate(all_eigenvalues_list)
             data_min = np.min(all_selected_eigenvalues)
             data_max = np.max(all_selected_eigenvalues)
             # Wigner semicircle extends from -2R to 2R. Use max_R_fit for scale.
             radius_scale = max(max_R_fit, 1.0)  # Use at least 1.0 if std is tiny
             plot_min = min(data_min * 1.1 if data_min < 0 else data_min * 0.9, -2.0 * radius_scale * 1.05, -0.1)
             plot_max = max(data_max * 1.1 if data_max > 0 else data_max * 0.9, 2.0 * radius_scale * 1.05, 0.1)
             # Ensure range is reasonable, e.g., not excessively large if data is clustered
             plot_min = max(plot_min, -5.0 * radius_scale)  # Limit how far left
             plot_max = min(plot_max, 5.0 * radius_scale)  # Limit how far right
             if plot_max - plot_min < 1e-6:  # Avoid zero range
                  plot_min -= 0.5
                  plot_max += 0.5
             common_plot_range = (plot_min, plot_max)
             print(f"  Determined common plot range: ({common_plot_range[0]:.2f}, {common_plot_range[1]:.2f}) based on R_fit up to {max_R_fit:.2f}")

        plot_count = 0
        for i, data_idx in enumerate(valid_indices):  # Iterate through valid indices only
            ax = axes[plot_count]  # Use plot_count for indexing axes
            batch_num = batch_numbers[data_idx] if data_idx < len(batch_numbers) else "Unknown"
            eigenvalues = results_data[data_idx]

            sigma_empirical = np.std(eigenvalues)
            # Use empirical std for R_fit, ensuring it's positive for the formula
            R_fit = sigma_empirical if (np.isfinite(sigma_empirical) and sigma_empirical > 1e-9) else 1.0  # Use 1.0 as fallback

            # Wigner semicircle formula
            x_semicircle_fit = np.linspace(-2 * R_fit, 2 * R_fit, 300)
            # Prevent sqrt of negative numbers due to float precision
            sqrt_term_fit = np.sqrt(np.maximum(0, (2 * R_fit)**2 - x_semicircle_fit**2))
            # Avoid division by zero if R_fit is extremely small
            rho_semicircle_fit = (1 / (np.pi * max(R_fit**2, 1e-15) * 2.0)) * sqrt_term_fit

            num_bins = max(min(len(eigenvalues)//10, 75), 30)  # Adaptive binning
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

        layer_info_str = f"Layer '{layer_name}', Shape: {layer_shape}"
        title_with_runs = f"Spectral Density Evolution{run_info}\n{matrix_description}\n{layer_info_str}"
        fig.suptitle(title_with_runs, fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.91])  # Adjust rect to prevent title overlap
        
        # Save plot to file
        plot_filename = f"{self.timestamp}_{layer_name.replace('.', '_')}_spectral_density.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved spectral density plot for {layer_name} to: {plot_path}")
        plt.close()
    
    def plot_level_spacing(self, layer_name, layer_shape, std_dev_list, last_spacings, batch_numbers, matrix_description, runs=1):
        """
        Plot level spacing evolution and final P(s)
        
        Args:
            layer_name (str): Name of the layer
            layer_shape (tuple): Shape of the layer
            std_dev_list (list): List of std dev values
            last_spacings (numpy.ndarray): Array of last normalized spacings
            batch_numbers (list): Batch numbers
            matrix_description (str): Description of the matrix type
            runs (int): Number of runs the results are averaged over
        """
        if not std_dev_list:
            print(f"No level spacing data for {layer_name}. Skipping plot.")
            return
            
        run_info = f" (Averaged over {runs} runs)" if runs > 1 else ""
        print(f"Plotting Level Spacing evolution for {layer_name} {matrix_description}{run_info}...")
        
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
                if not np.isfinite(min_s_val) or min_s_val <= 1e-10: min_s_val = 1e-5  # Set a small positive floor
                if not np.isfinite(max_s_val) or max_s_val <= min_s_val: max_s_val = min_s_val * 1000  # Ensure max > min

                min_log_s = np.log10(min_s_val)
                max_log_s = np.log10(max_s_val)
                num_bins = 50
                # Ensure log bins are valid
                if max_log_s <= min_log_s: max_log_s = min_log_s + 3  # Ensure range if somehow still equal
                log_bins = np.logspace(min_log_s, max_log_s, num=num_bins + 1)

                ax.hist(positive_spacings, bins=log_bins, density=True, alpha=0.75, label='Empirical P(s)')
                ax.set_xscale('log')  # Set X-axis to log scale

                # Plot reference distributions on the log scale
                s_min_plot, s_max_plot = ax.get_xlim()
                # Ensure limits are positive for logspace, using calculated min/max if plot limits are bad
                s_min_plot = max(s_min_plot, min_s_val * 0.9, 1e-6)
                s_max_plot = max(s_max_plot, max_s_val * 1.1)
                if s_max_plot <= s_min_plot: s_max_plot = s_min_plot * 1000  # Final safety

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

        layer_info_str = f"Layer '{layer_name}', Shape: {layer_shape}"
        fig.suptitle(f"Level Spacing Analysis{run_info}\n{matrix_description}\n{layer_info_str}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.91])
        
        # Save plot to file
        plot_filename = f"{self.timestamp}_{layer_name.replace('.', '_')}_level_spacing.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved level spacing plot for {layer_name} to: {plot_path}")
        plt.close()
        
    def plot_singular_values(self, layer_name, layer_shape, results_data, batch_numbers, matrix_description, runs=1):
        """
        Plot singular value distribution
        
        Args:
            layer_name (str): Name of the layer
            layer_shape (tuple): Shape of the layer
            results_data (list): List of singular value arrays
            batch_numbers (list): Batch numbers
            matrix_description (str): Description of the matrix type
            runs (int): Number of runs the results are averaged over
        """
        if not results_data:
            print(f"No singular value data for {layer_name}. Skipping plot.")
            return
            
        run_info = f" (Averaged over {runs} runs)" if runs > 1 else ""
        print(f"Plotting Singular Value distribution for {layer_name} {matrix_description}{run_info}...")
        
        num_plots_aim = 6
        num_plots_to_show = min(len(results_data), num_plots_aim)
        if num_plots_to_show <= 0:
            print(f"No singular value data points available for {layer_name}.")
            return
            
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
            num_plots_to_show = len(indices_to_plot)  # Update actual number of plots

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
                  valid_indices.append(idx)  # Keep track of indices with valid data

        # Recalculate plot layout if some data was invalid
        num_plots_to_show = len(valid_indices)
        if num_plots_to_show <= 0:
             print(f"No valid singular value data points available for plotting {layer_name}.")
             return

        common_plot_range = (0, 3.0)  # Default
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
        for i, data_idx in enumerate(valid_indices):  # Iterate through valid indices only
            ax = axes[plot_count]  # Use plot_count for indexing axes
            batch_num = batch_numbers[data_idx] if data_idx < len(batch_numbers) else "Unknown"
            sv_values = results_data[data_idx]

            # Calculate quarter-circle law theoretical density
            # For normalized matrices
            R = 1.0  # R=1 for quarter-circle law
            x_quarter_circle = np.linspace(0, 2 * R, 300)
            # Quarter-circle law formula
            quarter_circle_density = (2 / (np.pi * R**2)) * np.sqrt(np.maximum(0, (2*R)**2 - x_quarter_circle**2)) / 2
            
            # Plot histogram of singular values
            num_bins = max(min(len(sv_values)//10, 75), 30)  # Adaptive binning
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

        layer_info_str = f"Layer '{layer_name}', Shape: {layer_shape}"
        fig.suptitle(f"Singular Value Distribution{run_info}\n{matrix_description}\n{layer_info_str}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.91])  # Adjust rect to prevent title overlap
        
        # Save plot to file
        plot_filename = f"{self.timestamp}_{layer_name.replace('.', '_')}_singular_values.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved singular value plot for {layer_name} to: {plot_path}")
        plt.close() 