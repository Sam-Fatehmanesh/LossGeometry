import numpy as np
import matplotlib.pyplot as plt
import os
import math
from skrmt.ensemble.spectral_law import MarchenkoPasturDistribution
from skrmt.ensemble.spectral_law import TracyWidomDistribution

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
            batch_numbers (list or array): Batch numbers
            loss_values (list or array): Loss values
            plot_title (str): Optional custom title for the plot
        """
        # Check if loss_values exists and has elements
        if loss_values is None or len(loss_values) == 0:
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
        if results_data is None or len(results_data) == 0:
            print(f"No eigenvalues data for {layer_name}. Skipping plot.")
            return
        
        run_info = f" (Averaged over {runs} runs)" if runs > 1 else ""
        print(f"Plotting Spectral Density evolution for {layer_name} {matrix_description}{run_info}...")
        
        num_plots_aim = 20
        num_plots_to_show = min(len(results_data), num_plots_aim)
        if num_plots_to_show <= 0:
            print(f"No density data points available for {layer_name}.")
            return
            
        # Create a grid for the plots
        ncols = 2
        nrows = math.ceil(num_plots_to_show / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False)
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
             # For circular law, eigenvalues are in a disk of radius R
             radius_scale = max(max_R_fit, 1.0)  # Use at least 1.0 if std is tiny
             plot_min = min(data_min * 1.1 if data_min < 0 else data_min * 0.9, -radius_scale * 1.05, -0.1)
             plot_max = max(data_max * 1.1 if data_max > 0 else data_max * 0.9, radius_scale * 1.05, 0.1)
             # Ensure range is reasonable, e.g., not excessively large if data is clustered
             plot_min = max(plot_min, -2.0 * radius_scale)  # Limit how far left
             plot_max = min(plot_max, 2.0 * radius_scale)  # Limit how far right
             if plot_max - plot_min < 1e-6:  # Avoid zero range
                  plot_min -= 0.5
                  plot_max += 0.5
             common_plot_range = (plot_min, plot_max)
             print(f"  Determined common plot range: ({common_plot_range[0]:.2f}, {common_plot_range[1]:.2f}) based on R_fit up to {max_R_fit:.2f}")

        plot_count = 0
        for i, data_idx in enumerate(valid_indices):  # Iterate through valid indices only
            if plot_count >= len(axes):
                break  # Safety check
                
            ax = axes[plot_count]  # Use plot_count for indexing axes
            batch_num = batch_numbers[data_idx] if data_idx < len(batch_numbers) else "Unknown"
            eigenvalues = results_data[data_idx]
            sigma_empirical = np.std(eigenvalues)
            
            # Get matrix dimensions for proper scaling
            if layer_shape is not None:
                m, n = layer_shape
            else:
                # Default shape if not available
                m, n = (1024, 1024)
            
            # Determine if we're dealing with a square matrix
            is_square = (m == n)
            
            # Use empirical std for R_fit, ensuring it's positive for the formula
            R_fit = 2 * sigma_empirical if (np.isfinite(sigma_empirical) and sigma_empirical > 1e-9) else 2.0
            
            # STEP 1: Create a regular histogram (NOT density-normalized)
            num_bins = max(min(len(eigenvalues) // 10, 512), 30)  # Adaptive binning
            counts, bin_edges, _ = ax.hist(eigenvalues, bins=num_bins, density=False, 
                                        alpha=0.7, color='royalblue', label='Eigenvalue Histogram',
                                        range=common_plot_range)
            
            # Calculate histogram bin widths for proper scaling
            bin_width = bin_edges[1] - bin_edges[0]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # STEP 2: Plot the circular law (quarter circle for real eigenvalues)
            # For proper scaling with non-normalized histogram, we need to scale by total count and bin width
            total_count = len(eigenvalues)
            
            # Create x-values for circular law
            x_circular_fit = np.linspace(-R_fit, R_fit, 300)
            
            # Calculate circular law density (unnormalized)
            # ρ_Re(x) = (2/(π*R^2))*sqrt(R^2 - x^2) for |x| <= R
            sqrt_term_fit = np.sqrt(np.maximum(0, R_fit**2 - x_circular_fit**2))
            # Raw density function
            rho_circular_fit = (2.0 / (np.pi * max(R_fit**2, 1e-15))) * sqrt_term_fit
            
            # Scale to match histogram height
            # For a histogram with N samples and bin width w, the scaling factor is N*w
            circular_scaling = total_count * bin_width
            scaled_circular = rho_circular_fit * circular_scaling
            
            # Plot the scaled circular law
            ax.plot(x_circular_fit, scaled_circular, 'r--', linewidth=1.8, 
                   label=f'Circular Law (R={R_fit:.2f})')
            
            # STEP 3: Add Tracy-Widom distribution for largest eigenvalue
            tw_dist = TracyWidomDistribution(beta=1)
            
            # Calculate edge position and scaling based on matrix dimensions
            if is_square:
                # For square matrices, use formula from image: λ_{(k)} - 4n
                edge_position = 1.0  # For normalized eigenvalues
                scale_factor = 2.0**(4/3) * n**(-1/3)  # Adjusted for normalization
            else:
                # For rectangular matrices, use formula: λ_{(k)} - (√m + √n)²
                sqrt_m = np.sqrt(m)
                sqrt_n = np.sqrt(n)
                edge_position = (sqrt_m + sqrt_n)**2
                scale_factor = (sqrt_m + sqrt_n) * ((m**(-1/2)) + (n**(-1/2)))**(1/3)
                
                # Adjust for normalized eigenvalues
                edge_position = R_fit  # Using radius of circular law
            
            print(f"  TW edge: {edge_position:.4f}, scale: {scale_factor:.4e}")
            
            # Create range for displaying Tracy-Widom distribution
            x_tw = np.linspace(edge_position - 4*scale_factor, edge_position + 4*scale_factor, 1000)
            
            # Calculate Tracy-Widom PDF values
            tw_args = (x_tw - edge_position) / scale_factor
            pdf_tw = tw_dist.pdf(tw_args) / scale_factor  # Proper normalization
            
            # Scale TW to match histogram height
            # First find the maximum height near the edge
            max_pdf_index = np.argmax(pdf_tw)
            max_pdf_value = pdf_tw[max_pdf_index]
            
            # Find histogram height near the edge
            edge_bin_indices = np.where((bin_centers > edge_position - scale_factor) & 
                                       (bin_centers < edge_position + scale_factor))[0]
            hist_height_near_edge = np.max(counts[edge_bin_indices]) if len(edge_bin_indices) > 0 else np.max(counts)
            
            # Scale TW curve to approximately match histogram height
            tw_scaling = hist_height_near_edge / max_pdf_value if max_pdf_value > 0 else 1.0
            scaled_tw = pdf_tw * tw_scaling
            
            # Create a horizontal line at the Tracy-Widom edge
            # This visually shows where the edge of the spectrum should be
            ax.axhline(y=0, color='green', linestyle='-', alpha=0.3)
            
            # Plot Tracy-Widom at the edge
            ax.plot(x_tw, scaled_tw, 'g-', linewidth=2.0,
                   label=f'TW (edge={edge_position:.2f}, scale={scale_factor:.2e})')
            
            # Add the formula from the image as text annotation
            if is_square:
                formula_text = r"$A_k(\tau) \approx \frac{\lambda_{(k)}(t_{scaled}) - 4n}{(2^{2/3}n^{1/3})}$"
            else:
                formula_text = r"$A_k(\tau) \approx \frac{\lambda_{(k)}(t_{scaled}) - (\sqrt{m}+\sqrt{n})^2}{(\sqrt{m}+\sqrt{n})(m^{-1/2}+n^{-1/2})^{1/3}}$"
            
            ax.text(0.5, 0.93, formula_text, transform=ax.transAxes, 
                  horizontalalignment='center', fontsize=10, 
                  bbox=dict(facecolor='white', alpha=0.7))
            
            ax.set_xlabel("Normalized Eigenvalue (λ)")
            ax.set_ylabel("Count")
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
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to prevent title overlap
        
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
                num_bins = 1024
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
        
    def plot_singular_values(self, layer_name, layer_shape, results_data, batch_numbers, matrix_description, runs=1, num_epochs=1):
        """
        Plot singular value evolution
        
        Args:
            layer_name (str): Name of the layer
            layer_shape (tuple): Shape of the layer (m, n)
            results_data (list): List of singular value arrays
            batch_numbers (list): Batch numbers
            matrix_description (str): Description of the matrix type
            runs (int): Number of runs the results are averaged over
            num_epochs (int): Number of epochs to plot
        """
        if results_data is None or len(results_data) == 0:
            print(f"No singular values data for {layer_name}. Skipping plot.")
            return
            
        run_info = f" (Averaged over {runs} runs)" if runs > 1 else ""
        print(f"Plotting Singular Value distribution for {layer_name} {matrix_description}{run_info}...")
        
        # Determine number of snapshots: always plot 20 equally spaced snapshots
        num_plots_aim = 20
        num_plots_to_show = min(len(results_data), num_plots_aim)
        if num_plots_to_show <= 0:
            print(f"No singular value data points available for {layer_name}.")
            return
        # Create a grid: one axis per snapshot, flattened
        ncols = 2
        nrows = math.ceil(num_plots_to_show / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False)
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
             # Pad range
             plot_min = 0
             plot_max = max(max_sv * 1.1, 3.0)
             common_plot_range = (plot_min, plot_max)
             print(f"  Determined common plot range: ({common_plot_range[0]:.2f}, {common_plot_range[1]:.2f}) for singular values")

        # Iterate and plot on each axis
        plot_count = 0
        for data_idx in valid_indices:
            if plot_count >= len(axes):
                break
            ax = axes[plot_count]
            batch_num = batch_numbers[data_idx] if data_idx < len(batch_numbers) else "Unknown"
            sv_values = results_data[data_idx]
            # Dimensions and MP parameters
            if layer_shape is not None:
                m, n = layer_shape
                p = min(m, n); n_max = max(m, n)
                ratio = p / n_max; is_square = (m == n)
            else:
                ratio = 1.0; m = n_max = 1024; p = 1024; is_square = True
            total_count = len(sv_values)
            # STEP 1: Histogram of normalized singular values (raw counts)
            num_bins = max(min(total_count // 10, 100), 30)
            counts, bin_edges, _ = ax.hist(sv_values, bins=num_bins, density=False,
                                           alpha=0.7, color='royalblue',
                                           label='Singular Value Histogram',
                                           range=common_plot_range)
            bin_width = bin_edges[1] - bin_edges[0]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # STEP 2: Scaled MP distribution
            mp_dist = MarchenkoPasturDistribution(beta=1, ratio=ratio, sigma=1.0)
            x_sv = np.linspace(common_plot_range[0], common_plot_range[1], 1000)
            sv_density = np.array([2*s*mp_dist.pdf(s*s) if s>0 else 0 for s in x_sv])
            scaled_mp = sv_density * total_count * bin_width
            ax.plot(x_sv, scaled_mp, 'r--', linewidth=1.5,
                    label=f'MP (λ={ratio:.2f})')
            # STEP 3: Tracy-Widom distribution for largest singular value
            tw_dist = TracyWidomDistribution(beta=1)
            bulk_edge = 1.0 + np.sqrt(ratio)

            edge_position = 2.0
            scale_factor = 0.5*(np.sqrt(m) + np.sqrt(n))**(-1/3) #bulk_edge * (n_max**(-2/3)) * 0.5
            
            print(f"  TW edge: {edge_position:.4f}, scale: {scale_factor:.4e}")
            x_tw = np.linspace(edge_position - 4*scale_factor,
                               edge_position + 4*scale_factor, 300)
            tw_args = (x_tw - edge_position) / scale_factor
            pdf_tw = tw_dist.pdf(tw_args) / scale_factor
            # scale TW curve to histogram height
            max_pdf = np.max(pdf_tw)
            edge_idx = np.where((bin_centers > edge_position - scale_factor) &
                                (bin_centers < edge_position + scale_factor))[0]
            hist_edge = np.max(counts[edge_idx]) if len(edge_idx)>0 else np.max(counts)
            tw_scale = hist_edge / max_pdf if max_pdf>0 else 1.0
            ax.plot(x_tw, pdf_tw * tw_scale, 'g-', linewidth=2.0,
                    label=f'TW (edge={edge_position:.2f}, scale={scale_factor:.2e})')

            # Annotation of TW formula
            if is_square:
                formula_text = r"$A_k(\tau) \approx \frac{\sigma_{(k)} - 2\sqrt{n}}{2^{1/3}n^{1/6}}$"
            else:
                formula_text = r"$A_k(\tau) \approx \frac{\sigma_{(k)} - (\sqrt{m}+\sqrt{n})}{(\sqrt{m}+\sqrt{n})^{1/2}(m^{-1/2}+n^{-1/2})^{1/3}}$"
            ax.text(0.5, 0.93, formula_text, transform=ax.transAxes,
                    horizontalalignment='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))
            # Styling
            ax.set_xlabel("Normalized Singular Value (σ)")
            ax.set_ylabel("Count")
            ax.set_title(f"Batch: {batch_num}")
            ax.legend(fontsize='small')
            ax.grid(True, alpha=0.5)
            ax.set_xlim(common_plot_range)
            ax.set_ylim(bottom=0)
            plot_count += 1
        # Turn off unused axes
        for j in range(plot_count, len(axes)):
            axes[j].axis('off')
        # Title and save
        layer_info_str = f"Layer '{layer_name}', Shape: {layer_shape}"
        fig.suptitle(f"Singular Value Distribution{run_info}\n{matrix_description}\n{layer_info_str}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f"{self.timestamp}_{layer_name.replace('.', '_')}_singular_values.png"
        plot_path = os.path.join(self.output_dir, plot_filename)
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved singular value plot for {layer_name} to: {plot_path}")
        plt.close() 