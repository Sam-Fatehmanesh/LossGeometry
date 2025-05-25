import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.signal import find_peaks
from scipy.special import gamma
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
        
        # Create organized subdirectories for different plot types
        self.subdirs = {
            'loss': os.path.join(output_dir, 'loss_plots'),
            'singular_values': os.path.join(output_dir, 'singular_value_plots'),
            'dynamics_terms': os.path.join(output_dir, 'dynamics_terms_plots'),
            'analysis_text': os.path.join(output_dir, 'analysis_text_files'),
            'tail_plots': os.path.join(output_dir, 'only_tail_plots')
        }
        
        # Create all subdirectories
        for subdir in self.subdirs.values():
            if not os.path.exists(subdir):
                os.makedirs(subdir)
                print(f"Created subdirectory: {subdir}")
        
        # Ensure matplotlib uses a non-interactive backend
        plt.switch_backend('Agg')
    
    def _compute_theoretical_distribution(self, sigma_values, eta, m, n):
        """
        Compute theoretical distribution using formula 14:
        p_σ(σ) = 2 * (1/(4η))^((m-n+3)/4) * σ^((m-n+3)/2 - 1) / Γ((m-n+3)/4) * exp(-(1/(4η))σ²)
        
        Args:
            sigma_values (array): Singular value points to evaluate
            eta (float): Learning rate
            m, n (int): Matrix dimensions
            
        Returns:
            array: Probability density values
        """
        # Formula 14 parameters
        exponent = (m - n + 3) / 4
        power_term = (m - n + 3) / 2 - 1
        
        # Adjust scale factor
        # This controls the width of the theoretical distribution
        scale_factor = 1 / (4 * eta)  #
        
        # Handle edge cases for exponent
        if exponent <= 0:
            print(f"  Warning: Invalid exponent {exponent:.3f} for dimensions m={m}, n={n}")
            return np.zeros_like(sigma_values)
        
        # Compute the distribution
        try:
            normalization = 2 * (scale_factor ** exponent) / gamma(exponent)
        except (ValueError, OverflowError) as e:
            print(f"  Warning: Gamma function error for exponent {exponent:.3f}: {e}")
            return np.zeros_like(sigma_values)
        
        # Handle potential numerical issues
        sigma_values = np.maximum(sigma_values, 1e-12)  # Avoid zero values
        
        # Compute PDF with numerical stability
        try:
            # Split the computation to avoid overflow
            log_pdf = (np.log(normalization) + 
                      power_term * np.log(sigma_values) - 
                      scale_factor * sigma_values**2)
            pdf_values = np.exp(log_pdf)
        except (OverflowError, RuntimeWarning):
            # Fallback to direct computation
            pdf_values = normalization * (sigma_values ** power_term) * np.exp(-scale_factor * sigma_values**2)
        
        # Handle any NaN or inf values
        pdf_values = np.nan_to_num(pdf_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Additional check for reasonable values
        if np.max(pdf_values) == 0:
            print(f"  Warning: Theoretical distribution is zero everywhere (eta={eta}, m={m}, n={n})")
        
        return pdf_values
    
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
        plot_path = os.path.join(self.subdirs['loss'], plot_filename)
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.tight_layout()
        print(f"Saved loss plot to: {plot_path}")
        plt.close()
    
    def plot_singular_values(self, layer_name, layer_shape, results_data, batch_numbers, matrix_description, runs=1, plot_non_normalized=True, normalization_factors=None, learning_rate=0.01):
        """
        Plot singular value evolution
        
        Args:
            layer_name (str): Name of the layer
            layer_shape (tuple): Shape of the layer (m, n)
            results_data (list): List of singular value arrays
            batch_numbers (list): Batch numbers
            matrix_description (str): Description of the matrix type
            runs (int): Number of runs the results are averaged over
            plot_non_normalized (bool): Whether to also plot non-normalized singular values (default: True)
            normalization_factors (list): List of normalization factors used (if available)
        """
        if results_data is None or len(results_data) == 0:
            print(f"No singular values data for {layer_name}. Skipping plot.")
            return
            
        run_info = f" (Averaged over {runs} runs)" if runs > 1 else ""
        print(f"Plotting Singular Value distribution for {layer_name} {matrix_description}{run_info}...")
        
        # We'll create plots for both normalized and non-normalized if requested
        plot_types = ["normalized"]
        if plot_non_normalized:
            plot_types.append("non_normalized")
            print(f"  Also creating non-normalized singular value plots")
            
        for plot_type in plot_types:
            # Determine filename suffix for this plot type
            plot_type_suffix = "_non_normalized" if plot_type == "non_normalized" else ""
            num_plots_aim = 4
            num_plots_to_show = min(len(results_data), num_plots_aim)
            if num_plots_to_show <= 0:
                print(f"No singular value data points available for {layer_name}.")
                continue
                
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
                    # For non-normalized plots, apply the reverse normalization if factors available
                    if plot_type == "non_normalized" and normalization_factors is not None and idx < len(normalization_factors):
                        sv_values = results_data[idx] * normalization_factors[idx]
                    else:
                        sv_values = results_data[idx]
                        
                    all_sv_list.append(sv_values)
                    max_val = np.max(sv_values)
                    if np.isfinite(max_val) and max_val > 0:
                        max_sv = max(max_sv, max_val)
                    valid_indices.append(idx)  # Keep track of indices with valid data

            # Recalculate plot layout if some data was invalid
            num_plots_to_show = len(valid_indices)
            if num_plots_to_show <= 0:
                print(f"No valid singular value data points available for plotting {layer_name}.")
                continue

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
                
                # For non-normalized plots, apply the reverse normalization if factors available
                if plot_type == "non_normalized" and normalization_factors is not None and data_idx < len(normalization_factors):
                    sv_values = results_data[data_idx] * normalization_factors[data_idx]
                    normalization_info = f" (Factor: {normalization_factors[data_idx]:.2e})"
                else:
                    sv_values = results_data[data_idx]
                    normalization_info = ""
                
                # Dimensions and MP parameters
                if layer_shape is not None:
                    m, n = layer_shape
                    p = min(m, n); n_max = max(m, n)
                    ratio = p / n_max; is_square = (m == n)
                else:
                    ratio = 1.0; m = n_max = 1024; p = 1024; is_square = True
                total_count = len(sv_values)
                
                # STEP 1: Histogram of singular values (raw counts)
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
                        label=f'Marchenko Pastur Distribution')
                        
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

                # Styling
                value_type = "Singular Value" if plot_type == "non_normalized" else "Normalized Singular Value (σ)"
                ax.set_xlabel(value_type)
                ax.set_ylabel("Count")
                ax.set_title(f"Batch: {batch_num}{normalization_info}")
                ax.legend(fontsize='small')
                ax.grid(True, alpha=0.5)
                ax.set_xlim(common_plot_range)
                ax.set_ylim(bottom=0)

                # Plot only-tail histogram (values beyond end of TW distribution)
                tail_cut = edge_position + 4*scale_factor
                tail_values = sv_values[sv_values > tail_cut]
                if len(tail_values) > 0:
                    fig_tail, ax_tail = plt.subplots(figsize=(8, 6))
                    tail_bins = max(min(len(tail_values) // 10, 100), 30)
                    
                    # Create histogram and get bin information
                    counts, bin_edges, patches = ax_tail.hist(tail_values, bins=tail_bins, density=False, 
                                                            alpha=0.7, color='royalblue', 
                                                            label='Tail Histogram')
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Perform peak detection on the histogram
                    # Use relative height threshold and minimum distance between peaks
                    min_peak_height = np.max(counts) * 0.1  # Peaks must be at least 10% of max height
                    min_distance = max(1, len(counts) // 20)  # Minimum distance between peaks
                    
                    peaks, peak_properties = find_peaks(counts, 
                                                       height=min_peak_height,
                                                       distance=min_distance,
                                                       prominence=np.max(counts) * 0.05)
                    
                    # Plot detected peaks
                    if len(peaks) > 0:
                        peak_positions = bin_centers[peaks]
                        peak_heights = counts[peaks]
                        
                        # Mark peaks with red dots
                        ax_tail.scatter(peak_positions, peak_heights, 
                                      color='red', s=50, zorder=5, 
                                      label=f'Detected Peaks ({len(peaks)})')
                        
                        # Add vertical lines at peak positions
                        for i, (pos, height) in enumerate(zip(peak_positions, peak_heights)):
                            ax_tail.axvline(x=pos, color='red', linestyle='--', alpha=0.6)
                            # Annotate peak with its position
                            ax_tail.annotate(f'Peak {i+1}\nσ={pos:.3f}', 
                                           xy=(pos, height), 
                                           xytext=(5, 10), 
                                           textcoords='offset points',
                                           fontsize=8, 
                                           bbox=dict(boxstyle='round,pad=0.3', 
                                                   facecolor='yellow', alpha=0.7),
                                           arrowprops=dict(arrowstyle='->', 
                                                         connectionstyle='arc3,rad=0'))
                        
                        print(f"  Detected {len(peaks)} peaks in tail for {layer_name} batch {batch_num}")
                        print(f"  Peak positions: {[f'{pos:.3f}' for pos in peak_positions]}")
                    else:
                        print(f"  No significant peaks detected in tail for {layer_name} batch {batch_num}")
                    
                    # Styling
                    ax_tail.set_xlabel(f"Singular Value (σ > {tail_cut:.2f})")
                    ax_tail.set_ylabel("Count")
                    ax_tail.set_title(f"Singular Value Tail with Peak Detection\n{layer_name} - Batch {batch_num}")
                    ax_tail.legend()
                    ax_tail.grid(True, alpha=0.3)
                    
                    # Add theoretical distributions at detected peaks
                    if len(peaks) > 0 and layer_shape is not None:
                        m, n = layer_shape
                        
                        # For each detected peak, plot a theoretical distribution centered at that peak
                        for i, (peak_pos, peak_height) in enumerate(zip(peak_positions, peak_heights)):
                            # Create a wider range centered around the detected peak for theoretical curve
                            # This ensures we capture the full theoretical distribution shape
                            tail_range = np.max(tail_values) - np.min(tail_values)
                            curve_width = max(tail_range, peak_pos * 0.5)  # At least 50% of peak position or tail range
                            
                            # Create sigma range centered on the detected peak with sufficient width
                            sigma_start = max(0.1, peak_pos - curve_width)
                            sigma_end = peak_pos + curve_width
                            sigma_range = np.linspace(sigma_start, sigma_end, 1000)
                            
                            # Compute theoretical distribution using formula 14
                            theoretical_pdf = self._compute_theoretical_distribution(sigma_range, learning_rate, m, n)
                            
                            # Scale the theoretical curve to match the histogram scale
                            # We want the peak of the theoretical curve to match the detected peak height
                            if np.max(theoretical_pdf) > 0:
                                # Find the theoretical peak position in the full range
                                theoretical_peak_idx = np.argmax(theoretical_pdf)
                                theoretical_peak_pos = sigma_range[theoretical_peak_idx]
                                
                                # Shift the curve to center it at the detected peak
                                # This creates a new sigma array where the theoretical peak aligns with detected peak
                                shift_amount = peak_pos - theoretical_peak_pos
                                shifted_sigma = sigma_range + shift_amount
                                
                                # Scale to match peak height
                                scale_factor = peak_height / np.max(theoretical_pdf)
                                scaled_pdf = theoretical_pdf * scale_factor
                                
                                # Only plot points that are within the visible range
                                visible_mask = (shifted_sigma >= np.min(tail_values) * 0.9) & (shifted_sigma <= np.max(tail_values) * 1.1)
                                if np.any(visible_mask):
                                    ax_tail.plot(shifted_sigma[visible_mask], scaled_pdf[visible_mask], 
                                               '--', linewidth=2, alpha=0.8,
                                               label=f'Theory Peak {i+1} (η={learning_rate})',
                                               color=plt.cm.Set1(i % 9))
                                    
                                    print(f"  Plotted theoretical curve for peak {i+1}: center={peak_pos:.3f}, shift={shift_amount:.3f}")
                                else:
                                    print(f"  Warning: Theoretical curve for peak {i+1} is outside visible range")
                        
                        # Update legend to include theoretical curves
                        ax_tail.legend()
                    
                    # Save the enhanced tail plot
                    tail_filename = f"{self.timestamp}_{layer_name.replace('.', '_')}_singular_values{plot_type_suffix}_only_tail_batch{batch_num}.png"
                    tail_path = os.path.join(self.subdirs['tail_plots'], tail_filename)
                    fig_tail.savefig(tail_path, dpi=self.dpi, bbox_inches='tight')
                    print(f"Saved tail-only singular value histogram with peak detection for {layer_name} batch {batch_num} to: {tail_path}")
                    
                    # Create a detailed peak analysis text file
                    if len(peaks) > 0:
                        peak_analysis_filename = f"{self.timestamp}_{layer_name.replace('.', '_')}_tail_peaks_batch{batch_num}.txt"
                        peak_analysis_path = os.path.join(self.subdirs['tail_plots'], peak_analysis_filename)
                        
                        with open(peak_analysis_path, 'w') as peak_file:
                            peak_file.write(f"Peak Detection Analysis for {layer_name} - Batch {batch_num}\n")
                            peak_file.write(f"{'='*60}\n\n")
                            peak_file.write(f"Tail cutoff: σ > {tail_cut:.4f}\n")
                            peak_file.write(f"Total tail values: {len(tail_values)}\n")
                            peak_file.write(f"Tail range: [{np.min(tail_values):.4f}, {np.max(tail_values):.4f}]\n\n")
                            
                            peak_file.write(f"Peak Detection Parameters:\n")
                            peak_file.write(f"  Minimum peak height: {min_peak_height:.2f} (10% of max)\n")
                            peak_file.write(f"  Minimum distance between peaks: {min_distance} bins\n")
                            peak_file.write(f"  Prominence threshold: {np.max(counts) * 0.05:.2f}\n\n")
                            
                            peak_file.write(f"Detected Peaks ({len(peaks)}):\n")
                            peak_file.write("-" * 40 + "\n")
                            peak_file.write("Peak#\tPosition\tHeight\tProminence\n")
                            
                            for i, (peak_idx, pos, height) in enumerate(zip(peaks, peak_positions, peak_heights)):
                                prominence = peak_properties['prominences'][i] if 'prominences' in peak_properties else 'N/A'
                                peak_file.write(f"{i+1}\t{pos:.4f}\t{height:.0f}\t{prominence:.2f}\n")
                            
                            # Statistical summary
                            peak_file.write(f"\nStatistical Summary:\n")
                            peak_file.write("-" * 20 + "\n")
                            peak_file.write(f"Mean peak position: {np.mean(peak_positions):.4f}\n")
                            peak_file.write(f"Std peak position: {np.std(peak_positions):.4f}\n")
                            peak_file.write(f"Peak spacing (if >1 peak): ")
                            if len(peak_positions) > 1:
                                spacings = np.diff(sorted(peak_positions))
                                peak_file.write(f"{np.mean(spacings):.4f} ± {np.std(spacings):.4f}\n")
                            else:
                                peak_file.write("N/A (single peak)\n")
                        
                        print(f"Saved peak analysis details to: {peak_analysis_path}")
                    
                    plt.close(fig_tail)

                plot_count += 1
                
            # Turn off unused axes
            for j in range(plot_count, len(axes)):
                axes[j].axis('off')
                
            # Title and save
            layer_info_str = f"Layer '{layer_name}', Shape: {layer_shape}"
            plot_type_title = "Non-Normalized" if plot_type == "non_normalized" else "Normalized"
            fig.suptitle(f"{plot_type_title} Singular Value Distribution{run_info}\n{matrix_description}\n{layer_info_str}", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save with appropriate naming
            plot_filename = f"{self.timestamp}_{layer_name.replace('.', '_')}_singular_values{plot_type_suffix}.png"
            plot_path = os.path.join(self.subdirs['singular_values'], plot_filename)
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved {plot_type_title.lower()} singular value plot for {layer_name} to: {plot_path}")
            plt.close() 

    def plot_singular_value_dynamics_terms(self, layer_name, layer_shape, singular_values_data, 
                                         loss_gradients, batch_numbers, matrix_description, 
                                         runs=1, gradient_noise=None):
        """
        Plot the magnitude of key terms in singular value dynamics:
        - Loss gradient magnitude
        - 1/sigma_k (inverse singular values)
        - sigma_k^2 / (sigma_k^2 - sigma_j^2) (interaction terms)
        - Gradient noise (mini-batch variance)
        
        Args:
            layer_name (str): Name of the layer
            layer_shape (tuple): Shape of the layer (m, n)
            singular_values_data (list): List of singular value arrays for each batch
            loss_gradients (list): List of loss gradient magnitudes for each batch
            batch_numbers (list): Batch numbers
            matrix_description (str): Description of the matrix type
            runs (int): Number of runs the results are averaged over
            gradient_noise (list): List of gradient noise estimates for each batch
        """
        if singular_values_data is None or len(singular_values_data) == 0:
            print(f"No singular values data for {layer_name}. Skipping dynamics terms plot.")
            return
            
        if loss_gradients is None or len(loss_gradients) == 0:
            print(f"No loss gradient data for {layer_name}. Skipping dynamics terms plot.")
            return
            
        run_info = f" (Averaged over {runs} runs)" if runs > 1 else ""
        print(f"Plotting Singular Value Dynamics Terms for {layer_name} {matrix_description}{run_info}...")
        
        # Ensure we have matching data lengths
        min_length = min(len(singular_values_data), len(loss_gradients), len(batch_numbers))
        singular_values_data = singular_values_data[:min_length]
        loss_gradients = loss_gradients[:min_length]
        batch_numbers = batch_numbers[:min_length]
        
        # Handle gradient noise data
        if gradient_noise is not None:
            gradient_noise = gradient_noise[:min_length]
        
        # Calculate terms for each batch
        inverse_sigma_means = []
        interaction_term_means = []
        
        for i, sv_array in enumerate(singular_values_data):
            if sv_array is None or len(sv_array) == 0:
                inverse_sigma_means.append(np.nan)
                interaction_term_means.append(np.nan)
                continue
                
            # Sort singular values in descending order
            sv_sorted = np.sort(sv_array)[::-1]
            
            # Calculate 1/sigma_k terms (avoid division by zero)
            sv_nonzero = sv_sorted[sv_sorted > 1e-12]
            if len(sv_nonzero) > 0:
                inverse_sigma = 1.0 / sv_nonzero
                inverse_sigma_means.append(np.mean(inverse_sigma))
            else:
                inverse_sigma_means.append(np.nan)
            
            # Calculate interaction terms: sigma_k^2 / (sigma_k^2 - sigma_j^2)
            # This represents the coupling between different singular value modes
            interaction_terms = []
            for k in range(len(sv_nonzero)):
                sigma_k = sv_nonzero[k]
                for j in range(len(sv_nonzero)):
                    if k != j:
                        sigma_j = sv_nonzero[j]
                        denominator = sigma_k**2 - sigma_j**2
                        if abs(denominator) > 1e-12:  # Avoid division by zero
                            interaction_term = sigma_k**2 / abs(denominator)
                            interaction_terms.append(interaction_term)
            
            if len(interaction_terms) > 0:
                # Cap extremely large values for visualization
                interaction_terms = np.array(interaction_terms)
                interaction_terms = np.clip(interaction_terms, 0, 1e6)
                interaction_term_means.append(np.mean(interaction_terms))
            else:
                interaction_term_means.append(np.nan)
        
        # Create a single plot (bottom-right comparison only)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot all terms on the same scale for comparison
        if np.any(~np.isnan(inverse_sigma_means)):
            ax.semilogy(batch_numbers, loss_gradients, 'b-', linewidth=2, label='Loss Gradient', alpha=0.8)
            
            valid_inv = ~np.isnan(inverse_sigma_means)
            if np.any(valid_inv):
                ax.semilogy(np.array(batch_numbers)[valid_inv], 
                           np.array(inverse_sigma_means)[valid_inv], 
                           'r-', linewidth=2, label='Mean(1/σ_k)', alpha=0.8)
            
            valid_int = ~np.isnan(interaction_term_means)
            if np.any(valid_int):
                ax.semilogy(np.array(batch_numbers)[valid_int], 
                           np.array(interaction_term_means)[valid_int], 
                           'g-', linewidth=2, label='Mean(σ_k²/|σ_k²-σ_j²|)', alpha=0.8)
        
        # Add gradient noise if provided
        if gradient_noise is not None and len(gradient_noise) > 0:
            # Filter out NaN values
            valid_noise = ~np.isnan(gradient_noise)
            if np.any(valid_noise):
                ax.semilogy(np.array(batch_numbers)[valid_noise], 
                           np.array(gradient_noise)[valid_noise], 
                           'm--', linewidth=2, label='Gradient Noise (||g_batch - g_full||)', alpha=0.8)
        
        ax.set_xlabel('Batch Number')
        ax.set_ylabel('Magnitude (log scale)')
        ax.set_title('Singular Value Dynamics Terms Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Overall title
        layer_info_str = f"Layer '{layer_name}', Shape: {layer_shape}"
        fig.suptitle(f"Singular Value Dynamics Terms{run_info}\n{matrix_description}\n{layer_info_str}", 
                     fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot
        plot_filename = f"{self.timestamp}_{layer_name.replace('.', '_')}_sv_dynamics_terms.png"
        plot_path = os.path.join(self.subdirs['dynamics_terms'], plot_filename)
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        print(f"Saved singular value dynamics terms plot for {layer_name} to: {plot_path}")
        plt.close()
        
        # Also create a detailed analysis text file
        analysis_filename = f"{self.timestamp}_{layer_name.replace('.', '_')}_sv_dynamics_analysis.txt"
        analysis_path = os.path.join(self.subdirs['analysis_text'], analysis_filename)
        
        with open(analysis_path, 'w') as f:
            f.write(f"Singular Value Dynamics Analysis for {layer_name}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Layer Shape: {layer_shape}\n")
            f.write(f"Matrix Description: {matrix_description}\n")
            f.write(f"Number of Runs: {runs}\n\n")
            
            f.write("Summary Statistics:\n")
            f.write("-" * 20 + "\n")
            
            # Loss gradient stats
            f.write(f"Loss Gradient:\n")
            f.write(f"  Mean: {np.mean(loss_gradients):.4e}\n")
            f.write(f"  Std:  {np.std(loss_gradients):.4e}\n")
            f.write(f"  Min:  {np.min(loss_gradients):.4e}\n")
            f.write(f"  Max:  {np.max(loss_gradients):.4e}\n\n")
            
            # Gradient noise stats
            if gradient_noise is not None and len(gradient_noise) > 0:
                valid_noise = np.array(gradient_noise)[~np.isnan(gradient_noise)]
                if len(valid_noise) > 0:
                    f.write(f"Gradient Noise:\n")
                    f.write(f"  Mean: {np.mean(valid_noise):.4e}\n")
                    f.write(f"  Std:  {np.std(valid_noise):.4e}\n")
                    f.write(f"  Min:  {np.min(valid_noise):.4e}\n")
                    f.write(f"  Max:  {np.max(valid_noise):.4e}\n\n")
            else:
                f.write(f"Gradient Noise: DISABLED\n\n")
            
            # Inverse sigma stats
            valid_inv_means = np.array(inverse_sigma_means)[~np.isnan(inverse_sigma_means)]
            if len(valid_inv_means) > 0:
                f.write(f"Mean(1/σ_k):\n")
                f.write(f"  Mean: {np.mean(valid_inv_means):.4e}\n")
                f.write(f"  Std:  {np.std(valid_inv_means):.4e}\n")
                f.write(f"  Min:  {np.min(valid_inv_means):.4e}\n")
                f.write(f"  Max:  {np.max(valid_inv_means):.4e}\n\n")
            
            # Interaction term stats
            valid_int_means = np.array(interaction_term_means)[~np.isnan(interaction_term_means)]
            if len(valid_int_means) > 0:
                f.write(f"Mean(σ_k²/|σ_k²-σ_j²|):\n")
                f.write(f"  Mean: {np.mean(valid_int_means):.4e}\n")
                f.write(f"  Std:  {np.std(valid_int_means):.4e}\n")
                f.write(f"  Min:  {np.min(valid_int_means):.4e}\n")
                f.write(f"  Max:  {np.max(valid_int_means):.4e}\n\n")
            
            # Batch-by-batch data
            f.write("Batch-by-Batch Data:\n")
            f.write("-" * 20 + "\n")
            if gradient_noise is not None and len(gradient_noise) > 0:
                f.write("Batch\tLoss_Grad\tGrad_Noise\tMean(1/σ)\tMean(Interact)\n")
                for i in range(len(batch_numbers)):
                    noise_val = gradient_noise[i] if i < len(gradient_noise) else np.nan
                    f.write(f"{batch_numbers[i]}\t{loss_gradients[i]:.4e}\t{noise_val:.4e}\t")
                    f.write(f"{inverse_sigma_means[i]:.4e}\t{interaction_term_means[i]:.4e}\n")
            else:
                f.write("Batch\tLoss_Grad\tMean(1/σ)\tMean(Interact)\n")
                for i in range(len(batch_numbers)):
                    f.write(f"{batch_numbers[i]}\t{loss_gradients[i]:.4e}\t")
                    f.write(f"{inverse_sigma_means[i]:.4e}\t{interaction_term_means[i]:.4e}\n")
        
        print(f"Saved detailed analysis to: {analysis_path}") 

    def create_metadata_summary(self, data):
        """
        Create a comprehensive metadata summary text file
        
        Args:
            data (dict): Loaded analysis data containing metadata
        """
        summary_filename = f"{self.timestamp}_experiment_metadata.txt"
        summary_path = os.path.join(self.subdirs['analysis_text'], summary_filename)
        
        with open(summary_path, 'w') as f:
            f.write(f"Experiment Metadata Summary\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Generated: {self.timestamp}\n\n")
            
            # Basic metadata
            if 'metadata' in data:
                f.write("Basic Metadata:\n")
                f.write("-" * 20 + "\n")
                for key, value in data['metadata'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Training parameters
            if 'training_parameters' in data:
                f.write("Training Parameters:\n")
                f.write("-" * 20 + "\n")
                for key, value in data['training_parameters'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Analysis parameters
            if 'analysis_parameters' in data:
                f.write("Analysis Parameters:\n")
                f.write("-" * 20 + "\n")
                for key, value in data['analysis_parameters'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Model parameters
            if 'model' in data:
                f.write("Model Parameters:\n")
                f.write("-" * 20 + "\n")
                for key, value in data['model'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # System information
            if 'system_info' in data:
                f.write("System Information:\n")
                f.write("-" * 20 + "\n")
                for key, value in data['system_info'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Data summary
            f.write("Data Summary:\n")
            f.write("-" * 20 + "\n")
            if 'loss_values' in data and len(data['loss_values']) > 0:
                f.write(f"  Total loss data points: {len(data['loss_values'])}\n")
                f.write(f"  Final loss: {data['loss_values'][-1]:.6f}\n")
                f.write(f"  Initial loss: {data['loss_values'][0]:.6f}\n")
            
            if 'batch_numbers' in data and len(data['batch_numbers']) > 0:
                f.write(f"  Batch range: {data['batch_numbers'][0]} - {data['batch_numbers'][-1]}\n")
            
            if 'results' in data:
                f.write(f"  Analyzed layers: {len(data['results'])}\n")
                for layer_name, layer_data in data['results'].items():
                    f.write(f"    - {layer_name}: shape {layer_data.get('shape', 'unknown')}\n")
                    if 'singular_values_list' in layer_data:
                        f.write(f"      Singular value snapshots: {len(layer_data['singular_values_list'])}\n")
        
        print(f"Saved metadata summary to: {summary_path}")
        return summary_path 