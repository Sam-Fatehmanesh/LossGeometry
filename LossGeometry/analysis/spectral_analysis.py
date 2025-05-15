import numpy as np
import torch

class SpectralAnalyzer:
    """
    Class for performing spectral analysis on neural network weights
    Computes eigenvalues and singular values of weight matrices
    """
    def __init__(self, analyze_W=True, analyze_delta_W=False, analyze_spectral_density=True, 
                 analyze_level_spacing=False, analyze_singular_values=True):
        """
        Initialize the spectral analyzer
        
        Args:
            analyze_W (bool): Whether to analyze weight matrices
            analyze_delta_W (bool): Whether to analyze weight update matrices
            analyze_spectral_density (bool): Whether to analyze spectral density
            analyze_level_spacing (bool): Whether to analyze level spacing
            analyze_singular_values (bool): Whether to analyze singular values
        """
        # Sanity checks
        if not analyze_W and not analyze_delta_W:
            raise ValueError("Please select a matrix type to analyze (analyze_W or analyze_delta_W).")
        if not analyze_spectral_density and not analyze_level_spacing and not analyze_singular_values:
            raise ValueError("Please select at least one analysis type.")
        if analyze_spectral_density and analyze_level_spacing:
            raise ValueError("Please select only ONE eigenvalue analysis type.")
        
        self.analyze_W = analyze_W
        self.analyze_delta_W = analyze_delta_W
        self.analyze_spectral_density = analyze_spectral_density
        self.analyze_level_spacing = analyze_level_spacing
        self.analyze_singular_values = analyze_singular_values
        
        # Figure out GPU vs CPU once
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Description for plotting titles
        if analyze_W:
            self.matrix_description = "Centered & Normalized Weight (W' / (std(W') * sqrt(max(N))))"
        else:
            self.matrix_description = "Normalized Weight Update (ΔW / (std(ΔW) * sqrt(max(N))))"
            
        # Initialize results structure
        self.reset_stats()
        
    def reset_stats(self):
        """Reset all analysis stats"""
        self.stats = {
            'batch': [],
            'matrix_type': 'W' if self.analyze_W else 'DeltaW',
            'analysis_type': 'Density' if self.analyze_spectral_density else 'Spacing',
            'loss_values': [],
            'batch_numbers': [],
            'results': {}
        }
    
    def initialize_layer_stats(self, target_layers):
        """Initialize result structure for each layer"""
        for layer in target_layers:
            self.stats['results'][layer] = {}
            
            if self.analyze_spectral_density:
                self.stats['results'][layer]['eigenvalues_list'] = []
                self.stats['results'][layer]['last_eigenvalues'] = None
            
            elif self.analyze_level_spacing:
                self.stats['results'][layer]['std_dev_norm_spacing_list'] = []
                self.stats['results'][layer]['last_normalized_spacings'] = None
            
            if self.analyze_singular_values:
                self.stats['results'][layer]['singular_values_list'] = []
                self.stats['results'][layer]['last_singular_values'] = None
    
    def analyze_batch(self, model, optimizer, current_batch):
        """
        Analyze the model parameters at the current batch
        
        Args:
            model (nn.Module): The model to analyze
            optimizer: The optimizer used for training
            current_batch (int): Current batch number
            
        Returns:
            bool: Whether batch was recorded
        """
        need_to_record_batch = False
        layer_shapes = {}
        
        with torch.no_grad():
            for target_layer_name in model.get_target_layers():
                try:
                    target_param = model.get_parameter(target_layer_name)
                    layer_shapes[target_layer_name] = target_param.shape
                    
                    # --- Get the raw matrix and keep it on the original device ---
                    if self.analyze_W:
                        matrix_tensor = target_param.data.to(self.device)
                    elif self.analyze_delta_W:
                        if target_param.grad is None:
                            print(f"  Skipping {target_layer_name} ΔW analysis: Gradient is None.")
                            matrix_tensor = None
                        else:
                            grad_W_tensor = target_param.grad.to(self.device)
                            lr = optimizer.param_groups[0]['lr']
                            # Delta W = -lr * gradient
                            matrix_tensor = -lr * grad_W_tensor
                    
                    # --- Perform analysis ---
                    if matrix_tensor is not None:
                        # Check if matrix is effectively zero
                        matrix_std = torch.std(matrix_tensor)
                        if matrix_std < 1e-15:
                            print(f"  Skipping {target_layer_name} analysis: Matrix standard deviation is near zero ({matrix_std.item():.2e}).")
                            continue
                        
                        # Try to compute eigenvalues if needed (only for square matrices)
                        if (self.analyze_spectral_density or self.analyze_level_spacing) and matrix_tensor.shape[0] == matrix_tensor.shape[1]:
                            try:
                                if self.analyze_spectral_density:
                                    if self._analyze_spectral_density(matrix_tensor, target_layer_name):
                                        need_to_record_batch = True
                                
                                elif self.analyze_level_spacing:
                                    if self._analyze_level_spacing(matrix_tensor, target_layer_name):
                                        need_to_record_batch = True
                            
                            except RuntimeError as e:
                                print(f"  Error during eigenvalue computation for {target_layer_name}: {e}")
                        elif self.analyze_spectral_density or self.analyze_level_spacing:
                            # Non-square matrix
                            print(f"  Skipping eigenvalue analysis for non-square matrix {target_layer_name} of shape {matrix_tensor.shape}.")
                        
                        # Compute singular values if requested (works for any matrix shape)
                        if self.analyze_singular_values:
                            if self._analyze_singular_values(matrix_tensor, target_layer_name):
                                need_to_record_batch = True
                
                except AttributeError:
                    print(f"  Error: Layer '{target_layer_name}' not found during analysis.")
                except Exception as e:
                    print(f"  An unexpected error occurred during analysis of {target_layer_name}: {e}")
        
        # Record batch if any analysis was successful
        if need_to_record_batch:
            self.stats['batch'].append(current_batch)
        
        return need_to_record_batch
    
    def _analyze_spectral_density(self, matrix_tensor, target_layer_name):
        """Analyze spectral density of the matrix"""
        # --- Density Analysis: Normalize matrix THEN compute eigenvalues ---
        matrix_norm = None
        
        if self.analyze_W:
            W_centered = matrix_tensor - torch.mean(matrix_tensor)
            sigma_W_centered = torch.std(W_centered)
            if sigma_W_centered > 1e-15:
                # Use sqrt(min(shape)) for normalization
                min_dim = torch.tensor(min(matrix_tensor.shape), device=self.device, dtype=torch.float)
                matrix_norm = W_centered / (sigma_W_centered * torch.sqrt(min_dim))
            else:
                print(f"  Skipping {target_layer_name} Density(W): Std dev near zero.")
        else:  # analyze_delta_W
            sigma_delta_w = torch.std(matrix_tensor)
            if sigma_delta_w > 1e-15:
                # Use sqrt(min(shape)) for normalization
                min_dim = torch.tensor(min(matrix_tensor.shape), device=self.device, dtype=torch.float)
                matrix_norm = matrix_tensor / (sigma_delta_w * torch.sqrt(min_dim))
            else:
                print(f"  Skipping {target_layer_name} Density(ΔW): Std dev near zero.")

        if matrix_norm is not None:
            try:
                # Compute eigenvalues on GPU
                eigs = torch.linalg.eigvals(matrix_norm)
                
                # Extract real part and move to CPU
                real_eigs = eigs.real.cpu().numpy()
                
                print(f"  Calculated {len(real_eigs)} normalized eigenvalues for {target_layer_name}.")
                self.stats['results'][target_layer_name]['eigenvalues_list'].append(real_eigs)
                self.stats['results'][target_layer_name]['last_eigenvalues'] = real_eigs
                return True
            except RuntimeError as e:
                print(f"  Error: Eigenvalue computation failed for *normalized* matrix of {target_layer_name}: {e}")
        
        return False
    
    def _analyze_level_spacing(self, matrix_tensor, target_layer_name):
        """Analyze level spacing of the matrix eigenvalues"""
        try:
            # Compute eigenvalues on GPU
            eigs = torch.linalg.eigvals(matrix_tensor)
            
            # Extract real part and move to CPU for spacing analysis
            eigenvalues = eigs.real.cpu().numpy()
            
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
                            normalized_spacings = spacings / mean_spacing  # Normalize original spacings
                            # Filter *again* for stats/plotting if needed (e.g., std dev of s>0)
                            norm_spacings_positive = normalized_spacings[normalized_spacings > 1e-10]

                            if len(norm_spacings_positive) > 0:
                                std_dev_norm = np.std(norm_spacings_positive)
                                print(f"  Std Dev of Normalized Spacings for {target_layer_name} (s > 1e-10): {std_dev_norm:.4f}")
                                self.stats['results'][target_layer_name]['std_dev_norm_spacing_list'].append(std_dev_norm)
                                self.stats['results'][target_layer_name]['last_normalized_spacings'] = normalized_spacings  # Store all norm. spacings
                                return True
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
                
        except RuntimeError as e:
            print(f"  Error during level spacing computation for {target_layer_name}: {e}")
        
        return False
    
    def _analyze_singular_values(self, matrix_tensor, target_layer_name):
        """Analyze singular values of the matrix"""
        try:
            # Normalize matrix to make comparisons more meaningful
            if self.analyze_W:
                W_centered = matrix_tensor - torch.mean(matrix_tensor)
                sigma_W_centered = torch.std(W_centered)
                if sigma_W_centered > 1e-15:
                    # Use sqrt(max(shape)) for normalization to match MP theory
                    # where n is the number of samples (larger dimension)
                    max_dim = torch.tensor(max(matrix_tensor.shape), device=self.device, dtype=torch.float)
                    matrix_norm = W_centered / (sigma_W_centered * torch.sqrt(max_dim))
                else:
                    print(f"  Skipping {target_layer_name} SVD: Std dev near zero.")
                    matrix_norm = None
            else:  # analyze_delta_W
                sigma_delta_w = torch.std(matrix_tensor)
                if sigma_delta_w > 1e-15:
                    # Use sqrt(max(shape)) for normalization to match MP theory
                    max_dim = torch.tensor(max(matrix_tensor.shape), device=self.device, dtype=torch.float)
                    matrix_norm = matrix_tensor / (sigma_delta_w * torch.sqrt(max_dim))
                else:
                    print(f"  Skipping {target_layer_name} SVD: Std dev near zero.")
                    matrix_norm = None
            
            if matrix_norm is not None:
                # Compute singular values on GPU
                singular_values = torch.linalg.svdvals(matrix_norm)
                
                # Move to CPU for storage and plotting
                singular_values_cpu = singular_values.cpu().numpy()
                
                print(f"  Calculated {len(singular_values_cpu)} normalized singular values for {target_layer_name}.")
                self.stats['results'][target_layer_name]['singular_values_list'].append(singular_values_cpu)
                self.stats['results'][target_layer_name]['last_singular_values'] = singular_values_cpu
                return True
        
        except RuntimeError as e:
            print(f"  Error: SVD computation failed for {target_layer_name}: {e}")
        except Exception as e:
            print(f"  Error during singular value computation for {target_layer_name}: {e}")
        
        return False
    
    def track_loss(self, loss_value, batch_number):
        """Track loss values for plotting"""
        self.stats['loss_values'].append(loss_value)
        self.stats['batch_numbers'].append(batch_number)
        
    def get_stats(self):
        """Get the current analysis stats"""
        return self.stats 