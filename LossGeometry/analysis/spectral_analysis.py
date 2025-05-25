import numpy as np
import torch

class SpectralAnalyzer:
    """
    Class for performing spectral analysis on neural network weights
    Focuses on singular value analysis only
    """
    def __init__(self, analyze_W=True, analyze_delta_W=False, analyze_singular_values=True):
        """
        Initialize the spectral analyzer
        
        Args:
            analyze_W (bool): Whether to analyze weight matrices
            analyze_delta_W (bool): Whether to analyze weight update matrices
            analyze_singular_values (bool): Whether to analyze singular values
        """
        # Sanity checks
        if not analyze_W and not analyze_delta_W:
            raise ValueError("Please select a matrix type to analyze (analyze_W or analyze_delta_W).")
        if not analyze_singular_values:
            raise ValueError("Singular value analysis must be enabled.")
        
        self.analyze_W = analyze_W
        self.analyze_delta_W = analyze_delta_W
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
            'analysis_type': 'SingularValues',
            'loss_values': [],
            'batch_numbers': [],
            'loss_gradients': [],  # Track loss gradient magnitudes
            'results': {}
        }
    
    def initialize_layer_stats(self, target_layers):
        """Initialize result structure for each layer"""
        for layer in target_layers:
            self.stats['results'][layer] = {}
            
            if self.analyze_singular_values:
                self.stats['results'][layer]['singular_values_list'] = []
                self.stats['results'][layer]['last_singular_values'] = None
                # Add loss gradient tracking for each layer
                self.stats['results'][layer]['loss_gradients'] = []
    
    def estimate_gradient_noise_fullbatch_diff(self, model, data_loader, criterion, current_inputs, current_labels):
        """
        Estimate gradient noise as ||g_batch - g_full|| where g_batch is the current mini-batch gradient
        and g_full is the full-batch gradient computed over the entire dataset
        
        Args:
            model: The neural network model
            data_loader: DataLoader for the full dataset
            criterion: Loss function
            current_inputs: Current mini-batch inputs
            current_labels: Current mini-batch labels
            
        Returns:
            float: Gradient noise magnitude ||g_batch - g_full||
        """
        device = next(model.parameters()).device
        
        # Store current gradients (mini-batch)
        batch_grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                batch_grad_list.append(param.grad.view(-1).clone())
        
        if not batch_grad_list:
            return None
            
        batch_grad = torch.cat(batch_grad_list)
        
        # Compute full-batch gradient
        model.zero_grad()
        total_loss = 0.0
        total_samples = 0
        
        # Accumulate gradients over the full dataset
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Scale loss by batch size to get proper averaging
            batch_size = inputs.size(0)
            scaled_loss = loss * batch_size
            scaled_loss.backward()
            
            total_samples += batch_size
        
        # Get full-batch gradient and normalize by total samples
        full_grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                # Normalize by total number of samples
                normalized_grad = param.grad.view(-1) / total_samples
                full_grad_list.append(normalized_grad.clone())
        
        if not full_grad_list:
            return None
            
        full_grad = torch.cat(full_grad_list)
        
        # Compute noise as ||g_batch - g_full||
        noise_vector = batch_grad - full_grad
        noise_magnitude = torch.norm(noise_vector).item()
        
        # Restore the original mini-batch gradients for the optimizer step
        model.zero_grad()
        current_outputs = model(current_inputs)
        current_loss = criterion(current_outputs, current_labels)
        current_loss.backward()
        
        return noise_magnitude

    def analyze_batch(self, model, optimizer, current_batch, loss_tensor=None, data_loader=None, criterion=None, current_inputs=None, current_labels=None):
        """
        Analyze the model parameters at the current batch
        
        Args:
            model (nn.Module): The model to analyze
            optimizer: The optimizer used for training
            current_batch (int): Current batch number
            loss_tensor (torch.Tensor, optional): Current loss tensor for gradient computation
            data_loader (DataLoader, optional): DataLoader for full-batch gradient computation
            criterion (loss function, optional): Loss function for gradient computation
            current_inputs (torch.Tensor, optional): Current mini-batch inputs
            current_labels (torch.Tensor, optional): Current mini-batch labels
            
        Returns:
            bool: Whether batch was recorded
        """
        need_to_record_batch = False
        layer_shapes = {}
        
        # Calculate loss gradient magnitude if loss tensor is provided
        loss_grad_magnitude = None
        if loss_tensor is not None:
            try:
                # Compute gradient of loss w.r.t. all parameters
                total_grad_norm = 0.0
                param_count = 0
                
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad).item()
                        total_grad_norm += grad_norm**2
                        param_count += 1
                
                if param_count > 0:
                    loss_grad_magnitude = np.sqrt(total_grad_norm)
                    
            except Exception as e:
                print(f"  Error computing loss gradient magnitude: {e}")
        
        # Estimate gradient noise using full-batch vs mini-batch difference
        gradient_noise = None
        if (data_loader is not None and criterion is not None and 
            current_inputs is not None and current_labels is not None):
            try:
                gradient_noise = self.estimate_gradient_noise_fullbatch_diff(
                    model, data_loader, criterion, current_inputs, current_labels)
                if gradient_noise is not None:
                    print(f"  Estimated gradient noise (||g_batch - g_full||): {gradient_noise:.4e}")
            except Exception as e:
                print(f"  Error computing gradient noise: {e}")
        
        with torch.no_grad():
            for target_layer_name in model.get_target_layers():
                try:
                    target_param = model.get_parameter(target_layer_name)
                    layer_shapes[target_layer_name] = target_param.shape
                    
                    # Track loss gradient for this layer if available
                    if loss_grad_magnitude is not None:
                        self.stats['results'][target_layer_name]['loss_gradients'].append(loss_grad_magnitude)
                    
                    # Track gradient noise for this layer if available
                    if gradient_noise is not None:
                        if 'gradient_noise' not in self.stats['results'][target_layer_name]:
                            self.stats['results'][target_layer_name]['gradient_noise'] = []
                        self.stats['results'][target_layer_name]['gradient_noise'].append(gradient_noise)
                    
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
            # Also track the global loss gradient magnitude and noise
            if loss_grad_magnitude is not None:
                self.stats['loss_gradients'].append(loss_grad_magnitude)
            else:
                self.stats['loss_gradients'].append(np.nan)
                
            # Track global gradient noise
            if 'gradient_noise' not in self.stats:
                self.stats['gradient_noise'] = []
            if gradient_noise is not None:
                self.stats['gradient_noise'].append(gradient_noise)
            else:
                self.stats['gradient_noise'].append(np.nan)
        
        return need_to_record_batch
    
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
        
    def track_loss_gradient(self, loss_grad_magnitude, batch_number):
        """Track loss gradient magnitude for plotting"""
        self.stats['loss_gradients'].append(loss_grad_magnitude)
        
    def estimate_noise_level(self, layer_name, method='smallest_sv_ratio', percentile=5):
        """
        Estimate noise level for a given layer using various methods
        
        Args:
            layer_name (str): Name of the layer
            method (str): Method for noise estimation ('smallest_sv_ratio', 'percentile', 'mp_tail')
            percentile (float): Percentile to use for percentile method
            
        Returns:
            float or list: Estimated noise level(s)
        """
        if layer_name not in self.stats['results']:
            return None
            
        layer_results = self.stats['results'][layer_name]
        if 'singular_values_list' not in layer_results or not layer_results['singular_values_list']:
            return None
        
        sv_list = layer_results['singular_values_list']
        noise_estimates = []
        
        for sv_array in sv_list:
            if sv_array is None or len(sv_array) == 0:
                noise_estimates.append(np.nan)
                continue
                
            if method == 'smallest_sv_ratio':
                # Use ratio of smallest to largest singular value
                sv_sorted = np.sort(sv_array)
                if len(sv_sorted) > 1:
                    noise_est = sv_sorted[0] / sv_sorted[-1]
                else:
                    noise_est = sv_sorted[0] if len(sv_sorted) > 0 else np.nan
                    
            elif method == 'percentile':
                # Use a percentile of the singular value distribution
                noise_est = np.percentile(sv_array, percentile)
                
            elif method == 'mp_tail':
                # Estimate based on deviation from Marchenko-Pastur tail
                # This is more sophisticated and would require fitting
                # For now, use a simple approximation
                sv_sorted = np.sort(sv_array)
                # Take the bottom 10% as noise estimate
                bottom_10_percent = int(0.1 * len(sv_sorted))
                if bottom_10_percent > 0:
                    noise_est = np.mean(sv_sorted[:bottom_10_percent])
                else:
                    noise_est = sv_sorted[0] if len(sv_sorted) > 0 else np.nan
            else:
                noise_est = np.nan
                
            noise_estimates.append(noise_est)
        
        return noise_estimates if len(noise_estimates) > 1 else (noise_estimates[0] if noise_estimates else None)
        
    def get_stats(self):
        """Get the current analysis stats"""
        return self.stats 