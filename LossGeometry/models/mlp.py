import torch
import torch.nn as nn
import math

class SimpleMLP(nn.Module):
    """
    MLP with variable number of hidden layers, all with the same hidden size.
    First and last layers have non-square weight matrices.
    Middle layers have square weight matrices.
    Uses Gaussian initialization for weight matrices.
    """
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
        
        # Initialize weights with Gaussian distribution
        self._init_gaussian_weights()
        
        self.relu = nn.ReLU()

    def _init_gaussian_weights(self):
        """
        Initialize weights with Gaussian distribution N(0, 1/n)
        where n is the number of input neurons (fan_in)
        """
        for layer in self.fc_layers:
            # Calculate fan_in (number of input features)
            fan_in = layer.weight.data.size(1)
            
            # Standard deviation = 1/sqrt(fan_in)
            std = 1.0 / math.sqrt(fan_in)
            
            # Initialize with Gaussian distribution
            nn.init.normal_(layer.weight.data, mean=0.0, std=std)
            
            # Initialize bias to zero
            if layer.bias is not None:
                nn.init.zeros_(layer.bias.data)
        
        print("Initialized all layers with Gaussian weights N(0, 1/fan_in)")

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image
        
        # Process through all layers except the last one with ReLU
        for i in range(len(self.fc_layers) - 1):
            x = self.relu(self.fc_layers[i](x))
        
        # Last layer without ReLU
        x = self.fc_layers[-1](x)
        return x
    
    def get_target_layers(self):
        """Return a list of target layer names for analysis"""
        target_layers = []
        # All weight layers
        for i in range(len(self.fc_layers)):
            target_layers.append(f'fc_layers.{i}.weight')
        return target_layers 