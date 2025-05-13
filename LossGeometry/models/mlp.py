import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """
    MLP with variable number of hidden layers, all with the same hidden size.
    First and last layers have non-square weight matrices.
    Middle layers have square weight matrices.
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
        
        self.relu = nn.ReLU()

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