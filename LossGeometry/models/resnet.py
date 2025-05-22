import torch
import torch.nn as nn
import math
from torchvision.models import resnet18

class CustomResNet18(nn.Module):
    """
    Wrapper around torchvision's ResNet-18, with optional Gaussian initialization
    of fc and conv layers, and adjustable input channels.
    """
    def __init__(self, num_classes=10, input_channels=1,
                 gaussian_init_fc=True, gaussian_init_conv=False):
        super().__init__()
        # Expose attributes for io_utils compatibility
        self.input_size = input_channels
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.gaussian_init_fc = gaussian_init_fc
        self.gaussian_init_conv = gaussian_init_conv

        # Load base ResNet-18
        # Use pretrained=False to start from scratch
        self.model = resnet18(pretrained=False)

        # Adjust first conv for different input channels
        if input_channels != 3:
            conv1 = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=(conv1.bias is not None)
            )

        # Replace final fully-connected layer for our num_classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes, bias=True)
        # Assign hidden_size, output_size, num_hidden_layers for compatibility
        self.hidden_size = in_features
        self.output_size = num_classes
        self.num_hidden_layers = 0

        # Apply Gaussian initialization
        if self.gaussian_init_conv:
            self._init_gaussian_weights_conv()
        if self.gaussian_init_fc:
            self._init_gaussian_weights_fc()

    def _init_gaussian_weights_fc(self):
        """Initialize the fc layer weights with N(0,1/fan_in)"""
        layer = self.model.fc
        fan_in = layer.weight.data.size(1)
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(layer.weight.data, mean=0.0, std=std)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias.data)

    def _init_gaussian_weights_conv(self):
        """Initialize all Conv2d layers with N(0,1/fan_in)"""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                # fan_in = in_channels * kernel_height * kernel_width
                fan_in = m.weight.data.size(1) * m.weight.data.size(2) * m.weight.data.size(3)
                std = 1.0 / math.sqrt(fan_in)
                nn.init.normal_(m.weight.data, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.model(x)

    def get_target_layers(self):
        """Return a list of parameter names to analyze (only the final fc weight by default)"""
        return ['model.fc.weight']

    def get_parameter(self, layer_name):
        """Retrieve a parameter by its dotted path name"""
        parts = layer_name.split('.')
        obj = self
        for p in parts:
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        return obj 