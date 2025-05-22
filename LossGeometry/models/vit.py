import torch
import torch.nn as nn
import math

class PatchEmbed(nn.Module):
    """
    Split image into patches and embed them.
    """
    def __init__(self, img_size=28, patch_size=7, in_chans=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch_size, in_chans, img_size, img_size)
        x = self.proj(x)  # (batch_size, embed_dim, H/ps, W/ps)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

class CustomViT(nn.Module):
    """
    Very small Vision Transformer (ViT) wrapper for LossGeometry.
    Implements patch embedding, transformer encoder, and classification head.
    """
    def __init__(self,
                 num_classes=10,
                 image_size=28,
                 patch_size=7,
                 embed_dim=64,
                 depth=2,
                 num_heads=4,
                 mlp_ratio=2.0,
                 input_channels=1,
                 gaussian_init_fc=False):
        super().__init__()
        # Expose attributes for io_utils compatibility
        self.input_size = input_channels
        self.hidden_size = embed_dim
        self.output_size = num_classes
        self.num_hidden_layers = depth

        # Patch embedding module
        self.patch_embed = PatchEmbed(image_size, patch_size, input_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(p=0.0)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialization
        # Initialize patch embedding conv with Xavier
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        # Initialize class token and positional embeddings
        nn.init.zeros_(self.cls_token)
        nn.init.zeros_(self.pos_embed)
        # Optionally initialize classification head with Gaussian
        if gaussian_init_fc:
            fan_in = self.head.weight.data.size(1)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(self.head.weight.data, mean=0.0, std=std)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias.data)

    def forward(self, x):
        # x: (batch_size, in_chans, img_size, img_size)
        batch_size = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        # Transformer expects (sequence_len, batch_size, embed_dim)
        x = self.pos_dropout(x).transpose(0, 1)
        x = self.transformer(x)
        x = x[0]  # take cls token output
        x = self.head(x)
        return x

    def get_target_layers(self):
        """Return a list of parameter names to analyze (classification head by default)."""
        layers = ['head.weight']
        # Include feedforward layers (linear1 and linear2) from each transformer encoder block
        for i in range(len(self.transformer.layers)):
            layers.append(f'transformer.layers.{i}.linear1.weight')
            layers.append(f'transformer.layers.{i}.linear2.weight')
        return layers

    def get_parameter(self, layer_name):
        """Retrieve a parameter by its dotted path name."""
        parts = layer_name.split('.')
        obj = self
        for p in parts:
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        return obj 