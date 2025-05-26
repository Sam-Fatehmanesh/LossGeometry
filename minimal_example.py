# !pip install scikit-rmt

import os
import math
import random
import pickle
import tiktoken
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from skrmt.ensemble.spectral_law import MarchenkoPasturDistribution, TracyWidomDistribution
import requests

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
MODEL_TYPE = 'vit'  # Choose 'mlp', 'vit', or 'nanogpt'
NUM_RUNS            = 5
NUM_EPOCHS          = 10
BATCH_SIZE          = 64
LEARNING_RATE       = 1e-2
MOMENTUM            = 0.9
LOG_EVERY_N_BATCHES = 200

INPUT_SIZE       = 28 * 28
HIDDEN_SIZE      = 1024
OUTPUT_SIZE      = 10
NUM_HIDDEN_LAYERS = 2

DATA_ROOT = './data'

# ViT configuration parameters
VIT_IMAGE_SIZE = 28
VIT_PATCH_SIZE = 7
VIT_EMBED_DIM = 64
VIT_DEPTH = 2
VIT_NUM_HEADS = 4
VIT_MLP_RATIO = 2.0
VIT_INPUT_CHANNELS = 1
VIT_INIT_FC = True


# ------------------------------------------------------------------------------
# MODEL DEFINITION
# ------------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden_layers):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc_layers.append(nn.Linear(hidden_size, output_size))
        self.relu = nn.ReLU()
        self._init_gaussian_weights()

    def _init_gaussian_weights(self):
        for layer in self.fc_layers:
            fan_in = layer.weight.size(1)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers[:-1]:
            x = self.relu(layer(x))
        return self.fc_layers[-1](x)

    def get_target_layers(self):
        return [f"fc_layers.{i}.weight" for i in range(len(self.fc_layers))]

    def get_parameter(self, name):
        parts = name.split('.')
        obj = self
        for p in parts:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        return obj

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
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
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
        # Expose attributes for compatibility
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
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        nn.init.zeros_(self.cls_token)
        nn.init.zeros_(self.pos_embed)
        if gaussian_init_fc:
            fan_in = self.head.weight.data.size(1)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(self.head.weight.data, mean=0.0, std=std)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias.data)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x).transpose(0, 1)
        x = self.transformer(x)
        x = x[0]
        x = self.head(x)
        return x

    def get_target_layers(self):
        layers = ['head.weight']
        for i in range(len(self.transformer.layers)):
            layers.append(f'transformer.layers.{i}.linear1.weight')
            layers.append(f'transformer.layers.{i}.linear2.weight')
        return layers

    def get_parameter(self, layer_name):
        parts = layer_name.split('.')
        obj = self
        for p in parts:
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        return obj

if MODEL_TYPE == 'mlp':
    _reference_model = SimpleMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS)
elif MODEL_TYPE == 'vit':
    _reference_model = CustomViT(
        num_classes=OUTPUT_SIZE,
        image_size=VIT_IMAGE_SIZE,
        patch_size=VIT_PATCH_SIZE,
        embed_dim=VIT_EMBED_DIM,
        depth=VIT_DEPTH,
        num_heads=VIT_NUM_HEADS,
        mlp_ratio=VIT_MLP_RATIO,
        input_channels=VIT_INPUT_CHANNELS,
        gaussian_init_fc=VIT_INIT_FC
    )
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")
LAYER_NAMES = _reference_model.get_target_layers()
LAYER_SHAPES = [tuple(_reference_model.get_parameter(name).shape) for name in LAYER_NAMES]

# ------------------------------------------------------------------------------
# TRAIN + AGGREGATE SINGULAR VALUES
# ------------------------------------------------------------------------------
def train_and_aggregate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # MNIST loader (exact same normalize as main.py)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)

    aggregated_sv   = None
    reference_batches = None

    for run in range(NUM_RUNS):
        print(f"\n--- Run {run+1}/{NUM_RUNS} ---")
        if MODEL_TYPE == 'mlp':
            model = SimpleMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS).to(device)
        else:
            model = CustomViT(
                num_classes=OUTPUT_SIZE,
                image_size=VIT_IMAGE_SIZE,
                patch_size=VIT_PATCH_SIZE,
                embed_dim=VIT_EMBED_DIM,
                depth=VIT_DEPTH,
                num_heads=VIT_NUM_HEADS,
                mlp_ratio=VIT_MLP_RATIO,
                input_channels=VIT_INPUT_CHANNELS,
                gaussian_init_fc=VIT_INIT_FC
            ).to(device)
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        criterion = nn.CrossEntropyLoss()

        loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)

        # collect per-layer snapshots this run
        layer_names = model.get_target_layers()
        sv_snapshots = {name: [] for name in layer_names}
        batch_list   = []
        batch_idx    = 0

        for epoch in range(NUM_EPOCHS):
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
            for imgs, labels in pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss   = criterion(logits, labels)
                loss.backward()

                # snapshot
                if batch_idx % LOG_EVERY_N_BATCHES == 0:
                    for name in layer_names:
                        with torch.no_grad():
                            # move weight matrix to CPU to avoid GPU SVD nan issues
                            W_cpu = model.get_parameter(name).data.cpu()
                            # center the matrix
                            W_center = W_cpu - W_cpu.mean()
                            sigma = W_center.std()
                            if sigma > 1e-15:
                                # normalize by sigma * sqrt(max_dim) to match MP theory
                                max_dim = float(max(W_center.shape))
                                W_norm = W_center / (sigma * math.sqrt(max_dim))
                                sv = torch.linalg.svdvals(W_norm).numpy()
                                # filter out any non-finite values
                                sv = sv[np.isfinite(sv)]
                            else:
                                sv = np.array([], dtype=float)
                        sv_snapshots[name].append(sv)
                    batch_list.append(batch_idx)

                optimizer.step()
                batch_idx += 1

        # first run initializes aggregator
        if aggregated_sv is None:
            aggregated_sv    = {name: [sv_snapshots[name]] for name in sv_snapshots}
            reference_batches = batch_list
        else:
            for name in sv_snapshots:
                aggregated_sv[name].append(sv_snapshots[name])

    # concatenate across runs
    final_sv = {}
    for name, runs_list in aggregated_sv.items():
        num_snaps = len(runs_list[0])
        combined = []
        for snap in range(num_snaps):
            arrays = [ runs_list[r][snap] for r in range(NUM_RUNS) ]
            combined.append(np.concatenate(arrays))
        final_sv[name] = combined

    return final_sv, reference_batches, layer_names


# ------------------------------------------------------------------------------
# PLOTTING 20 EQUALLY-SPACED HISTOGRAMS PER LAYER
# ------------------------------------------------------------------------------
def plot_singular_evolution(final_sv, batches, layer_names, layer_shapes):
    for name in layer_names:
        # Advanced singular-value plotting with MP & TW overlays
        results_data = final_sv[name]
        num_plots    = min(20, len(results_data))
        indices      = np.linspace(0, len(results_data)-1, num_plots, dtype=int)
        # Determine common range robustly, ignoring NaNs
        all_vals = np.concatenate([results_data[i] for i in indices if len(results_data[i]) > 0])
        finite_vals = all_vals[np.isfinite(all_vals)]
        if finite_vals.size > 0:
            max_val = np.max(finite_vals)
            max_plot = max(max_val * 1.1, 3.0)
        else:
            max_plot = 3.0
        common_range = (0, max_plot)
        ncols = 5
        nrows = math.ceil(num_plots / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), squeeze=False)
        axes = axes.flatten()
        # Get layer shape
        idx = layer_names.index(name)
        m, n = layer_shapes[idx]
        ratio = min(m,n)/max(m,n)
        is_square = (m==n)
        for pi, snap in enumerate(indices):
            ax = axes[pi]
            sv_vals = results_data[snap]
            nb = max(min(len(sv_vals)//10, 100), 30)
            counts, bins, _ = ax.hist(sv_vals, bins=nb, density=False,
                                     alpha=0.7, color='royalblue',
                                     label='Histogram', range=common_range)
            w = bins[1]-bins[0]
            centers = (bins[:-1]+bins[1:])/2
            # MP overlay
            mp = MarchenkoPasturDistribution(beta=1, ratio=ratio, sigma=1.0)
            x_mp = np.linspace(common_range[0], common_range[1], 500)
            mp_d = np.array([2*x*mp.pdf(x*x) if x>0 else 0 for x in x_mp])
            ax.plot(x_mp, mp_d * len(sv_vals)*w, 'r--', linewidth=1.5, label=f'MP (λ={ratio:.2f})')
            # TW overlay
            tw = TracyWidomDistribution(beta=1)
            edge = 2.0
            scale = 0.5*(np.sqrt(m)+np.sqrt(n))**(-1/3)
            x_tw = np.linspace(edge-4*scale, edge+4*scale, 300)
            args = (x_tw-edge)/scale
            pdf_tw = tw.pdf(args)/scale
            mpv = np.max(pdf_tw)
            mask = (centers>edge-scale)&(centers<edge+scale)
            peak = np.max(counts[mask]) if np.any(mask) else np.max(counts)
            ax.plot(x_tw, pdf_tw*peak/mpv, 'g-', linewidth=2.0,
                    label=f'TW (edge={edge:.2f}, scale={scale:.2e})')
            formula = (r"$A_k(\tau)\approx\frac{\sigma_{(k)}-2\sqrt{n}}{2^{1/3}n^{1/6}}$"
                       if is_square else
                       r"$A_k(\tau)\approx\frac{\sigma_{(k)}-(\sqrt{m}+\sqrt{n})}{(\sqrt{m}+\sqrt{n})^{1/2}(m^{-1/2}+n^{-1/2})^{1/3}}$")
            ax.text(0.5, 0.93, formula, transform=ax.transAxes,
                    horizontalalignment='center', fontsize=10,
                    bbox=dict(facecolor='white',alpha=0.7))
            ax.set_title(f"Batch {batches[snap]}")
            ax.set_xlim(common_range)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Singular Value (σ)")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize='small')
        for ax in axes[num_plots:]: ax.axis('off')
        fig.suptitle(f"Singular-Value Evolution for {name} (runs={NUM_RUNS})", fontsize=14)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.show()


# ------------------------------------------------------------------------------
# TEXT DATA PREPARATION
# ------------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")

def download_text_sample(data_dir):
    """Download a sample text dataset"""
    os.makedirs(data_dir, exist_ok=True)
    
    # Use a tiny Shakespeare dataset instead - it's reliable and small
    sample_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_file = os.path.join(data_dir, "shakespeare.txt")
    
    if not os.path.exists(output_file):
        print(f"Downloading Shakespeare sample to {output_file}...")
        with requests.get(sample_url, stream=True) as r:
            r.raise_for_status()  # Ensure we catch HTTP errors
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(output_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
        print(f"Download complete: {output_file}")
    else:
        print(f"File already exists: {output_file}")
    
    return output_file

def prepare_text_data(data_dir, val_split=0.1, seed=42):
    """Prepare text dataset for nanoGPT by tokenizing and converting to .bin files"""
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the text sample
    text_file = download_text_sample(data_dir)
    
    # Verify the file exists and has content
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Text file not found: {text_file}")
    
    if os.path.getsize(text_file) == 0:
        raise ValueError(f"Downloaded file is empty: {text_file}")
    
    # Read the text file and split into documents
    print("Processing text data...")
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # For Shakespeare, split by newlines to get reasonable chunks
    raw_documents = [doc.strip() for doc in text.split('\n') if doc.strip()]
    
    # Combine short lines to form more substantial documents
    documents = []
    current_doc = []
    for line in raw_documents:
        current_doc.append(line)
        # If we've accumulated about 100 tokens or hit a scene break, create a document
        if len(' '.join(current_doc)) > 400 or line.strip() == '':
            if current_doc:  # Ensure we don't add empty documents
                documents.append(' '.join(current_doc))
                current_doc = []
    
    # Add any remaining lines
    if current_doc:
        documents.append(' '.join(current_doc))
    
    print(f"Split text into {len(documents)} documents")
    
    # Shuffle the documents
    random.shuffle(documents)
    
    # Split into train/val
    n_val = max(1, int(len(documents) * val_split))
    train_documents = documents[:-n_val]
    val_documents = documents[-n_val:]
    
    # Tokenize
    print(f"Tokenizing {len(train_documents)} train documents...")
    train_tokens = []
    for doc in tqdm(train_documents):
        train_tokens.extend(enc.encode(doc))
        # Add a separator token (newline)
        train_tokens.append(enc.eot_token)  # EOT token
    
    print(f"Tokenizing {len(val_documents)} validation documents...")
    val_tokens = []
    for doc in tqdm(val_documents):
        val_tokens.extend(enc.encode(doc))
        # Add a separator token (newline)
        val_tokens.append(enc.eot_token)  # EOT token
    
    # Convert to uint16 arrays and save
    print(f"Train set: {len(train_tokens)} tokens")
    train_tokens = np.array(train_tokens, dtype=np.uint16)
    save_path = os.path.join(data_dir, 'train.bin')
    train_tokens.tofile(save_path)
    
    print(f"Validation set: {len(val_tokens)} tokens")
    val_tokens = np.array(val_tokens, dtype=np.uint16)
    save_path = os.path.join(data_dir, 'val.bin')
    val_tokens.tofile(save_path)
    
    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens)
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        import pickle
        pickle.dump(meta, f)
    
    print("Done!")

prepare_text_data(DATA_ROOT)

start = datetime.now()
final_sv, batches, layers = train_and_aggregate()
print(f"\nTotal time: {datetime.now() - start}\n")
plot_singular_evolution(final_sv, batches, layers, LAYER_SHAPES)

