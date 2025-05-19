"""
Prepare text data for training.
This script downloads a small text dataset and prepares it for training with nanoGPT.
"""

import os
import sys
import json
import random
import numpy as np
import tiktoken
import requests
import argparse
from tqdm import tqdm
import gzip
import shutil
from pathlib import Path

# Setup tokenizer
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare text data for nanoGPT training")
    parser.add_argument("--data_dir", type=str, default="data/pile", help="Directory to store the processed data")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    prepare_text_data(args.data_dir, args.val_split, args.seed) 