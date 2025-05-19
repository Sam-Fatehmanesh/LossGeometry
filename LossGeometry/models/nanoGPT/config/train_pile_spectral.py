# Config for training nanoGPT on the Shakespeare dataset with spectral analysis
# Run as: python train_spectral.py config/train_pile_spectral.py

wandb_log = True
wandb_project = 'shakespeare_spectral'
wandb_run_name = 'nanogpt-shakespeare-spectral'

# Dataset
dataset = 'pile'  # we're still using the data_dir 'pile' for compatibility

# Model size - using smaller model for faster spectral analysis
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
bias = False

# Training parameters
batch_size = 12
block_size = 256
gradient_accumulation_steps = 1

# Shorter training run for spectral analysis
max_iters = 2000
lr_decay_iters = 2000
warmup_iters = 200

# Learning rate
learning_rate = 5e-4
min_lr = 5e-5

# Evaluation and logging
eval_interval = 100
save_interval = 200
log_interval = 10
eval_iters = 50

# Weight decay
weight_decay = 0.1

# Spectral analysis parameters
enable_spectral_analysis = True
spectral_analysis_interval = 200