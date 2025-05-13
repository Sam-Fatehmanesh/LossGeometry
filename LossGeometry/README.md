# LossGeometry


## Project Structure

```
LossGeometry/
├── __init__.py
├── models/
│   ├── __init__.py
│   └── mlp.py                # SimpleMLP and other models
├── datasets/
│   ├── __init__.py
│   └── mnist_dataset.py      # MNIST dataset loading
├── analysis/
│   ├── __init__.py
│   └── spectral_analysis.py  # Eigenvalue and singular value analysis
├── visualization/
│   ├── __init__.py
│   └── plot_utils.py         # Plotting functions
├── utils/
│   ├── __init__.py
│   └── io_utils.py           # Saving and loading data
└── main.py                   # Entry point
```

## Usage

### Basic Usage

Run the main script with default parameters:

```bash
python -m LossGeometry.main
```

This will:
1. Train a SimpleMLP model on MNIST for 1 epoch
2. Analyze weight matrices (W) for spectral density and singular values
3. Create an experiment directory with a timestamp
4. Save analysis data in HDF5 format
5. Generate plots for the analysis results

### Command Line Arguments

#### Model Parameters
- `--input_size`: Input size (default: 784)
- `--hidden_size`: Hidden layer size (default: 1024)
- `--output_size`: Output size (default: 10)
- `--num_hidden_layers`: Number of hidden layers with square dimensions (default: 2)

#### Training Parameters
- `--num_epochs`: Number of epochs (default: 1)
- `--batch_size`: Batch size (default: 64)
- `--log_every_n_batches`: Frequency of analysis calculation (default: 200)
- `--learning_rate`: Learning rate (default: 0.001)
- `--num_runs`: Number of training runs to average results over (default: 1)

#### Analysis Parameters
- `--analyze_W`: Analyze weight matrices (default if none specified)
- `--analyze_delta_W`: Analyze weight update matrices
- `--analyze_spectral_density`: Analyze spectral density (default if none specified)
- `--analyze_level_spacing`: Analyze level spacing
- `--analyze_singular_values`: Analyze singular values (default if none specified)

#### Output Parameters
- `--experiment_dir`: Base directory for experiments (default: "experiments")

### Examples

1. Analyze with 3 hidden layers:

```bash
python -m LossGeometry.main --num_hidden_layers 3
```

2. Analyze level spacing instead of spectral density:

```bash
python -m LossGeometry.main --analyze_level_spacing --analyze_singular_values
```

3. Analyze weight updates (ΔW) instead of weights (W):

```bash
python -m LossGeometry.main --analyze_delta_W
```

4. Run multiple training runs and average the results:

```bash
python -m LossGeometry.main --num_runs 5
```

## Data Storage

Analysis data is stored in HDF5 format with the following structure:

- `metadata/`: Timestamp, matrix type, analysis type, matrix description, number of runs
- `model/`: Model parameters (input size, hidden size, output size, num hidden layers)
- `batch_numbers`: Array of batch numbers for loss values
- `batches`: Array of batch numbers for analysis points
- `loss_values`: Array of loss values
- `layers/`: Group containing data for each layer
  - `layer_name/`: Group for each layer
    - `shape`: Shape of the layer
    - `eigenvalues/`: Group containing eigenvalue data
      - `batch_i`: Array of eigenvalues for batch i
    - `spacing/`: Group containing level spacing data
      - `std_dev_list`: Array of standard deviations for normalized spacings
      - `last_spacings`: Array of last normalized spacings
    - `singular_values/`: Group containing singular value data
      - `batch_i`: Array of singular values for batch i

## Requirements

This project requires the following Python packages:
- PyTorch
- NumPy
- Matplotlib
- h5py

You can install all required dependencies using:
```bash
pip install -r requirements.txt
``` 