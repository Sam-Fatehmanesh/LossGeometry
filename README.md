# LossGeometry




## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
./run_analysis.py
```

This will train a model on MNIST for 1 epoch, analyzing spectral density and singular values of weight matrices.

### Multiple Runs

To run multiple training runs and average the results:

```bash
./run_analysis.py -n 5  # Run 5 times and average
```

The `-n` parameter (or `--num_runs`) determines how many times to train the model from scratch. After all runs complete, the results are aggregated and plots show the averaged distributions.

Multiple runs help identify trends and patterns that persist across different random initializations.

### Other Parameters

```bash
./run_analysis.py --hidden_size 512 --num_hidden_layers 3 -n 3
```

See the full list of options with:

```bash
./run_analysis.py --help
```

## Project Organization

The core functionality is packaged in the `LossGeometry` directory, which can be imported as a Python module or run via the main script.

Results are saved in the `experiments/` directory with timestamps, including:
- HDF5 files with all analysis data
- Loss curves
- Spectral density plots
- Singular value distribution plots
- Level spacing plots (if requested)

## Example

Train a 3-layer network with 5 different random initializations, analyzing the singular value distributions:

```bash
./run_analysis.py --num_hidden_layers 3 --analyze_singular_values -n 5
```

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