# Paper1 Data Experiment

This repository contains training scripts for two neural network architectures (PointNetMLPJoint and VanillaDeepONet) for stress prediction on L-bracket geometries.

## Overview

This experiment trains two models of similar relative size with different architectures on various dataset sizes to compare their performance. The key feature is that **all models share the same validation dataset** regardless of the training dataset size, ensuring fair comparison.

## Dataset

The primary dataset used is the **L_bracket** dataset, stored in `L_Bracket/L_bracket_stress.h5`.

## Data Split Strategy

### Consistent Validation Set

The data splitting process ensures that:

1. **Reproducible Split**: The entire dataset is split into training (80%) and validation (20%) using a fixed random seed (`random_state=42`)
2. **Same Validation Data**: The validation set remains constant across all experiments, regardless of training data size
3. **Training Subsampling**: When using a subset of training data (e.g., 40%), only the training set is subsampled, keeping the validation set intact

### Implementation Details

```python
# Step 1: Split into train/val (80/20) with fixed seed
train_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=42)

# Step 2: Subsample training data if data_perc < 100
if data_perc < 100:
    set_seed(42)  # Ensures reproducible subsampling
    n_train = len(train_idx)
    n_subsample = max(1, int(n_train * data_perc / 100))
    train_idx = random.sample(train_idx, n_subsample)
```

### Normalization

Normalization statistics (coordinate center/range and stress mean/std) are computed **only from the training data** after subsampling. This ensures:
- No data leakage from validation set
- Consistent normalization approach across all experiments
- Fair comparison between models trained on different data percentages

## Model Architectures

### 1. PointNetMLPJoint
Located in `PointNetMLPJoint/`
- Uses PointNet-style encoder with set abstraction blocks
- Configurable via `model_presets.json`
- Available presets: S, M, L, XL, XXL

### 2. VanillaDeepONet
Located in `VanillaDeepONet/`
- Uses DeepONet architecture with branch and trunk networks
- Configurable via `model_presets.json`
- Available presets: L (and other sizes)

## Training Configuration

### Key Parameters
- **Epochs**: 500 (with early stopping)
- **Early Stopping Patience**: 100 epochs
- **Learning Rate**: 3e-4 with OneCycleLR scheduler
- **Batch Size**: Dynamically adjusted based on GPU memory
- **Training Mode**: `batched_all` (uses all nodes per geometry)

### Data Percentages
Common training data percentages used:
- 40% of training data
- 100% of training data

## Running Experiments

### Single Model Training

```python
# PointNetMLPJoint
python PointNetMLPJoint/Training_script.py

# VanillaDeepONet
python VanillaDeepONet/Training_script.py
```

### GPU Job Scripts

For batch processing on GPU clusters:

```bash
# Run preset configurations on GPU0
python PointNetMLPJoint/GPU0.py
python VanillaDeepONet/GPU0.py

# Run preset configurations on GPU1
python PointNetMLPJoint/GPU1.py
python VanillaDeepONet/GPU1.py
```

### SLURM Batch Jobs

```bash
sbatch run0.sh  # PointNetMLPJoint GPU0
sbatch run1.sh  # PointNetMLPJoint GPU1
sbatch run2.sh  # VanillaDeepONet GPU0
sbatch run3.sh  # VanillaDeepONet GPU1
```

## Model Checkpoints

Trained models are saved in `{ModelName}/Trained_models/` with the following naming convention:

```
{geom_prefix}_{data_perc}p_{model_name}_{arch_hash}.pt
```

Example: `L-_40p_pnmlp_a1b2c3d4.pt`

### Checkpoint Contents

Each `.pt` file contains:
- `model_state`: Model state dictionary
- `arch`: Architecture configuration
- `coord_center`, `coord_half_range`: Coordinate normalization parameters
- `stress_mean`, `stress_std`: Stress normalization parameters
- `config`: Training configuration (epochs trained, best validation loss)
- `best_val_loss`: Best validation MSE
- **`train_loss_history`**: List of training MSE loss per epoch
- **`val_loss_history`**: List of validation MSE loss per epoch

### Loading and Plotting Loss History

```python
import torch
import matplotlib.pyplot as plt

# Load checkpoint
ckpt = torch.load('path/to/model.pt')

# Extract loss histories
train_losses = ckpt['train_loss_history']
val_losses = ckpt['val_loss_history']

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss History')
plt.legend()
plt.grid(True)
plt.show()
```

## Validation Data Consistency Verification

To verify that validation data is consistent across runs:

1. The validation set indices are determined by `train_test_split` with `random_state=42`
2. These indices are **never modified** - only training indices are subsampled
3. All models evaluate on the exact same validation geometries

## File Structure

```
Paper1_data_experiment/
├── L_Bracket/
│   └── L_bracket_stress.h5
├── PointNetMLPJoint/
│   ├── Training_script.py
│   ├── pn_models.py
│   ├── model_presets.json
│   ├── GPU0.py
│   ├── GPU1.py
│   └── Trained_models/
├── VanillaDeepONet/
│   ├── Training_script.py
│   ├── benchmarks.py
│   ├── pn_models.py
│   ├── model_presets.json
│   ├── GPU0.py
│   ├── GPU1.py
│   └── Trained_models/
├── run0.sh
├── run1.sh
├── run2.sh
├── run3.sh
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- h5py
- numpy
- scikit-learn
- CUDA-capable GPU (optional but recommended)

## Notes

- The seed is set to 42 throughout the codebase for reproducibility
- Training uses mixed precision (AMP) when CUDA is available
- Gradient clipping is enabled (max norm: 0.5)
- Models implement stress-weighted loss to focus on high-stress regions
