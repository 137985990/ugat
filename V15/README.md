# V13 Multi-modal Time Series Training Project Setup Guide

## Project Overview
V13 is a PyTorch-based multi-modal time series model training project that supports:
- Multiple datasets (FM/OD/MEFAR) mixed training
- Dynamic channel masking with configurable strategies
- Graph Attention U-Net architecture with Transformer bottleneck
- Robust data processing and validation workflows

## Essential Files and Directory Structure

### Core Files (Required)
```
V13/
├── train.py                 # Main training script
├── data.py                  # Data loading and processing
├── model.py                 # Model architecture definition
├── graph.py                 # Graph construction utilities
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
└── simple_multimodal_integration.py  # Optional: Enhanced criterion
```

### Data Files (Required)
```
Data/
├── FM_original.csv          # FM dataset
├── OD_original.csv          # OD dataset
└── MEFAR_original.csv       # MEFAR dataset
```

### Generated Directories (Auto-created)
```
V13/
├── Checkpoints/             # Model checkpoints
├── Logs/                    # Training logs
└── runs/                    # TensorBoard logs
```

## Installation and Setup

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# For GPU support with CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Verify Data Files
Ensure the following data files exist:
- `../Data/FM_original.csv`
- `../Data/OD_original.csv`
- `../Data/MEFAR_original.csv`

Each CSV file must contain:
- A `block` column for grouping
- A label column `F` (uppercase)
- All modality columns as specified in `config.yaml`

### 3. Configuration
Edit `config.yaml` to customize:
- Data file paths
- Modality mappings for each dataset
- Training parameters
- Masking strategies

## Usage

### Basic Training
```bash
python train.py
```

### Training with Custom Parameters
```bash
python train.py --epochs 100 --batch_size 32 --learning_rate 0.001
```

### Training with Specific Device
```bash
python train.py --device cuda:0  # Use specific GPU
python train.py --device cpu     # Use CPU only
```

### Debug Mode
```bash
python train.py --debug  # Enable debug logging
```

## Configuration Details

### Key Configuration Parameters

#### Data Configuration
```yaml
data_dir: Data
data_files:
  - ../Data/FM_original.csv
  - ../Data/OD_original.csv
  - ../Data/MEFAR_original.csv
block_col: block
label_col: F  # Must be uppercase F
```

#### Modality Configuration
```yaml
common_modalities:  # Shared across all datasets
  - acc_x
  - acc_y
  - acc_z
  - ppg
  - gsr
  - hr
  - skt

dataset_modalities:
  FM:
    have: [list of FM-specific channels]
    need: [channels FM needs but doesn't have]
  OD:
    have: [list of OD-specific channels]
    need: [channels OD needs but doesn't have]
  MEFAR:
    have: []  # MEFAR typically has no unique channels
    need: [all channels MEFAR needs]
```

#### Training Configuration
```yaml
training:
  window_size: 300
  batch_size: 16
  epochs: 50
  learning_rate: 0.0001
  patience: 10
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

#### Masking Configuration
```yaml
masking:
  unmasked_datasets: ['MEFAR']  # Datasets to skip masking
  mask_ratio: 0.5               # Ratio of channels to mask
  mask_strategy: 'random'       # 'random', 'sequential', etc.
```

## Key Features

### 1. Dynamic Channel Masking
- Masks "have" channels during training to learn cross-modal relationships
- Supports dataset-specific unmasking (e.g., MEFAR remains unmasked)
- Configurable masking ratios and strategies

### 2. Multi-Dataset Support
- Combines multiple datasets in a single training run
- Handles different channel configurations per dataset
- Automatic source dataset tracking for loss computation

### 3. Robust Data Processing
- Sliding window sampling with configurable window size
- Automatic data validation and error handling
- Support for missing channels and flexible data formats

### 4. Advanced Training Features
- Mixed precision training (if available)
- Gradient clipping for stability
- Learning rate scheduling
- Early stopping with patience
- Comprehensive logging and monitoring

## Troubleshooting

### Common Issues

1. **Missing Data Files**
   - Verify data files exist at specified paths in `config.yaml`
   - Check file permissions and accessibility

2. **CUDA Out of Memory**
   - Reduce `batch_size` in config or command line
   - Use gradient accumulation if needed
   - Consider using CPU training for debugging

3. **Missing Dependencies**
   - Reinstall from `requirements.txt`
   - Check PyTorch compatibility with your CUDA version

4. **Configuration Errors**
   - Validate `config.yaml` syntax
   - Ensure all required modalities are defined
   - Check label column name (must be uppercase 'F')

5. **Model Architecture Issues**
   - Verify `model.py` and `graph.py` are present
   - Check for import errors in model modules

### Debug Tools
The project includes several debug utilities:
- `debug_dataset.py` - Test dataset loading
- `debug_dataloader.py` - Test DataLoader functionality
- `test_collate.py` - Test collate function
- `test_dataset.py` - Test individual dataset components

## Performance Optimization

### For Better Training Performance
1. Use GPU if available
2. Enable mixed precision training (automatic if supported)
3. Tune batch size based on GPU memory
4. Use appropriate number of DataLoader workers
5. Consider data preprocessing and caching

### For Memory Efficiency
1. Reduce window size or batch size
2. Use gradient accumulation for effective larger batches
3. Enable gradient checkpointing if implemented
4. Monitor memory usage during training

## Monitoring and Logging

### TensorBoard
```bash
tensorboard --logdir=runs
```

### Log Files
- Training logs: `Logs/training_YYYYMMDD_HHMMSS.log`
- Model checkpoints: `Checkpoints/`
- Console output with progress bars and metrics

## Extension Points

### Adding New Datasets
1. Add data file to `config.yaml`
2. Define modality mappings
3. Update masking strategies if needed

### Custom Loss Functions
Implement in `simple_multimodal_integration.py` or create new modules

### Model Architecture Changes
Modify `model.py` and `graph.py` for architecture experiments

This setup guide ensures you have all the necessary components and knowledge to successfully run the V13 multi-modal time series training project.
