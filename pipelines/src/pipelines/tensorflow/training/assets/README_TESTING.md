# VAE Training Component - Local Testing Guide

This guide explains how to test the VAE (Variational Autoencoder) training component locally before deploying it to your Vertex AI pipeline.

## Overview

The VAE training component consists of:
- `train_vae.py` - Main training script
- `test_vae_local.py` - Local testing script with mock data
- `test_vae_training_local.py` - Unit tests for the component
- `test_local.sh` - Automated testing script

## Quick Start

### 1. Automated Testing (Recommended)

Run the automated test script:

```bash
cd pipelines/src/pipelines/tensorflow/training/assets/
./test_local.sh
```

This script will:
- Install required dependencies
- Run unit tests
- Run integration tests with mock data
- Provide feedback on test results

### 2. Manual Testing

If you prefer to run tests manually:

```bash
# Install dependencies
pip install -r requirements-test.txt

# Run unit tests
python -m pytest ../../../../tests/tensorflow/training/test_vae_training_local.py -v

# Run integration test
python test_vae_local.py
```

## Test Components

### Unit Tests (`test_vae_training_local.py`)

These tests verify individual components:

- **Model Construction**: Tests VAE encoder/decoder architecture
- **Data Loading**: Tests numpy dataset loading functionality
- **Training Step**: Tests single training iteration
- **Hyperparameter Variations**: Tests different model configurations
- **KL Weight Annealing**: Tests the KL divergence weight scheduling

### Integration Test (`test_vae_local.py`)

This test runs the complete training pipeline:

1. **Mock Data Generation**: Creates realistic time series data
2. **Model Training**: Runs the full training script
3. **Artifact Verification**: Checks that models and metrics are saved correctly

## Mock Data

The test scripts generate synthetic time series data with the following characteristics:

- **Shape**: `(n_samples, seq_length, n_features)`
- **Features**: 
  - Feature 1: Price-like signal with trend and seasonality
  - Feature 2: Rate of change (gradient)
  - Feature 3: Random noise
  - Feature 4: Cumulative trend

## Testing Your Own Data

To test with your own data, modify the `generate_mock_data()` function in `test_vae_local.py`:

```python
def generate_mock_data(n_samples=1000, seq_length=50, n_features=4):
    # Replace this with your data loading logic
    # Your data should be numpy arrays with shape (n_samples, seq_length, n_features)
    
    # Example: Load from CSV
    # import pandas as pd
    # df = pd.read_csv('your_data.csv')
    # data = df.values.reshape(-1, seq_length, n_features)
    
    # Example: Load from numpy file
    # data = np.load('your_data.npy')
    
    return train_data, valid_data
```

## Expected Output

Successful tests should produce:

### Unit Tests
```
✓ VAE model built successfully
✓ Encoder output shapes: (32, 3), (32, 3), (32, 3)
✓ Decoder output shape: (32, 50, 4)
✓ Full VAE output shape: (32, 50, 4)
✓ Training step completed
  - Loss: 0.123456
  - Reconstruction Loss: 0.098765
  - KL Loss: 0.024691
```

### Integration Test
```
✓ Model saved to: /tmp/tmpXXXXXX/model
  - Model files: ['saved_model.pb', 'variables/']
✓ Metrics saved to: /tmp/tmpXXXXXX/metrics.json
  - Final Loss: 0.123456
  - Reconstruction Loss: 0.098765
  - KL Loss: 0.024691
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the correct directory and Python path is set
2. **Memory Issues**: Reduce `batch_size` or `n_samples` in test scripts
3. **GPU Issues**: The tests run on CPU by default; modify `get_distribution_strategy()` if needed

### Debug Mode

Run tests with verbose output:

```bash
python -m pytest ../../../../tests/tensorflow/training/test_vae_training_local.py -v -s
```

### Specific Test

Run a specific test:

```bash
python -m pytest ../../../../tests/tensorflow/training/test_vae_training_local.py::TestVAETrainingLocal::test_build_vae -v
```

## Integration with Pipeline

Once local tests pass, your training script is ready for the pipeline. The script expects:

### Input Arguments
- `--train_data`: Path to training data (.npz file)
- `--valid_data`: Path to validation data (.npz file)
- `--model_dir`: Directory to save the trained model
- `--metrics`: Path to save training metrics (JSON)
- `--hparams`: JSON string of hyperparameters

### Output Artifacts
- Trained model in SavedModel format
- Training metrics in JSON format

## Hyperparameters

Default hyperparameters (can be overridden via `--hparams`):

```json
{
    "batch_size": 128,
    "epochs": 200,
    "learning_rate": 1e-3,
    "latent_dim": 3,
    "seq_length": 50,
    "n_features": 4,
    "kl_init_weight": 1e-4,
    "patience": 10
}
```

## Next Steps

After successful local testing:

1. **Deploy to Pipeline**: Your script is ready for the Vertex AI pipeline
2. **Monitor Training**: Check logs and metrics in Vertex AI console
3. **Optimize**: Tune hyperparameters based on validation performance
4. **Scale**: Increase data size and training epochs for production

## Support

If you encounter issues:

1. Check the test output for specific error messages
2. Verify your Python environment and dependencies
3. Ensure your data format matches the expected input
4. Review the training script logs for detailed error information 