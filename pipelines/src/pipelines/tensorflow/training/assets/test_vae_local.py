#!/usr/bin/env python3
"""
Local testing script for VAE training component.
This script generates mock data and runs the training script locally.
"""

import os
import tempfile
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# Import the training script
from train_vae import build_vae, load_numpy_dataset, main


def generate_mock_data(n_samples=1000, seq_length=50, n_features=4):
    """Generate realistic mock time series data for testing."""
    print(f"Generating {n_samples} samples with shape ({seq_length}, {n_features})")
    
    np.random.seed(42)
    
    # Create realistic time series data
    time_steps = np.linspace(0, 10, seq_length)
    data = []
    
    for i in range(n_samples):
        # Generate base signal with some trend and seasonality
        trend = 0.1 * time_steps
        seasonality = 0.5 * np.sin(2 * np.pi * time_steps / 5)
        noise = 0.1 * np.random.randn(seq_length)
        base_signal = trend + seasonality + noise
        
        # Create features
        feature1 = base_signal  # Price-like feature
        feature2 = np.gradient(base_signal)  # Rate of change
        feature3 = np.random.randn(seq_length) * 0.1  # Noise feature
        feature4 = np.cumsum(np.random.randn(seq_length) * 0.01)  # Trend feature
        
        sample = np.column_stack([feature1, feature2, feature3, feature4])
        data.append(sample)
    
    data = np.array(data)
    
    # Split into train/valid
    split_idx = int(0.8 * n_samples)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Valid data shape: {valid_data.shape}")
    
    return train_data, valid_data


def test_vae_model():
    """Test VAE model construction and forward pass."""
    print("\n=== Testing VAE Model Construction ===")
    
    latent_dim = 3
    seq_length = 50
    n_features = 4
    kl_init_weight = 1e-4
    
    vae = build_vae(latent_dim, seq_length, n_features, kl_init_weight)
    print(f"✓ VAE model built successfully")
    print(f"  - Latent dimension: {latent_dim}")
    print(f"  - Sequence length: {seq_length}")
    print(f"  - Features: {n_features}")
    print(f"  - Initial KL weight: {kl_init_weight}")
    
    # Test forward pass
    batch_size = 32
    test_input = tf.random.normal((batch_size, seq_length, n_features))
    
    # Test encoder
    z_mean, z_log_var, z = vae.encoder(test_input)
    print(f"✓ Encoder output shapes:")
    print(f"  - z_mean: {z_mean.shape}")
    print(f"  - z_log_var: {z_log_var.shape}")
    print(f"  - z: {z.shape}")
    
    # Test decoder
    decoded = vae.decoder(z)
    print(f"✓ Decoder output shape: {decoded.shape}")
    
    # Test full VAE
    reconstructed = vae(test_input)
    print(f"✓ Full VAE output shape: {reconstructed.shape}")
    
    return vae


def test_training_step(vae):
    """Test a single training step."""
    print("\n=== Testing Training Step ===")
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3)
    vae.compile(optimizer=optimizer)
    
    # Create test data
    batch_size = 16
    test_data = tf.random.normal((batch_size, 50, 4))
    
    # Test training step
    metrics = vae.train_step(test_data)
    
    print(f"✓ Training step completed")
    print(f"  - Loss: {metrics['loss']:.6f}")
    print(f"  - Reconstruction Loss: {metrics['reconstruction_loss']:.6f}")
    print(f"  - KL Loss: {metrics['kl_loss']:.6f}")
    print(f"  - KL Weight: {vae.kl_weight:.6f}")


def run_full_training():
    """Run the full training script with mock data."""
    print("\n=== Running Full Training Script ===")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Generate mock data
        train_data, valid_data = generate_mock_data(n_samples=500, seq_length=50, n_features=4)
        
        # Save data to files
        train_file = os.path.join(temp_dir, "train_data.npz")
        valid_file = os.path.join(temp_dir, "valid_data.npz")
        
        np.savez(train_file, arr_0=train_data)
        np.savez(valid_file, arr_0=valid_data)
        
        print(f"✓ Data saved to:")
        print(f"  - Train: {train_file}")
        print(f"  - Valid: {valid_file}")
        
        # Set up model and metrics paths
        model_dir = os.path.join(temp_dir, "model")
        metrics_file = os.path.join(temp_dir, "metrics.json")
        
        # Set up command line arguments
        import sys
        from unittest.mock import patch
        
        test_args = [
            '--train_data', train_file,
            '--valid_data', valid_file,
            '--model_dir', model_dir,
            '--metrics', metrics_file,
            '--hparams', '{"batch_size": 32, "epochs": 3, "learning_rate": 1e-3, "patience": 2}'
        ]
        
        print(f"Running training with args: {test_args}")
        
        with patch('sys.argv', ['train_vae.py'] + test_args):
            # Run the main function
            main()
        
        # Check results
        print(f"\n=== Training Results ===")
        
        # Check that model was saved
        if os.path.exists(model_dir):
            print(f"✓ Model saved to: {model_dir}")
            model_files = os.listdir(model_dir)
            print(f"  - Model files: {model_files}")
        else:
            print(f"✗ Model directory not found: {model_dir}")
        
        # Check that metrics were written
        if os.path.exists(metrics_file):
            print(f"✓ Metrics saved to: {metrics_file}")
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            print(f"  - Final Loss: {metrics.get('loss', 'N/A'):.6f}")
            print(f"  - Reconstruction Loss: {metrics.get('reconstruction_loss', 'N/A'):.6f}")
            print(f"  - KL Loss: {metrics.get('kl_loss', 'N/A'):.6f}")
        else:
            print(f"✗ Metrics file not found: {metrics_file}")


def main():
    """Run all local tests."""
    print("🚀 Starting VAE Training Component Local Tests")
    print("=" * 50)
    
    try:
        # Test 1: Model construction
        vae = test_vae_model()
        
        # Test 2: Training step
        test_training_step(vae)
        
        # Test 3: Full training script
        run_full_training()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 