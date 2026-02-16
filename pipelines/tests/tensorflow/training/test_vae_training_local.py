# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import json
import numpy as np
import pytest
import tensorflow as tf
from pathlib import Path

# Add the training assets to the path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src" / "pipelines" / "tensorflow" / "training" / "assets"))

from train_vae import build_vae, load_numpy_dataset, main


class TestVAETrainingLocal:
    """Test VAE training script locally with mock data."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.fixture
    def mock_data(self):
        """Generate mock training and validation data."""
        # Generate synthetic time series data
        np.random.seed(42)
        n_samples = 1000
        seq_length = 50
        n_features = 4
        
        # Create realistic time series data
        time_steps = np.linspace(0, 10, seq_length)
        data = []
        
        for _ in range(n_samples):
            # Generate base signal
            base_signal = np.sin(time_steps) + 0.1 * np.random.randn(seq_length)
            
            # Add features
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
        
        return train_data, valid_data
    
    @pytest.fixture
    def data_files(self, temp_dir, mock_data):
        """Save mock data to temporary files."""
        train_data, valid_data = mock_data
        
        train_file = os.path.join(temp_dir, "train_data.npz")
        valid_file = os.path.join(temp_dir, "valid_data.npz")
        
        np.savez(train_file, arr_0=train_data)
        np.savez(valid_file, arr_0=valid_data)
        
        return train_file, valid_file
    
    def test_build_vae(self):
        """Test that VAE model can be built successfully."""
        latent_dim = 3
        seq_length = 50
        n_features = 4
        kl_init_weight = 1e-4
        
        vae = build_vae(latent_dim, seq_length, n_features, kl_init_weight)
        
        # Check model structure
        assert vae is not None
        assert hasattr(vae, 'encoder')
        assert hasattr(vae, 'decoder')
        assert vae.kl_weight == kl_init_weight
        
        # Test forward pass
        batch_size = 32
        test_input = tf.random.normal((batch_size, seq_length, n_features))
        
        # Test encoder
        z_mean, z_log_var, z = vae.encoder(test_input)
        assert z_mean.shape == (batch_size, latent_dim)
        assert z_log_var.shape == (batch_size, latent_dim)
        assert z.shape == (batch_size, latent_dim)
        
        # Test decoder
        decoded = vae.decoder(z)
        assert decoded.shape == (batch_size, seq_length, n_features)
        
        # Test full VAE
        reconstructed = vae(test_input)
        assert reconstructed.shape == (batch_size, seq_length, n_features)
    
    def test_load_numpy_dataset(self, data_files):
        """Test dataset loading functionality."""
        train_file, _ = data_files
        
        # Test loading with shuffle
        ds = load_numpy_dataset(train_file, batch_size=32, shuffle=True)
        assert ds is not None
        
        # Check first batch
        for batch in ds.take(1):
            assert batch.shape[0] <= 32  # batch size
            assert batch.shape[1] == 50  # seq_length
            assert batch.shape[2] == 4   # n_features
            break
    
    # def test_vae_training_step(self):
    #     """Test that VAE training step works correctly."""
    #     vae = build_vae(latent_dim=3, seq_length=50, n_features=4, kl_init_weight=1e-4)
    #     optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3)
    #     vae.compile(optimizer=optimizer)
        
    #     # Create test data
    #     batch_size = 16
    #     test_data = tf.random.normal((batch_size, 50, 4))
        
    #     # Test training step
    #     metrics = vae.train_step(test_data)
        
    #     # Check that metrics are returned
    #     assert 'loss' in metrics
    #     assert 'reconstruction_loss' in metrics
    #     assert 'kl_loss' in metrics
        
    #     # Check that losses are finite
    #     assert tf.math.is_finite(metrics['loss'])
    #     assert tf.math.is_finite(metrics['reconstruction_loss'])
    #     assert tf.math.is_finite(metrics['kl_loss'])
    
    # def test_vae_training_script_main(self, data_files, temp_dir):
    #     """Test the main training script with mock data."""
    #     train_file, valid_file = data_files
    #     model_dir = os.path.join(temp_dir, "model")
    #     metrics_file = os.path.join(temp_dir, "metrics.json")
        
    #     # Set up command line arguments
    #     import argparse
    #     from unittest.mock import patch
        
    #     test_args = [
    #         '--train_data', train_file,
    #         '--valid_data', valid_file,
    #         '--model_dir', model_dir,
    #         '--metrics', metrics_file,
    #         '--hparams', '{"batch_size": 32, "epochs": 2, "learning_rate": 1e-3}'
    #     ]
        
    #     with patch('sys.argv', ['train_vae.py'] + test_args):
    #         # Run the main function
    #         main()
        
    #     # Check that model was saved
    #     assert os.path.exists(model_dir)
    #     assert os.path.exists(os.path.join(model_dir, "saved_model.pb"))
        
    #     # Check that metrics were written
    #     assert os.path.exists(metrics_file)
    #     with open(metrics_file, 'r') as f:
    #         metrics = json.load(f)
        
    #     # Check that metrics contain expected keys
    #     assert 'loss' in metrics
    #     assert 'reconstruction_loss' in metrics
    #     assert 'kl_loss' in metrics
    
    def test_vae_with_different_hyperparameters(self):
        """Test VAE with different hyperparameter configurations."""
        configs = [
            {'latent_dim': 2, 'seq_length': 30, 'n_features': 3},
            {'latent_dim': 5, 'seq_length': 100, 'n_features': 6},
            {'latent_dim': 1, 'seq_length': 20, 'n_features': 2},
        ]
        
        for config in configs:
            vae = build_vae(
                latent_dim=config['latent_dim'],
                seq_length=config['seq_length'],
                n_features=config['n_features'],
                kl_init_weight=1e-4
            )
            
            # Test forward pass
            batch_size = 8
            test_input = tf.random.normal((batch_size, config['seq_length'], config['n_features']))
            output = vae(test_input)
            
            assert output.shape == (batch_size, config['seq_length'], config['n_features'])
    
    # def test_kl_weight_annealing(self):
    #     """Test that KL weight annealing works correctly."""
    #     vae = build_vae(latent_dim=3, seq_length=50, n_features=4, kl_init_weight=1e-4)
    #     optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3)
    #     vae.compile(optimizer=optimizer)
        
    #     initial_kl_weight = vae.kl_weight
        
    #     # Run several training steps
    #     batch_size = 16
    #     test_data = tf.random.normal((batch_size, 50, 4))
        
    #     for _ in range(5):
    #         vae.train_step(test_data)
        
    #     # Check that KL weight increased
    #     assert vae.kl_weight > initial_kl_weight
    #     assert vae.kl_weight <= 1.0  # Should be capped at 1.0


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 