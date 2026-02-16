#!/bin/bash

# Local testing script for VAE training component
set -e

echo "🧪 Setting up local testing environment for VAE training component"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Not in a virtual environment. Consider creating one:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo ""
fi

# Install test dependencies
echo "📦 Installing test dependencies..."
pip install -r requirements-test.txt

echo ""
echo "🔍 Running unit tests..."
python -m pytest ../../../../../tests/tensorflow/training/test_vae_training_local.py -v

echo ""
echo "🚀 Running integration test..."
python test_vae_local.py

echo ""
echo "✅ All tests completed!"
echo ""
echo "📝 Next steps:"
echo "   1. If all tests pass, your training script is ready for the pipeline"
echo "   2. You can modify hyperparameters in the test scripts to experiment"
echo "   3. Check the generated model and metrics files in the test output"
echo ""
echo "🔧 To test with your own data:"
echo "   python test_vae_local.py  # Modify the data generation in this script"
echo ""
echo "🧪 To run specific tests:"
echo "   python -m pytest ../../../../tests/tensorflow/training/test_vae_training_local.py::TestVAETrainingLocal::test_build_vae -v" 