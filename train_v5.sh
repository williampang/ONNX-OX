#!/bin/bash
# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Install requirements
echo "Installing requirements..."
./venv/bin/pip install -r requirements_Version5.txt

# Run training
echo "Starting training..."
./venv/bin/python train_oxnet_Version5.py --epochs 30 --classes "C,O,X"

echo "Training complete. Model saved to model/model_v5.onnx"
