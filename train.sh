#!/bin/bash

# Activate your virtual environment if using one
# source /path/to/your/venv/bin/activate

# Install requirements
pip install -e .

# Run training with desired parameters
python run.py \
    --epochs 50 \
    --batch-size 16 \
    --aug-strategy baseline
