#!/bin/bash

# Create and activate conda environment
conda env create -f environment.yml
conda activate cell-seg

# Install the package in development mode
pip install -e .

# Run the training
python run.py \
    --epochs 50 \
    --batch-size 16 \
    --aug-strategy baseline
