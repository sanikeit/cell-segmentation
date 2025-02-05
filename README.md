# Cell Segmentation Project

Deep learning-based cell segmentation using PyTorch and the MoNuSeg dataset.

## Overview

The MoNuSeg dataset is used for training and evaluating segmentation models for cell images. This project includes functionalities for data loading, preprocessing, model training, and evaluation.

## Installation

To set up the project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd cell-segmentation
pip install -r requirements.txt
```

## Usage

1. **Data Loading**: The dataset can be loaded using the `Dataset` class defined in `src/data/dataset.py`. This class handles loading and preprocessing the MoNuSeg dataset, including image patching.

2. **Training the Model**: To train the UNet model, run the following command:

```bash
python src/train.py
```

3. **Data Exploration**: Use the Jupyter notebook located in `notebooks/data_exploration.ipynb` to visually inspect the dataset and explore its characteristics.

## Configuration

Configuration settings can be modified in `src/config.py` or `configs/default.yaml` to adjust paths and model parameters.

## Evaluation

Evaluation metrics such as accuracy, precision, and recall can be calculated using the functions defined in `src/utils/metrics.py`.

## Visualization

Training progress and results can be visualized using the functions in `src/utils/visualization.py`.

## License

This project is licensed under the MIT License.

## Project Structure