 MNIST Handwritten text task

This project implements an end-to-end deep learning pipeline for handwritten digit classification using the MNIST dataset.
It covers data loading, preprocessing, CNN modeling, training, evaluation, deployment, and inference, with both notebook-based exploration and script-based training.
A trained model is deployed to Hugging Face Hub at (Anni0401/mnist-cnn).

How to run: 

Environment Setup
1. Create and activate a virtual environment
2. Install dependencies with pip install -r requirements.txt
3. Download the MNIST dataset (IDX format). Place the files in the data/ directory. 
4. The main training entry point is src/training/train.py.
This script is sweep-ready and supports Weights & Biases integration.
Run training locally
From the project root: python src/training/train.py
5. Hyperparameter Sweeps (Weights & Biases)
After defining a sweep configuration (YAML or dict): wandb sweep sweep.yaml
wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>
6. Inference
The inference.py module provides standalone inference utilities. It loads a trained model and predicts a digit. Input must be a 28×28 grayscale image.
7. Testing
Basic unit tests validate critical components such as the data loader.
Run all tests with: pytest
8. Web Demo
An interactive web demo of the deployed model can be tested on Huggingface Spaces -> Anni0401/mnist-cnn-demo
