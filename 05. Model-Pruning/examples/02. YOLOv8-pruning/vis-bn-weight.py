import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics import YOLO

def visualize_bn_gamma_distribution(model):
    """
    Extract gamma weights from Batch Normalization layers of the model
    and visualize their distribution.

    Args:
        model: The deep learning model (e.g., PyTorch or TensorFlow/Keras model).
    """
    gamma_weights = []

    # Example for PyTorch model
    for layer in model.modules():
        if hasattr(layer, 'weight') and isinstance(layer, torch.nn.BatchNorm2d):
            gamma_weights.append(layer.weight.detach().cpu().numpy())

    # Flatten the list of gamma weights
    gamma_weights = np.concatenate(gamma_weights)

    # Plot the distribution
    plt.figure(figsize=(8, 5))
    plt.hist(gamma_weights, bins=100, color='blue', alpha=0.7)
    plt.title("Distribution of Gamma Weights (BN Layers)")
    plt.xlabel("Gamma Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig('bn-weight-distribution.jpg')

if __name__ == '__main__':
    # weight = "runs/train-norm/weights/best.pt"
    weight = "runs/train-sparsity/weights/last.pt"
    model = YOLO(weight)
    visualize_bn_gamma_distribution(model)
