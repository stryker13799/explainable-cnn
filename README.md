# Explainable CNN: A Pure NumPy Implementation

A CNN implementation from scratch using only NumPy. No PyTorch/TensorFlow/JAX, just gradient descent and matrix operations.

**Live Demo:** https://explainable-cnn-scratch.streamlit.app/


## Architecture

4 conv layers (1→8→16→32→64 channels), ReLU activations, 2×2 max pooling, 2 FC layers (128 units), softmax output. SGD optimization with He initialization and numerically stable cross-entropy loss.

## Features

- **Training Dynamics:** Live loss/accuracy curves, kernel evolution, confusion matrices
- **Interpretability:** Activation maps, Grad-CAM attention heatmaps, learned kernel visualizations
- **Pure NumPy:** Explicit backprop implementation, no DL frameworks

## Installation

```powershell
# Clone repository
git clone https://github.com/stryker13799/explainable-cnn.git
cd explainable-cnn

# Install dependencies
pip install -r requirements.txt
```

## Usage

```powershell
streamlit run app.py
```

**Training mode:** Configure hyperparameters and observe real-time optimization metrics.
**Inference mode:** Visualize activation maps and Grad-CAM attention for model behavior analysis.

## Dataset

MNIST handwritten digits: 2k training samples, 1k validation samples (28×28 grayscale, 10 classes). Subset enables ~20-30s epoch times on CPU.

## Technical Details

Numerically stable softmax (log-sum-exp trick), vectorized convolutions, gradient-verified backprop, cached activations for backward pass.

## Grad-CAM

Gradient-weighted Class Activation Mapping: computes ∂y_c/∂A^k, applies global average pooling for importance weights, produces weighted activation combination with ReLU to visualize discriminative spatial regions.

## Project Structure

```
explainable-cnn/
├── app.py              # Streamlit interface with training/inference modes
├── numpy_nn.py         # Core CNN implementation (layers, optimizers, Grad-CAM)
├── data_loader.py      # MNIST data acquisition and preprocessing
├── requirements.txt    # Dependency specifications
├── model_weights.npz   # Serialized trained parameters (auto-generated)
└── data/              # MNIST dataset cache (auto-downloaded)
```
