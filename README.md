# Explainable Computer Vision: A Pure NumPy Implementation

An end-to-end convolutional neural network framework built from first principles using only NumPy. This implementation exposes the fundamental mechanics of deep learning by eliminating high-level abstractions, providing a transparent view into backpropagation dynamics, optimization trajectories, and model interpretability.


## Architecture

**Network Configuration:**
- 4 convolutional layers with progressive channel expansion (1→8→16→32→64)
- ReLU activation functions for non-linear transformations
- 2×2 max pooling for spatial downsampling
- 2 fully-connected layers with 128 hidden units
- Softmax output layer for 10-class classification

**Computational Primitives:**
- Manual implementation of forward/backward passes
- Stochastic gradient descent optimization
- Cross-entropy loss with numerical stability
- He initialization for weight matrices

## Key Features

### Real-Time Training Dynamics
- Live visualization of loss convergence and accuracy evolution
- Per-batch and per-epoch metric tracking
- Convolutional kernel visualization during optimization
- Performance profiling with confusion matrices and per-class metrics

### Model Interpretability
- **Activation Maps:** Hierarchical feature representations across network depth
- **Grad-CAM:** Gradient-weighted attention heatmaps revealing discriminative spatial regions
- **Kernel Visualization:** Learned pattern detectors at each convolutional layer

### Pure Implementation
- Zero dependency on PyTorch, TensorFlow, or JAX
- Educational transparency into backpropagation mechanics
- Explicit gradient computation and parameter updates

## Installation

```powershell
# Clone repository
git clone https://github.com/stryker13799/explainable-cnn.git
cd explainable-cnn

# Install dependencies
pip install -r requirements.txt
```

## Usage

Launch the interactive Streamlit interface:

```powershell
streamlit run app.py
```

The application provides two primary modes:

1. **Live Training Dashboard:** Configure hyperparameters (epochs, batch size, learning rate) and observe optimization in real-time with continuous metric updates and kernel evolution.

2. **Inference Explorer:** Analyze trained model behavior through activation maps and Grad-CAM visualizations, revealing spatial attention mechanisms and learned feature hierarchies.

## Dataset

**MNIST Handwritten Digits**
- Training subset: 2,000 samples
- Validation subset: 1,000 samples  
- Input dimensionality: 28×28 grayscale images
- Classes: 10 (digits 0-9)

Subset selection enables rapid experimentation (~20-30 seconds per training epoch on standard CPU).

## Technical Details

- **Numerically Stable Softmax:** Log-sum-exp trick prevents overflow
- **Efficient Convolution:** Vectorized operations minimize Python loops
- **Gradient Verification:** Backpropagation correctness validated through numerical differentiation
- **Memory Optimization:** Checkpoint-free training with cached activations for backward pass

## Interpretability Methodology

**Grad-CAM** (Gradient-weighted Class Activation Mapping) visualizes discriminative regions by:
1. Computing gradients of target class score w.r.t. target convolutional layer
2. Global average pooling of gradients to obtain importance weights
3. Weighted combination of forward activation maps
4. ReLU application to highlight positive contributions

This reveals where the network "looks" when making predictions, enabling verification of learned representations.

## Project Structure

```
explainable-cnn/
├── app.py              # Streamlit interface with training/inference modes
├── numpy_nn.py         # Core CNN implementation (layers, optimizers, Grad-CAM)
├── data_loader.py      # MNIST data acquisition and preprocessing
├── requirements.txt    # Dependency specifications
├── model_weights.npz   # Serialized trained parameters (auto-generated)
└── data/              # MNIST dataset cache (auto-downloaded)
``