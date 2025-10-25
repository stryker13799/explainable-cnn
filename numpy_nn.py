import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros(out_channels)
        self.cache = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        N, C, H, W = x.shape
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                         mode='constant') if self.padding > 0 else x
        
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = np.zeros((N, self.out_channels, H_out, W_out))
        
        # slide kernel over input, computing dot product at each position
        for h in range(H_out):
            for w in range(W_out):
                h_start, w_start = h * self.stride, w * self.stride
                h_end, w_end = h_start + self.kernel_size, w_start + self.kernel_size
                
                receptive_field = x_padded[:, :, h_start:h_end, w_start:w_end]
                out[:, :, h, w] = np.tensordot(receptive_field, self.weights, axes=([1, 2, 3], [1, 2, 3])) + self.bias
        
        self.cache = x
        return out
    
    def backward(self, dout):
        # compute gradients for weights and propagate gradient backwards
        x = self.cache
        N, C, H, W = x.shape
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                         mode='constant') if self.padding > 0 else x
        
        _, _, H_out, W_out = dout.shape
        dx_padded = np.zeros_like(x_padded)
        self.dW = np.zeros_like(self.weights)
        self.db = np.sum(dout, axis=(0, 2, 3))
        
        for h in range(H_out):
            for w in range(W_out):
                h_start, w_start = h * self.stride, w * self.stride
                h_end, w_end = h_start + self.kernel_size, w_start + self.kernel_size
                
                receptive_field = x_padded[:, :, h_start:h_end, w_start:w_end]
                dout_hw = dout[:, :, h, w]
                
                # accumulate weight gradients and backprop to input
                self.dW += np.tensordot(dout_hw, receptive_field, axes=([0], [0]))
                dx_padded[:, :, h_start:h_end, w_start:w_end] += np.tensordot(dout_hw, self.weights, axes=([1], [0]))
        
        dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] if self.padding > 0 else dx_padded
        return dx


class ReLU:
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        return dout * (self.cache > 0)


class MaxPool:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
    
    def forward(self, x):
        N, C, H, W = x.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        out = np.zeros((N, C, H_out, W_out))
        
        for h in range(H_out):
            for w in range(W_out):
                h_start, w_start = h * self.stride, w * self.stride
                h_end, w_end = h_start + self.pool_size, w_start + self.pool_size
                out[:, :, h, w] = np.max(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
        
        self.cache = x
        return out
    
    def backward(self, dout):
        # gradient only flows back through the max element in each window
        x = self.cache
        N, C, H, W = x.shape
        _, _, H_out, W_out = dout.shape
        dx = np.zeros_like(x)
        
        for h in range(H_out):
            for w in range(W_out):
                h_start, w_start = h * self.stride, w * self.stride
                h_end, w_end = h_start + self.pool_size, w_start + self.pool_size
                
                window = x[:, :, h_start:h_end, w_start:w_end]
                max_vals = np.max(window, axis=(2, 3), keepdims=True)
                mask = (window == max_vals)
                dx[:, :, h_start:h_end, w_start:w_end] += mask * dout[:, :, h:h+1, w:w+1]
        
        return dx


class Flatten:
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        # (batch, channels, h, w) -> (batch, channels*h*w)
        self.cache = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dout):
        return dout.reshape(self.cache)


class Dense:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros(out_features)
        self.cache = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        # x: (batch, in_features) -> (batch, out_features)
        self.cache = x
        return x @ self.weights + self.bias
    
    def backward(self, dout):
        x = self.cache
        self.dW = x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.weights.T


class SoftmaxCrossEntropy:
    def __init__(self):
        self.cache = None
    
    def forward(self, logits, labels):
        # numerically stable softmax
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        N = logits.shape[0]
        loss = -np.mean(np.log(probs[np.arange(N), labels] + 1e-8))
        
        self.cache = (probs, labels)
        return loss
    
    def backward(self):
        probs, labels = self.cache
        N = probs.shape[0]
        
        dlogits = probs.copy()
        dlogits[np.arange(N), labels] -= 1
        dlogits /= N
        
        return dlogits


class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def step(self, layers):
        for layer in layers:
            if hasattr(layer, 'weights'):
                layer.weights -= self.lr * layer.dW
                layer.bias -= self.lr * layer.db


class Sequential:
    def __init__(self, input_channels=1):
        self.layers = [
            Conv2D(input_channels, 8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2D(8, 16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool(pool_size=2, stride=2),
            Conv2D(16, 32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2D(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool(pool_size=2, stride=2),
            Flatten(),
            Dense(64 * 7 * 7, 128),
            ReLU(),
            Dense(128, 10)
        ]
        self.loss_fn = SoftmaxCrossEntropy()
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def get_conv_layers(self):
        return [layer for layer in self.layers if isinstance(layer, Conv2D)]
    
    def get_all_layers(self):
        return self.layers
    
    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)
    
    def compute_gradcam(self, x, target_layer_idx, target_class=None):
        # generate heatmap showing which spatial regions influenced the prediction
        activations = x
        for i, layer in enumerate(self.layers):
            activations = layer.forward(activations)
            if isinstance(layer, Conv2D) and i == target_layer_idx:
                target_activations = activations.copy()
        
        logits = activations
        target_class = target_class if target_class is not None else np.argmax(logits)
        
        # backprop from target class to get gradients at conv layer
        dout = np.zeros_like(logits)
        dout[0, target_class] = 1.0
        
        for layer in reversed(self.layers[target_layer_idx+1:]):
            dout = layer.backward(dout)
        
        # weight feature maps by their gradient importance
        weights = np.mean(dout, axis=(2, 3), keepdims=True)
        heatmap = np.sum(weights * target_activations, axis=1)[0]
        heatmap = np.maximum(heatmap, 0)
        
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        return heatmap
    
    def save_weights(self, filepath):
        weights_dict = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                weights_dict[f'layer_{i}_weights'] = layer.weights
            if hasattr(layer, 'bias'):
                weights_dict[f'layer_{i}_bias'] = layer.bias
        np.savez(filepath, **weights_dict)
    
    def load_weights(self, filepath):
        weights_data = np.load(filepath)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights') and f'layer_{i}_weights' in weights_data:
                layer.weights = weights_data[f'layer_{i}_weights']
            if hasattr(layer, 'bias') and f'layer_{i}_bias' in weights_data:
                layer.bias = weights_data[f'layer_{i}_bias']

