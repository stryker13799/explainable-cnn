import numpy as np
from numpy.lib.stride_tricks import as_strided


def _pad_input(x, padding):
    if padding == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')


def _im2col(x, kernel, stride, padding):
    x_padded = _pad_input(x, padding)
    N, C, H_p, W_p = x_padded.shape
    out_h = (H_p - kernel) // stride + 1
    out_w = (W_p - kernel) // stride + 1
    shape = (N, C, out_h, out_w, kernel, kernel)
    strides = (
        x_padded.strides[0],
        x_padded.strides[1],
        x_padded.strides[2] * stride,
        x_padded.strides[3] * stride,
        x_padded.strides[2],
        x_padded.strides[3],
    )
    windows = as_strided(x_padded, shape=shape, strides=strides, writeable=False)
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N * out_h * out_w, -1)
    return cols, x_padded, out_h, out_w


def _col2im(cols, x_shape, kernel, stride, padding):
    N, C, H, W = x_shape
    H_p = H + 2 * padding
    W_p = W + 2 * padding
    out_h = (H_p - kernel) // stride + 1
    out_w = (W_p - kernel) // stride + 1
    cols_reshaped = cols.reshape(N, out_h, out_w, C, kernel, kernel).transpose(0, 3, 4, 5, 1, 2)
    x_padded = np.zeros((N, C, H_p, W_p), dtype=cols.dtype)
    for i in range(kernel):
        for j in range(kernel):
            x_padded[:, :, i:i + out_h * stride:stride, j:j + out_w * stride:stride] += cols_reshaped[:, :, i, j]
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def _pool_windows(x, pool, stride):
    N, C, H, W = x.shape
    out_h = (H - pool) // stride + 1
    out_w = (W - pool) // stride + 1
    shape = (N, C, out_h, out_w, pool, pool)
    strides = (
        x.strides[0],
        x.strides[1],
        x.strides[2] * stride,
        x.strides[3] * stride,
        x.strides[2],
        x.strides[3],
    )
    windows = as_strided(x, shape=shape, strides=strides, writeable=False)
    return windows, out_h, out_w


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = (np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale).astype(np.float32)
        self.bias = np.zeros(out_channels, dtype=np.float32)
        self.cache = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        x = x.astype(np.float32, copy=False)
        cols, x_padded, out_h, out_w = _im2col(x, self.kernel_size, self.stride, self.padding)
        cols = np.ascontiguousarray(cols, dtype=np.float32)
        W_col = self.weights.reshape(self.out_channels, -1)
        out = (cols @ W_col.T).astype(np.float32, copy=False)
        out = out.reshape(x.shape[0], out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        out += self.bias.reshape(1, -1, 1, 1)
        self.cache = {
            'input_shape': x.shape,
            'cols': cols,
            'x_padded_shape': x_padded.shape,
            'out_h': out_h,
            'out_w': out_w
        }
        return out
    
    def backward(self, dout):
        cache = self.cache
        cols = cache['cols']
        N = cache['input_shape'][0]
        out_h = cache['out_h']
        out_w = cache['out_w']
        dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)
        self.db = dout_reshaped.sum(axis=1).astype(np.float32, copy=False)
        self.dW = (dout_reshaped @ cols).reshape(self.weights.shape).astype(np.float32, copy=False)
        W_col = self.weights.reshape(self.out_channels, -1)
        dcols = dout_reshaped.T @ W_col
        dx = _col2im(dcols, cache['input_shape'], self.kernel_size, self.stride, self.padding)
        self.cache = None
        return dx.astype(np.float32, copy=False)


class ReLU:
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        x = x.astype(np.float32, copy=False)
        self.cache = x
        return np.maximum(x, 0)
    
    def backward(self, dout):
        out = dout * (self.cache > 0)
        self.cache = None
        return out


class MaxPool:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
    
    def forward(self, x):
        windows, out_h, out_w = _pool_windows(x, self.pool_size, self.stride)
        flat = windows.reshape(x.shape[0], x.shape[1], out_h, out_w, -1)
        out = flat.max(axis=-1)
        max_idx = flat.argmax(axis=-1).astype(np.int32)
        self.cache = {
            'input_shape': x.shape,
            'out_h': out_h,
            'out_w': out_w,
            'max_idx': max_idx
        }
        return out
    
    def backward(self, dout):
        cache = self.cache
        input_shape = cache['input_shape']
        N, C, _, _ = input_shape
        out_h = cache['out_h']
        out_w = cache['out_w']
        max_idx = cache['max_idx'].reshape(-1)
        dx = np.zeros(input_shape, dtype=dout.dtype)
        pool = self.pool_size
        stride = self.stride
        n_idx = np.repeat(np.arange(N), C * out_h * out_w)
        c_idx = np.tile(np.repeat(np.arange(C), out_h * out_w), N)
        h_idx = np.tile(np.repeat(np.arange(out_h), out_w), N * C)
        w_idx = np.tile(np.arange(out_w), N * C * out_h)
        h_offsets = max_idx // pool
        w_offsets = max_idx % pool
        h_final = h_idx * stride + h_offsets
        w_final = w_idx * stride + w_offsets
        np.add.at(dx, (n_idx, c_idx, h_final, w_final), dout.reshape(-1))
        self.cache = None
        return dx


class Flatten:
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        # (batch, channels, h, w) -> (batch, channels*h*w)
        self.cache = x.shape
        return x.reshape(x.shape[0], -1).astype(np.float32, copy=False)
    
    def backward(self, dout):
        out = dout.reshape(self.cache)
        self.cache = None
        return out


class Dense:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        scale = np.sqrt(2.0 / in_features)
        self.weights = (np.random.randn(in_features, out_features) * scale).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)
        self.cache = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        x = x.astype(np.float32, copy=False)
        self.cache = x
        return x @ self.weights + self.bias
    
    def backward(self, dout):
        x = self.cache
        self.dW = (x.T @ dout).astype(np.float32, copy=False)
        self.db = np.sum(dout, axis=0).astype(np.float32, copy=False)
        out = (dout @ self.weights.T).astype(np.float32, copy=False)
        self.cache = None
        return out


class SoftmaxCrossEntropy:
    def __init__(self):
        self.cache = None
    
    def forward(self, logits, labels):
        # numerically stable softmax
        logits = logits.astype(np.float32, copy=False)
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        N = logits.shape[0]
        loss = -np.mean(np.log(probs[np.arange(N), labels] + 1e-8))
        
        self.cache = (probs.astype(np.float32, copy=False), labels)
        return float(loss)
    
    def backward(self):
        probs, labels = self.cache
        N = probs.shape[0]
        
        dlogits = probs.copy()
        dlogits[np.arange(N), labels] -= 1
        dlogits /= N
        
        return dlogits.astype(np.float32, copy=False)


class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def step(self, layers):
        for layer in layers:
            if hasattr(layer, 'weights') and layer.dW is not None:
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
    
    def forward(self, x, return_activations=False):
        activations = [] if return_activations else None
        for layer in self.layers:
            x = layer.forward(x)
            if return_activations:
                activations.append(x)
        if return_activations:
            return x, activations
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
        return np.argmax(self.forward(x.astype(np.float32, copy=False)), axis=1)
    
    def compute_gradcam(self, x, target_layer_idx, target_class=None):
        if target_layer_idx < 0 or target_layer_idx >= len(self.layers):
            raise IndexError("target_layer_idx out of range")
        logits, activations = self.forward(x, return_activations=True)
        if target_class is None:
            target_class = int(np.argmax(logits, axis=1)[0])
        dout = np.zeros_like(logits)
        dout[0, target_class] = 1.0
        gradients = None
        target_features = None
        for idx in reversed(range(len(self.layers))):
            if idx > target_layer_idx:
                dout = self.layers[idx].backward(dout)
            elif idx == target_layer_idx:
                gradients = dout.copy()
                target_features = activations[idx]
                break
        if gradients is None:
            raise ValueError("target layer gradients unavailable")
        if target_features.ndim != 4:
            raise ValueError("Grad-CAM target layer must output a 4D tensor")
        weights = gradients.mean(axis=(2, 3), keepdims=True)
        heatmap = np.sum(weights * target_features, axis=1)[0]
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

