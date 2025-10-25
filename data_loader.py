import os
import gzip
import numpy as np
import requests


def download_file(url, filepath):
    if os.path.exists(filepath):
        return
    
    print(f"Downloading {os.path.basename(filepath)}...")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"{os.path.basename(filepath)}")


def download_mnist(data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    files = {
        "train-images-idx3-ubyte.gz": [
            "https://github.com/fgnt/mnist/raw/master/train-images-idx3-ubyte.gz",
        ],
        "train-labels-idx1-ubyte.gz": [
            "https://github.com/fgnt/mnist/raw/master/train-labels-idx1-ubyte.gz",
        ],
        "t10k-images-idx3-ubyte.gz": [
            "https://github.com/fgnt/mnist/raw/master/t10k-images-idx3-ubyte.gz",
        ],
        "t10k-labels-idx1-ubyte.gz": [
            "https://github.com/fgnt/mnist/raw/master/t10k-labels-idx1-ubyte.gz"
        ]
    }
    
    for filename, url in files.items():
        filepath = os.path.join(data_dir, filename)
        download_file(url, filepath)


def parse_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num_images, rows, cols = [int.from_bytes(f.read(4), 'big') for _ in range(4)]
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return data


def parse_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, num_labels = [int.from_bytes(f.read(4), 'big') for _ in range(2)]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist_data(data_dir='data'):
    download_mnist(data_dir)
    
    paths = {
        'train_img': 'train-images-idx3-ubyte.gz',
        'train_lbl': 'train-labels-idx1-ubyte.gz',
        'test_img': 't10k-images-idx3-ubyte.gz',
        'test_lbl': 't10k-labels-idx1-ubyte.gz'
    }
    
    train_images = parse_images(os.path.join(data_dir, paths['train_img']))
    train_labels = parse_labels(os.path.join(data_dir, paths['train_lbl']))
    test_images = parse_images(os.path.join(data_dir, paths['test_img']))
    test_labels = parse_labels(os.path.join(data_dir, paths['test_lbl']))
    
    # normalize to [0,1] and reshape to (batch, channels, height, width)
    X_train = (train_images.astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
    X_test = (test_images.astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
    y_train = train_labels.astype(np.int64)
    y_test = test_labels.astype(np.int64)
    
    return X_train, y_train, X_test, y_test
