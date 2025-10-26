import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
from data_loader import load_mnist_data
from numpy_nn import Sequential, SGD


st.set_page_config(page_title="CNN Visualizer", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 3rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem;}
    .subtitle {font-size: 1.2rem; color: #666; font-style: italic; margin-bottom: 2rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  padding: 1.5rem; border-radius: 10px; color: white; text-align: center;}
    .info-box {background-color: #f0f8ff; padding: 1.5rem; border-radius: 10px; 
               border-left: 5px solid #1f77b4; margin: 1rem 0; color: #333;}
    .stProgress > div > div > div > div {background-color: #667eea;}
    h1, h2, h3 {color: #1f77b4;}
    .stButton>button {border-radius: 20px; font-weight: 600; transition: all 0.3s;}
    .stButton>button:hover {transform: scale(1.05);}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    X_train_full, y_train_full, X_test, y_test = load_mnist_data()
    rng = np.random.default_rng(2025)
    train_indices = rng.permutation(len(X_train_full))[:2000]
    test_indices = rng.permutation(len(X_test))[:1000]
    return (
        X_train_full[train_indices],
        y_train_full[train_indices],
        X_test[test_indices],
        y_test[test_indices]
    )


@st.cache_resource
def initialize_model():
    model = Sequential(input_channels=1)
    if os.path.exists('model_weights.npz'):
        try:
            model.load_weights('model_weights.npz')
            if 'trained' not in st.session_state:
                st.session_state.trained = True
        except Exception as e:
            st.warning(f"Could not load saved weights: {e}")
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=64, lr=0.01):
    optimizer = SGD(learning_rate=lr)
    n_batches = len(X_train) // batch_size
    train_losses, train_accs, batch_losses, batch_accs = [], [], [], []
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_shuffled, y_shuffled = X_train[indices], y_train[indices]
        epoch_loss = epoch_acc = 0
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            X_batch = X_shuffled[start_idx:start_idx + batch_size]
            y_batch = y_shuffled[start_idx:start_idx + batch_size]
            
            logits = model.forward(X_batch)
            loss = model.loss_fn.forward(logits, y_batch)
            model.backward(model.loss_fn.backward())
            optimizer.step(model.get_all_layers())
            
            acc = np.mean(np.argmax(logits, axis=1) == y_batch)
            epoch_loss += loss
            epoch_acc += acc
            batch_losses.append(loss)
            batch_accs.append(acc)
            
            if batch_idx % 10 == 0:
                yield {
                    'epoch': epoch + 1,
                    'batch': batch_idx,
                    'total_batches': n_batches,
                    'train_loss': epoch_loss / (batch_idx + 1),
                    'train_acc': epoch_acc / (batch_idx + 1),
                    'done': False,
                    'batch_losses': batch_losses.copy(),
                    'batch_accs': batch_accs.copy()
                }
        
        train_losses.append(epoch_loss / n_batches)
        train_accs.append(epoch_acc / n_batches)
        
        yield {
            'epoch': epoch + 1,
            'batch': n_batches,
            'total_batches': n_batches,
            'train_loss': train_losses[-1],
            'train_acc': train_accs[-1],
            'done': False,
            'all_train_losses': train_losses,
            'all_train_accs': train_accs,
            'epoch_complete': True
        }
    
    test_preds = np.concatenate([model.predict(X_test[i:i+batch_size]) 
                                  for i in range(0, len(X_test), batch_size)])
    
    yield {
        'done': True,
        'all_train_losses': train_losses,
        'all_train_accs': train_accs,
        'final_test_acc': np.mean(test_preds == y_test)
    }


def plot_kernels(conv_layer, title):
    kernels = conv_layer.weights
    n_filters = min(8, kernels.shape[0])
    
    fig, axes = plt.subplots(1, n_filters, figsize=(12, 2))
    axes = [axes] if n_filters == 1 else axes
    
    for i in range(n_filters):
        axes[i].imshow(kernels[i, 0], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'F{i}', fontsize=8)
    
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    return fig


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    return np.bincount(num_classes * y_true + y_pred, minlength=num_classes**2).reshape(num_classes, num_classes)


def compute_metrics(cm):
    # compute per-class precision, recall, and f1-score from confusion matrix
    precision = np.diag(cm) / (np.sum(cm, axis=0) + 1e-10)
    recall = np.diag(cm) / (np.sum(cm, axis=1) + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # macro averages (unweighted mean across classes)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_per_class_metrics(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    classes = np.arange(10)
    
    axes[0].bar(classes, metrics['precision'], color='steelblue', alpha=0.8)
    axes[0].axhline(metrics['macro_precision'], color='red', linestyle='--', linewidth=2, label=f"Macro: {metrics['macro_precision']:.3f}")
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Per-Class Precision')
    axes[0].set_ylim([0, 1.05])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(classes, metrics['recall'], color='forestgreen', alpha=0.8)
    axes[1].axhline(metrics['macro_recall'], color='red', linestyle='--', linewidth=2, label=f"Macro: {metrics['macro_recall']:.3f}")
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Per-Class Recall')
    axes[1].set_ylim([0, 1.05])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    axes[2].bar(classes, metrics['f1'], color='darkorange', alpha=0.8)
    axes[2].axhline(metrics['macro_f1'], color='red', linestyle='--', linewidth=2, label=f"Macro: {metrics['macro_f1']:.3f}")
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_title('Per-Class F1-Score')
    axes[2].set_ylim([0, 1.05])
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def overlay_heatmap(image, heatmap, alpha=0.4):
    if heatmap.shape != (28, 28):
        zoom_factors = (28 / heatmap.shape[0], 28 / heatmap.shape[1])
        heatmap = zoom(heatmap, zoom_factors, order=1)
    
    image_rgb = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    heatmap_colored = plt.get_cmap('jet')(heatmap)[:, :, :3]
    
    return np.clip(image_rgb * (1 - alpha) + heatmap_colored * alpha, 0, 1)


def plot_architecture_diagram(sample_image=None, sample_output=None):
    fig, ax = plt.subplots(figsize=(18, 4.5), facecolor='none', dpi=150)
    ax.set_xlim(-1, 18)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    fig.patch.set_alpha(0.0)
    
    from matplotlib.patches import Rectangle, Polygon, Circle, RegularPolygon
    
    colors = {'input':'#4A90E2', 'conv': '#4A90E2', 'relu': '#50C878', 
              'pool': '#E94B3C', 'flatten': '#F5A623', 'dense': '#9B59B6', 'softmax': '#8E44AD'}
    layers = [
        ('Input\n1×28×28', 'input', 1, 28, 28), ('Conv2D\n8', 'conv', 8, 28, 28), ('ReLU', 'relu', 8, 28, 28),
        ('Conv2D\n16', 'conv', 16, 28, 28), ('ReLU', 'relu', 16, 28, 28), ('MaxPool\n16×14×14', 'pool', 16, 14, 14),
        ('Conv2D\n32', 'conv', 32, 14, 14), ('ReLU', 'relu', 32, 14, 14), ('Conv2D\n64', 'conv', 64, 14, 14),
        ('ReLU', 'relu', 64, 14, 14), ('MaxPool\n64×7×7', 'pool', 64, 7, 7), ('Flatten', 'flatten', 3136, 0, 0),
        ('Dense\n128', 'dense', 128, 0, 0), ('ReLU', 'relu', 128, 0, 0), ('Dense\n10', 'dense', 10, 0, 0),
        ('Softmax', 'softmax', 10, 0, 0)
    ]
    
    if sample_image is not None:
        ax.imshow(sample_image, cmap='gray', extent=(-0.8, -0.1, 1.8, 2.5), alpha=0.9)
        ax.add_patch(Rectangle((-0.8, 1.8), 0.7, 0.7, facecolor='none', edgecolor='white', linewidth=2))
        ax.text(-0.45, 1.5, 'Input', ha='center', va='top', fontsize=9, fontweight='bold', color='white')
    x_pos = 0.5
    for i, (label, ltype, depth, h, w) in enumerate(layers):
        color = colors[ltype]
        
        if ltype in ['conv', 'input', 'pool']:
            ds = 0.25 + (depth/64)*0.35 if ltype in ['conv', 'input'] else 0.3
            hs = 0.5 + (h/28)*1.3
            wo = ds * 0.3
            
            ax.add_patch(Rectangle((x_pos, 2.3-hs/2), ds, hs, facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9))
            ax.add_patch(Polygon([[x_pos, 2.3+hs/2], [x_pos+ds, 2.3+hs/2], [x_pos+ds+wo, 2.3+hs/2+wo], [x_pos+wo, 2.3+hs/2+wo]], 
                                facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.7))
            ax.add_patch(Polygon([[x_pos+ds, 2.3-hs/2], [x_pos+ds, 2.3+hs/2], [x_pos+ds+wo, 2.3+hs/2+wo], [x_pos+ds+wo, 2.3-hs/2+wo]], 
                                facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.65))
            ax.text(x_pos+ds/2+wo/3, 1.0, label, ha='center', va='top', fontsize=8, fontweight='bold', color='white')
            x_pos += ds + wo + 0.5
            
        elif ltype == 'relu':
            ax.add_patch(Circle((x_pos+0.2, 2.3), 0.3, facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9))
            ax.text(x_pos+0.2, 2.3, 'ReLU', ha='center', va='center', fontsize=6.5, fontweight='bold', color='white')
            x_pos += 1
        elif ltype == 'flatten':
            ax.add_patch(RegularPolygon((x_pos+0.25, 2.3), 6, radius=0.35, facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9))
            ax.text(x_pos+0.25, 2.3, 'Flatten', ha='center', va='center', fontsize=6, fontweight='bold', color='white')
            x_pos += 0.5 + 0.4  
        elif ltype in ['dense', 'softmax']:
            hs = 0.3 + (depth/128)*1.2
            ax.add_patch(Rectangle((x_pos-0.15, 2.3-hs/2), 0.3, hs, facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9))
            ax.text(x_pos, 1.0, label, ha='center', va='top', fontsize=8, fontweight='bold', color='white')
            x_pos += 0.3 + 0.4  
        if i < len(layers) - 1:
            ax.annotate('', xy=(x_pos-0.1, 2.3), xytext=(x_pos-0.4, 2.3), arrowprops=dict(arrowstyle='->', lw=1.5, color='white', alpha=1))
    
    if sample_output is not None:
        out_x = x_pos + 0.2
        ax.text(out_x + 0.5, 3.8, 'Output Vector', ha='center', fontsize=9, fontweight='bold', color='white')
        for i, prob in enumerate(sample_output):
            y_pos = 3.5 - i*0.3
            ax.text(out_x, y_pos, f'{i}:', ha='right', va='center', fontsize=8, color='white')
            ax.text(out_x+0.1, y_pos, f'{prob:.4f}', ha='left', va='center', 
                   fontsize=8, fontweight='bold' if prob == sample_output.max() else 'normal', 
                   color='#50C878' if prob == sample_output.max() else 'white')
    
    ax.text(7.5, 4.2, 'Network Architecture', ha='center', fontsize=12, fontweight='bold', color='white')
    
    legend_items = [('Conv/Input', colors['conv'], 1), ('Activation', colors['relu'], 3.5), 
                   ('Pooling', colors['pool'], 6), ('Flatten', colors['flatten'], 8.5), ('Dense', colors['dense'], 11)]
    for name, color, x in legend_items:
        ax.add_patch(Rectangle((x, 0.3), 0.25, 0.15, facecolor=color, edgecolor='white', linewidth=1, alpha=0.9))
        ax.text(x+0.35, 0.375, name, ha='left', va='center', fontsize=8, fontweight='bold', color='white')
    plt.tight_layout()
    return fig



st.markdown('<h1 class="main-header">CNN Visualizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">End-to-End NumPy Implementation with Real-Time Training Dynamics and Interpretability Analysis</p>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 1rem;'>
    <a href='https://github.com/stryker13799/explainable-cnn' target='_blank' style='color: #1f77b4; text-decoration: none; font-weight: 600;'>
        View Source Code on GitHub
    </a>
</div>
""", unsafe_allow_html=True)

X_train, y_train, X_test, y_test = load_data()

with st.expander("Technical Overview", expanded=True):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About
        This framework demonstrates convolutional neural network mechanics through a pure NumPy 
        implementation, eliminating framework abstractions to expose fundamental computational primitives.
        
        **Core Features:**
        - From-scratch backpropagation implementing automatic differentiation
        - Real-time optimization trajectory visualization during gradient descent
        - Hierarchical feature map analysis across convolutional layers
        - Gradient-weighted Class Activation Mapping (Grad-CAM) for interpretability
        - Educational infrastructure for understanding deep learning fundamentals
        """)
    
    with col2:
        st.markdown("""
        ### Experimental Dataset
        **MNIST Handwritten Digits**
        - Training subset: 2,000 samples
        - Validation subset: 1,000 samples
        - Dimensionality: 28×28 grayscale
        - Classes: 10 (digits 0-9)
        """)
    
    st.markdown("""
    ### Network Architecture
    **Optimization:** Stochastic Gradient Descent (SGD) with configurable learning rate  
    **Computational Complexity:** ~20-30 seconds on standard CPU
    """)
    
    sample_idx = np.where(y_train == 7)[0][0]
    sample_img = X_train[sample_idx, 0]
    sample_probs = np.array([0.0156, 0.0089, 0.0234, 0.0445, 0.0178, 0.0312, 0.0067, 0.7823, 0.0198, 0.0498])

    st.pyplot(plot_architecture_diagram(sample_img, sample_probs))
    
    st.markdown("**Sample MNIST Images:**")
    fig, axes = plt.subplots(1, 10, figsize=(12, 1.5))
    for i in range(10):
        idx = np.where(y_train == i)[0][0]
        axes[i].imshow(X_train[idx, 0], cmap='gray')
        axes[i].set_title(f'{i}', fontsize=10)
        axes[i].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

st.sidebar.markdown("## Dataset Statistics")
st.sidebar.metric("Training Samples", f"{len(X_train):,}")
st.sidebar.metric("Validation Samples", f"{len(X_test):,}")

st.sidebar.markdown("---")
st.sidebar.markdown("## Model State")
if st.session_state.get('trained', False):
    st.sidebar.success("Model Converged")
    if st.session_state.get('final_test_acc'):
        st.sidebar.metric("Validation Accuracy", f"{st.session_state.final_test_acc:.2%}")
    if st.session_state.get('metrics'):
        metrics = st.session_state.metrics
        st.sidebar.metric("Macro Precision", f"{metrics['macro_precision']:.3f}")
        st.sidebar.metric("Macro Recall", f"{metrics['macro_recall']:.3f}")
        st.sidebar.metric("Macro F1-Score", f"{metrics['macro_f1']:.3f}")
    if st.session_state.get('train_losses'):
        st.sidebar.metric("Final Training Loss", f"{st.session_state.train_losses[-1]:.4f}")
else:
    st.sidebar.warning("Model Uninitialized")
    st.sidebar.info("Commence training to initialize network parameters")

for key, default in [
    ('model', None),
    ('trained', False),
    ('train_losses', []),
    ('train_accs', []),
    ('final_test_acc', None),
    ('metrics', None),
    ('current_page', 'Live Training')
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.model is None:
    st.session_state.model = initialize_model()
    if os.path.exists('model_weights.npz'):
        st.session_state.trained = True

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Live Training", "Inference Explorer"],
    label_visibility="visible",
    index=0 if st.session_state.current_page == "Live Training" else 1
)
st.session_state.current_page = page

# page 1: training
if page == "Live Training":
    st.markdown("## Training Dashboard: Real-Time Optimization Dynamics")

    st.markdown('<div class="info-box"><strong>Note:</strong> Training executes in real-time with live metric updates. Monitor convergence behavior on this page, then proceed to Inference Explorer for model evaluation.</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.number_input("epochs", min_value=1, max_value=10, value=2)
    with col2:
        batch_size = st.number_input("batch size", min_value=32, max_value=128, value=64)
    with col3:
        lr = st.slider("learning rate", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.3f")
    
    train_button = False
    train_mode = None
    if not st.session_state.trained:
        if st.button("Initialize Training", type="primary", use_container_width=True):
            train_button = True
            train_mode = "initial"
    else:
        st.success("**Training Complete.** Model parameters converged. Proceed to Inference Explorer for evaluation.")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Reinitialize Training", type="secondary", use_container_width=True):
                train_button = True
                train_mode = "reinit"
        with btn_col2:
            if st.button("Proceed to Inference Explorer", type="primary", use_container_width=True):
                st.session_state.current_page = "Inference Explorer"
                st.rerun()

    if train_button:
        if train_mode == "reinit":
            st.session_state.model = Sequential(input_channels=1)
        elif st.session_state.model is None:
            st.session_state.model = initialize_model()
        st.session_state.trained = False
        st.session_state.train_losses = []
        st.session_state.train_accs = []
        st.session_state.final_test_acc = None
        st.session_state.metrics = None
        
        status_container = st.container()
        with status_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        col1, col2 = st.columns(2)
        with col1:
            loss_chart = st.empty()
        with col2:
            acc_chart = st.empty()
        
        st.markdown("### Learned Convolutional Kernels")
        st.markdown("""
        <div class="info-box">
        <strong>What are these filters?</strong><br>
        Each small grid (F0, F1, F2...) represents a learned <strong>convolutional filter/kernel</strong>, a small pattern detector that scans across the input image. 
        During training, the network learns what patterns to look for:<br>
        • <strong>White regions</strong>: Areas the filter activates strongly on (positive weights)<br>
        • <strong>Dark regions</strong>: Areas the filter suppresses (negative weights)<br>
        • <strong>Gray regions</strong>: Neutral areas (near-zero weights)<br><br>
        <strong>Layer progression:</strong><br>
        • <strong>Early layers</strong> (8, 16 filters): Learn basic edges, lines, and simple textures<br>
        • <strong>Deep layers</strong> (32, 64 filters): Combine early features into complex patterns like curves and digit parts<br><br>
        Watch how these patterns emerge and sharpen as training progresses
        </div>
        """, unsafe_allow_html=True)
        
        # Add another status indicator before kernels
        kernel_status = st.empty()
        
        col1, col2 = st.columns(2)
        with col1:
            kernel1_plot = st.empty()
            kernel2_plot = st.empty()
        with col2:
            kernel3_plot = st.empty()
            kernel4_plot = st.empty()
        
        # train
        train_losses_live = []
        train_accs_live = []
        
        for metrics in train_model(st.session_state.model, X_train, y_train, X_test, y_test, 
                                   epochs=epochs, batch_size=batch_size, lr=lr):
            
            if metrics['done']:
                st.session_state.train_losses = metrics['all_train_losses']
                st.session_state.train_accs = metrics['all_train_accs']
                st.session_state.final_test_acc = metrics['final_test_acc']
                st.session_state.trained = True
                try:
                    st.session_state.model.save_weights('model_weights.npz')
                    st.success("Model weights saved successfully!")
                except Exception as e:
                    st.warning(f"Could not save model weights: {e}")
                progress_bar.progress(1.0)
                status_text.success(f"Training converged. Validation accuracy: {metrics['final_test_acc']:.2%}")
                break
            
            # update progress
            progress = (metrics['epoch'] - 1 + metrics['batch'] / metrics['total_batches']) / epochs
            progress_bar.progress(min(progress, 1.0))
            current_status = f"**Optimizing...** Epoch {metrics['epoch']}/{epochs} | " \
                           f"Batch {metrics['batch']}/{metrics['total_batches']} | " \
                           f"Loss: {metrics['train_loss']:.4f} | " \
                           f"Accuracy: {metrics['train_acc']:.2%}"
            status_text.markdown(current_status)
            kernel_status.info(f"Training in progress... {current_status}")
            
            if 'batch_losses' in metrics:
                loss_chart.line_chart(metrics['batch_losses'], use_container_width=True)
                acc_chart.line_chart(metrics['batch_accs'], use_container_width=True)
            
            if 'epoch_complete' in metrics and metrics['epoch_complete']:
                train_losses_live = metrics['all_train_losses']
                train_accs_live = metrics['all_train_accs']
                
                loss_chart.line_chart(train_losses_live, use_container_width=True)
                acc_chart.line_chart(train_accs_live, use_container_width=True)
                
                # Update kernel status
                kernel_status.success(f"Epoch {metrics['epoch']}/{epochs} complete! Updating filters...")
                
                conv_layers = st.session_state.model.get_conv_layers()
                kernel1_plot.pyplot(plot_kernels(conv_layers[0], "Layer 0: Conv2D (8 filters)"))
                kernel2_plot.pyplot(plot_kernels(conv_layers[1], "Layer 2: Conv2D (16 filters)"))
                kernel3_plot.pyplot(plot_kernels(conv_layers[2], "Layer 5: Conv2D (32 filters)"))
                kernel4_plot.pyplot(plot_kernels(conv_layers[3], "Layer 7: Conv2D (64 filters)"))
        
        if st.session_state.trained:
            kernel_status.success(f"Training Complete! Validation Accuracy: {st.session_state.final_test_acc:.2%}")
        
        if st.session_state.trained:
            st.subheader("Performance Metrics")
            test_preds = []
            for i in range(0, len(X_test), batch_size):
                preds = st.session_state.model.predict(X_test[i:i+batch_size])
                test_preds.extend(preds)
            cm = compute_confusion_matrix(y_test, np.array(test_preds))
            metrics = compute_metrics(cm)
            st.session_state.metrics = metrics
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Macro Precision", f"{metrics['macro_precision']:.3f}")
            with col2:
                st.metric("Macro Recall", f"{metrics['macro_recall']:.3f}")
            with col3:
                st.metric("Macro F1-Score", f"{metrics['macro_f1']:.3f}")
            
            st.pyplot(plot_per_class_metrics(metrics))
            st.pyplot(plot_confusion_matrix(cm))
    
    if st.session_state.trained and not train_button:
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(st.session_state.train_losses, use_container_width=True)
            st.caption("Cross-Entropy Loss per Epoch")
        with col2:
            st.line_chart(st.session_state.train_accs, use_container_width=True)
            st.caption("Classification Accuracy per Epoch")
        
        if st.session_state.final_test_acc:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Validation Accuracy", f"{st.session_state.final_test_acc:.2%}")
            if st.session_state.get('metrics'):
                metrics = st.session_state.metrics
                with col2:
                    st.metric("Macro Precision", f"{metrics['macro_precision']:.3f}")
                with col3:
                    st.metric("Macro Recall", f"{metrics['macro_recall']:.3f}")
                with col4:
                    st.metric("Macro F1-Score", f"{metrics['macro_f1']:.3f}")
        
        st.subheader("Convolutional Kernel Visualization")
        st.markdown("""
        <div class="info-box">
        <strong>Understanding Convolutional Filters:</strong><br>
        Each grid shows the learned weights of a <strong>3×3 convolutional filter</strong>. These are the patterns the network searches for in images:<br>
        • <strong>White pixels</strong>: Strong positive weights (the filter "looks for" these patterns)<br>
        • <strong>Black pixels</strong>: Strong negative weights (the filter suppresses these patterns)<br>
        • <strong>Gray pixels</strong>: Neutral/weak weights<br><br>
        <strong>What to look for:</strong><br>
        • <strong>Layer 0 (8 filters)</strong>: Simple edge and gradient detectors (horizontal, vertical, diagonal lines)<br>
        • <strong>Layer 2 (16 filters)</strong>: Corner and texture detectors, combining multiple edges<br>
        • <strong>Layer 5 (32 filters)</strong>: More complex patterns like curves and stroke segments<br>
        • <strong>Layer 7 (64 filters)</strong>: High-level feature combinations that respond to digit-specific shapes<br><br>
        These filters work together: early layers detect simple patterns, deeper layers combine them into complex digit representations.
        </div>
        """, unsafe_allow_html=True)
        conv_layers = st.session_state.model.get_conv_layers()
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_kernels(conv_layers[0], "Layer 0: Conv2D (8 filters)"))
            st.pyplot(plot_kernels(conv_layers[1], "Layer 2: Conv2D (16 filters)"))
        with col2:
            st.pyplot(plot_kernels(conv_layers[2], "Layer 5: Conv2D (32 filters)"))
            st.pyplot(plot_kernels(conv_layers[3], "Layer 7: Conv2D (64 filters)"))
        
        st.subheader("Classification Performance Analysis")
        if st.session_state.get('metrics'):
            st.pyplot(plot_per_class_metrics(st.session_state.metrics))
        
        st.subheader("Confusion Matrix Analysis")
        if st.session_state.get('metrics'):
            test_preds = []
            batch_size = 64
            for i in range(0, len(X_test), batch_size):
                preds = st.session_state.model.predict(X_test[i:i+batch_size])
                test_preds.extend(preds)
            cm = compute_confusion_matrix(y_test, np.array(test_preds))
            st.pyplot(plot_confusion_matrix(cm))

# page 2: inference
elif page == "Inference Explorer":
    st.markdown("## Inference Explorer: Model Interpretability Analysis")
    
    if not st.session_state.trained:
        st.warning("Model parameters not initialized")
        st.markdown('<div class="info-box"><strong>Required Action:</strong> Navigate to "Live Training" to initialize network parameters through gradient descent optimization. Training requires approximately 20-30 seconds per epoch. Return to this interface for inference analysis upon convergence.</div>', unsafe_allow_html=True)
        if st.button("Navigate to Training Dashboard", type="primary", use_container_width=True):
            st.session_state.current_page = "Live Training"
            st.rerun()
    else:
        with st.expander("Interpretability Methodology", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### Hierarchical Feature Representations
                Convolutional layers learn hierarchical feature extractors:
                
                - **Early Layers (L0-L2)**: Low-level feature detectors (edges, gradients, textures)
                - **Intermediate Layers (L3-L5)**: Mid-level pattern recognition (corners, strokes, local structures)
                - **Deep Layers (L6+)**: High-level semantic representations (digit parts, global shapes)
                
                Activation intensity indicates feature presence at spatial locations.
                """)
            
            with col2:
                st.markdown("""
                ### Gradient-Weighted Class Activation Mapping
                Grad-CAM visualizes discriminative regions for classification decisions:
                
                - **Methodology**: Compute gradient flow from predicted class to target convolutional layer
                - **Interpretation**: Red/yellow regions indicate high discriminative importance
                - **Application**: Explains spatial attention mechanisms underlying predictions
                
                Enables verification of whether learned representations align with semantic image content.
                """)
        
        if st.button("Predict random number", type="primary", use_container_width=True):
            idx = np.random.randint(0, len(X_test))
            st.session_state.test_idx = idx
        
        if 'test_idx' in st.session_state:
            idx = st.session_state.test_idx
            test_image = X_test[idx:idx+1]  # (1, 1, 28, 28)
            true_label = y_test[idx]
            
            # predict
            pred_label = st.session_state.model.predict(test_image)[0]
            
            # display
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                st.image(test_image[0, 0], width=150, clamp=True)
                st.metric("Ground Truth Label", int(true_label))
                st.metric("Predicted Label", int(pred_label))
            
            with col2:
                st.subheader("Activation Maps")
                layer_names = []
                layer_indices = []
                for i, layer in enumerate(st.session_state.model.get_all_layers()):
                    if type(layer).__name__ in ['Conv2D', 'ReLU']:
                        layer_names.append(f"Layer {i}: {type(layer).__name__}")
                        layer_indices.append(i)
                    elif type(layer).__name__ == 'Flatten':
                        break
                
                selected_layer_name = st.selectbox("Select target layer", layer_names)
                selected_layer_idx = layer_indices[layer_names.index(selected_layer_name)]
                
                # get feature maps
                activations = test_image
                for i, layer in enumerate(st.session_state.model.get_all_layers()):
                    activations = layer.forward(activations)
                    if i == selected_layer_idx:
                        break
                
                # check if activations are 2D feature maps
                if len(activations.shape) == 4 and activations.shape[1] > 0:
                    # plot feature maps (first 8 channels)
                    num_channels = min(8, activations.shape[1])
                    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
                    axes = axes.flatten()
                    
                    for ch in range(num_channels):
                        axes[ch].imshow(activations[0, ch], cmap='viridis')
                        axes[ch].axis('off')
                        axes[ch].set_title(f'ch {ch}', fontsize=8)
                    
                    # hide unused subplots
                    for ch in range(num_channels, 8):
                        axes[ch].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("Selected layer produces flattened 1D representation. Please select convolutional or ReLU layer.")
            
            with col3:
                st.subheader("Grad-CAM Analysis")
                conv_layer_names = []
                conv_layer_indices = []
                for i, layer in enumerate(st.session_state.model.get_all_layers()):
                    if type(layer).__name__ == 'Conv2D':
                        conv_layer_names.append(f"Layer {i}: Conv2D")
                        conv_layer_indices.append(i)
                
                selected_conv_name = st.selectbox("Select convolutional layer", conv_layer_names)
                selected_conv_idx = conv_layer_indices[conv_layer_names.index(selected_conv_name)]
                
                # compute gradcam
                heatmap = st.session_state.model.compute_gradcam(test_image, selected_conv_idx, target_class=int(pred_label))
                
                original_img = test_image[0, 0]
                overlaid = overlay_heatmap(original_img, heatmap)
                
                fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(original_img, cmap='gray')
                axes[0].set_title('Original Input')
                axes[0].axis('off')
                
                axes[1].imshow(heatmap, cmap='jet')
                axes[1].set_title('Attention Heatmap')
                axes[1].axis('off')
                
                axes[2].imshow(overlaid)
                axes[2].set_title('Superimposed Visualization')
                axes[2].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.caption(f"Discriminative regions for predicted class: {pred_label}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p><strong>Implementation Details</strong></p>
    <p>Pure NumPy • No Framework Dependencies</p>
    <p>Educational Tool for Deep Learning Fundamentals</p>
</div>
""", unsafe_allow_html=True)
