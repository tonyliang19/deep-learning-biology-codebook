# Quick Reference Guide

A handy reference for common operations and concepts used in the notebooks.

## 📋 Table of Contents
- [Common PyTorch Operations](#common-pytorch-operations)
- [Data Shapes](#data-shapes)
- [Hyperparameters](#hyperparameters)
- [Activation Functions](#activation-functions)
- [Loss Functions](#loss-functions)
- [Optimizers](#optimizers)
- [Evaluation Metrics](#evaluation-metrics)
- [Common Errors](#common-errors)

---

## Common PyTorch Operations

### Creating Tensors
```python
# From Python list
x = torch.tensor([1, 2, 3])

# Random tensors
x = torch.randn(3, 4)        # Normal distribution
x = torch.rand(3, 4)         # Uniform [0, 1]
x = torch.zeros(3, 4)        # All zeros
x = torch.ones(3, 4)         # All ones

# From NumPy
x = torch.from_numpy(numpy_array)
```

### Moving to GPU
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
model = model.to(device)
```

### Tensor Operations
```python
# Matrix multiplication
z = torch.matmul(x, y)
z = x @ y  # Alternative

# Element-wise operations
z = x + y
z = x * y
z = torch.relu(x)

# Reshaping
x = x.view(batch_size, -1)    # Flatten
x = x.reshape(2, 3, 4)
x = x.permute(0, 2, 1)        # Swap dimensions
```

---

## Data Shapes

### DNA/Protein Sequences

**Input (one-hot encoded DNA)**:
```
Shape: (batch_size, 4, sequence_length)
Example: (32, 4, 200)
- 32 sequences per batch
- 4 nucleotides (A, T, C, G)
- 200 nucleotides long
```

**Input (embedded protein)**:
```
Shape: (batch_size, sequence_length, embedding_dim)
Example: (32, 150, 64)
- 32 sequences per batch
- 150 amino acids long
- 64-dimensional embeddings
```

### Gene Expression

**Input**:
```
Shape: (batch_size, n_genes)
Example: (32, 2000)
- 32 samples per batch
- 2000 genes measured
```

### Images

**Input**:
```
Shape: (batch_size, channels, height, width)
Example: (16, 3, 224, 224)
- 16 images per batch
- 3 color channels (RGB)
- 224x224 pixels
```

---

## Hyperparameters

### Learning Rate
```python
# Common ranges
lr = 0.001   # Good default
lr = 0.0001  # More stable, slower
lr = 0.01    # Faster, less stable
```

### Batch Size
```python
# Trade-offs
batch_size = 8    # More stable gradients, slower
batch_size = 32   # Good balance (common default)
batch_size = 128  # Faster, less stable, needs more memory
```

### Number of Epochs
```python
# Depends on dataset size and complexity
epochs = 10   # Small dataset or quick test
epochs = 50   # Medium dataset
epochs = 100  # Large dataset or complex task
```

### Hidden Dimensions
```python
# For fully connected layers
hidden_dim = 64    # Small model
hidden_dim = 128   # Medium (common)
hidden_dim = 256   # Large
hidden_dim = 512   # Very large
```

### Dropout Rate
```python
# Regularization strength
dropout = 0.1   # Light regularization
dropout = 0.3   # Medium (good default)
dropout = 0.5   # Strong regularization
```

---

## Activation Functions

### ReLU (Most Common)
```python
nn.ReLU()
# Advantages: Fast, works well, prevents vanishing gradients
# Use: Hidden layers in most networks
```

### Sigmoid
```python
nn.Sigmoid()
# Advantages: Outputs between 0 and 1
# Use: Binary classification output
```

### Softmax
```python
nn.Softmax(dim=1)
# Advantages: Outputs sum to 1 (probabilities)
# Use: Multi-class classification output
```

### Tanh
```python
nn.Tanh()
# Advantages: Outputs between -1 and 1
# Use: RNN hidden states, sometimes in autoencoders
```

---

## Loss Functions

### Cross-Entropy (Classification)
```python
criterion = nn.CrossEntropyLoss()
# Use: Multi-class classification
# Combines softmax + negative log likelihood
```

### Binary Cross-Entropy
```python
criterion = nn.BCELoss()
# Use: Binary classification
# Requires sigmoid activation on output
```

### Mean Squared Error (Regression)
```python
criterion = nn.MSELoss()
# Use: Regression tasks, autoencoders
# Measures average squared difference
```

### Mean Absolute Error
```python
criterion = nn.L1Loss()
# Use: Regression when outliers are present
# Less sensitive to outliers than MSE
```

---

## Optimizers

### Adam (Recommended Default)
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Advantages: Adaptive learning rates, works well in most cases
# Best for: Most deep learning tasks
```

### SGD
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# Advantages: Simple, well-understood
# Best for: When training is unstable with Adam
```

### AdamW
```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# Advantages: Better weight decay than Adam
# Best for: Transfer learning, fine-tuning
```

### RMSprop
```python
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
# Advantages: Good for RNNs
# Best for: Recurrent networks
```

---

## Evaluation Metrics

### Classification

**Accuracy**:
```python
accuracy = (predicted == labels).sum().item() / len(labels)
```

**Precision, Recall, F1**:
```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```

**Confusion Matrix**:
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

**ROC-AUC**:
```python
from sklearn.metrics import roc_auc_score, roc_curve
auc = roc_auc_score(y_true, y_probs)
fpr, tpr, _ = roc_curve(y_true, y_probs)
```

### Regression

**Mean Squared Error**:
```python
mse = ((predicted - actual) ** 2).mean()
```

**R² Score**:
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

---

## Common Errors

### RuntimeError: CUDA out of memory
**Solution**: Reduce batch size or model size
```python
batch_size = 16  # Try 8 or even 4
```

### RuntimeError: Expected tensor for argument #1 'input' to have the same device as tensor for argument #2 'weight'
**Solution**: Move data to same device as model
```python
data = data.to(device)
labels = labels.to(device)
```

### ValueError: Target size (torch.Size([32])) must be the same as input size (torch.Size([32, 1]))
**Solution**: Squeeze or reshape tensors
```python
output = output.squeeze()  # Remove dimension of size 1
# Or
target = target.view(-1, 1)  # Add dimension
```

### RuntimeError: size mismatch, m1: [32 x 1024], m2: [2048 x 10]
**Solution**: Check layer dimensions
```python
# Print shapes to debug
print(f"Input shape: {x.shape}")
# Adjust layer sizes accordingly
```

### IndexError: Target X is out of bounds
**Solution**: Check label range
```python
# For n classes, labels should be 0 to n-1
print(f"Min label: {labels.min()}, Max label: {labels.max()}")
print(f"Number of classes: {num_classes}")
```

---

## Quick Tips

### Debugging
```python
# Check shapes
print(f"Shape: {tensor.shape}")

# Check values
print(f"Min: {tensor.min()}, Max: {tensor.max()}, Mean: {tensor.mean()}")

# Check for NaN
print(f"Contains NaN: {torch.isnan(tensor).any()}")

# Check gradients
print(f"Requires grad: {tensor.requires_grad}")
```

### Memory Management
```python
# Clear GPU cache
torch.cuda.empty_cache()

# Delete large tensors
del large_tensor

# Use no_grad for inference
with torch.no_grad():
    output = model(input)
```

### Speed Up Training
```python
# Use DataLoader with multiple workers
loader = DataLoader(dataset, batch_size=32, num_workers=4)

# Enable cudnn autotuner
torch.backends.cudnn.benchmark = True

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

---

## Notebook-Specific Tips

### CNN (Notebook 1)
- Kernel size: 3-7 for sequence motifs
- Stride: Usually 1
- Padding: 'same' or kernel_size // 2

### LSTM (Notebook 2)
- Hidden size: 128-256 typically
- Number of layers: 2-4
- Always use gradient clipping: `nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)`

### Autoencoder (Notebook 3)
- Latent dim: 10-100 depending on data
- Symmetric architecture (encoder mirrors decoder)
- Batch normalization helps stabilize training

### Transfer Learning (Notebook 4)
- Always use ImageNet normalization
- Start with frozen layers
- Fine-tune with lower learning rate (0.0001)

---

## Further Reading

- **PyTorch Documentation**: https://pytorch.org/docs/
- **Deep Learning Book**: https://www.deeplearningbook.org/
- **Papers with Code**: https://paperswithcode.com/
- **PyTorch Tutorials**: https://pytorch.org/tutorials/

---

Keep this guide handy while working through the notebooks! 📚
