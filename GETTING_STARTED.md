# Getting Started Guide

This guide will help you set up and run the deep learning biology notebooks step by step.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Running the Notebooks](#running-the-notebooks)
4. [Understanding the Notebooks](#understanding-the-notebooks)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements:
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.8 or higher
- **Internet**: Required for downloading dependencies and data

### For GPU Support (Optional but Recommended):
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060 or better)
- **CUDA**: Version 12.1 or compatible
- **Driver**: Latest NVIDIA drivers

---

## Installation Methods

### Method 1: Docker (Recommended)

Docker provides the most reliable and reproducible environment.

#### Step 1: Install Docker
- **Windows/Mac**: Download [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: 
  ```bash
  sudo apt-get update
  sudo apt-get install docker.io docker-compose
  sudo usermod -aG docker $USER
  # Log out and back in
  ```

#### Step 2: Install NVIDIA Container Toolkit (for GPU)
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Step 3: Clone and Run
```bash
git clone https://github.com/tonyliang19/deep-learning-biology-codebook.git
cd deep-learning-biology-codebook
docker-compose up --build
```

#### Step 4: Access Jupyter
Open your browser to: `http://localhost:8888`

---

### Method 2: Local Installation

#### Step 1: Install Python
Ensure Python 3.8+ is installed:
```bash
python --version
```

#### Step 2: Create Virtual Environment
```bash
# Create environment
python -m venv dl-bio-env

# Activate (Linux/Mac)
source dl-bio-env/bin/activate

# Activate (Windows)
dl-bio-env\Scripts\activate
```

#### Step 3: Clone Repository
```bash
git clone https://github.com/tonyliang19/deep-learning-biology-codebook.git
cd deep-learning-biology-codebook
```

#### Step 4: Install Dependencies
```bash
# CPU-only (no GPU)
pip install -r requirements.txt

# GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

#### Step 5: Start Jupyter
```bash
jupyter notebook
```

---

### Method 3: Google Colab (No Installation)

Perfect for trying notebooks without local setup!

1. Go to [Google Colab](https://colab.research.google.com)
2. Click "File" → "Open notebook" → "GitHub"
3. Enter: `tonyliang19/deep-learning-biology-codebook`
4. Select a notebook to run

**Note**: For Colab, you'll need to add a cell at the top:
```python
!git clone https://github.com/tonyliang19/deep-learning-biology-codebook.git
%cd deep-learning-biology-codebook
!pip install -q -r requirements.txt
```

---

## Running the Notebooks

### Starting Order (Recommended for Beginners):

1. **Start Here**: `01_CNN_DNA_Sequence_Classification.ipynb`
   - Introduces deep learning basics
   - Explains CNNs step-by-step
   - Good for understanding the workflow

2. **Next**: `04_Transfer_Learning_Cell_Images.ipynb`
   - Shows power of pre-trained models
   - Faster training, easier to understand
   - Visual results are intuitive

3. **Then**: `03_Autoencoder_Gene_Expression.ipynb`
   - Introduces unsupervised learning
   - Visualizations help understand concepts
   - Less complex than RNNs

4. **Advanced**: `02_RNN_LSTM_Protein_Sequence.ipynb`
   - Most complex architecture
   - Build on knowledge from previous notebooks
   - Understand sequential data processing

### How to Run a Notebook:

1. **Open the notebook** in Jupyter
2. **Read the markdown cells** carefully - they explain concepts
3. **Run cells sequentially** from top to bottom
   - Click cell, then press `Shift + Enter`
   - Or use "Run All" from the Cell menu
4. **Observe outputs** - graphs, metrics, and results
5. **Experiment** - modify parameters and re-run

### Expected Runtime:

**With GPU:**
- Each notebook: 10-20 minutes
- All notebooks: ~1 hour

**With CPU:**
- Each notebook: 30-60 minutes
- All notebooks: 3-4 hours

---

## Understanding the Notebooks

### Structure of Each Notebook:

1. **Introduction** (Markdown)
   - What you'll learn
   - Why this technique is useful
   - Real-world applications

2. **Setup** (Code)
   - Import libraries
   - Check GPU availability
   - Set random seeds

3. **Data Generation** (Code + Markdown)
   - Create or load data
   - Explain data format
   - Visualize samples

4. **Data Preprocessing** (Code + Markdown)
   - Encoding/normalization
   - Train/validation/test splits
   - Data augmentation

5. **Model Building** (Code + Markdown)
   - Architecture explanation
   - Layer-by-layer breakdown
   - Parameter counting

6. **Training** (Code + Markdown)
   - Training loop explanation
   - Loss function and optimizer
   - Progress monitoring

7. **Evaluation** (Code + Markdown)
   - Test set performance
   - Confusion matrices
   - Detailed metrics

8. **Visualization** (Code + Markdown)
   - Training curves
   - Learned features
   - Predictions

9. **Summary** (Markdown)
   - Key takeaways
   - Real-world applications
   - Next steps

### Key Concepts Explained:

#### Notebook 1 - CNN:
- **One-hot encoding**: Converting DNA letters to numbers
- **Convolution**: Detecting patterns in sequences
- **Pooling**: Reducing dimensions while keeping important features
- **Motif detection**: Finding recurring biological patterns

#### Notebook 2 - LSTM:
- **Embeddings**: Converting amino acids to dense vectors
- **Recurrence**: Processing sequences step-by-step
- **Gates**: Memory mechanism in LSTMs
- **Bidirectional**: Reading sequences both ways

#### Notebook 3 - Autoencoder:
- **Encoding**: Compressing high-dimensional data
- **Latent space**: Lower-dimensional representation
- **Decoding**: Reconstructing original data
- **Dimensionality reduction**: Finding important features

#### Notebook 4 - Transfer Learning:
- **Pre-training**: Using knowledge from large datasets
- **Fine-tuning**: Adapting to new tasks
- **Freezing**: Keeping pre-trained weights fixed
- **Data augmentation**: Artificially increasing dataset size

---

## Troubleshooting

### Problem: Out of Memory

**Solution 1**: Reduce batch size
```python
batch_size = 8  # instead of 32
```

**Solution 2**: Use CPU instead
```python
device = torch.device('cpu')
```

**Solution 3**: Reduce model size
```python
# Use fewer layers or smaller hidden dimensions
hidden_dim = 64  # instead of 128
```

---

### Problem: CUDA Out of Memory

**Solution**:
```python
# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Or restart kernel and reduce batch size
```

---

### Problem: Slow Training

**Check GPU usage**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

**Speed up training**:
- Use GPU if available
- Reduce number of epochs
- Use smaller datasets for testing
- Enable mixed precision training

---

### Problem: Import Errors

**Solution**:
```bash
# Reinstall all dependencies
pip install --upgrade --force-reinstall -r requirements.txt

# Or specific package
pip install --upgrade torch torchvision
```

---

### Problem: Notebook Kernel Crashes

**Solutions**:
1. Restart kernel: `Kernel → Restart`
2. Clear outputs: `Cell → All Output → Clear`
3. Reduce memory usage (see "Out of Memory" above)
4. Check system resources with `htop` or Task Manager

---

### Problem: Can't Connect to Jupyter

**Docker**:
```bash
# Check if container is running
docker ps

# Check logs
docker-compose logs

# Restart
docker-compose down
docker-compose up
```

**Local**:
```bash
# Check if Jupyter is running
ps aux | grep jupyter

# Restart Jupyter
jupyter notebook --port=8888
```

---

### Problem: Results Don't Match Expected

This is **normal**! Due to:
- Random initialization
- Data splitting randomness
- Hardware differences

To get more consistent results:
```python
# Set all random seeds
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
```

---

## Tips for Learning

### 1. Don't Just Run - Understand
- Read markdown explanations carefully
- Predict what code will do before running
- Understand error messages

### 2. Experiment!
- Change hyperparameters
- Modify architectures
- Try different data

### 3. Visualize Everything
- Plot training curves
- Visualize data distributions
- Inspect model predictions

### 4. Start Simple
- Begin with small datasets
- Use fewer epochs for testing
- Gradually increase complexity

### 5. Debug Systematically
- Check data shapes
- Print intermediate outputs
- Use small batches for debugging

---

## Next Steps After Completing Notebooks

1. **Apply to Real Data**
   - Download biological datasets
   - Adapt code to your data format
   - Compare with published results

2. **Explore Advanced Topics**
   - Attention mechanisms
   - Graph neural networks
   - Transformers for biology

3. **Contribute**
   - Share improvements
   - Add new notebooks
   - Help others learn

4. **Build Projects**
   - Create end-to-end pipelines
   - Deploy models as web services
   - Publish your findings

---

## Getting Help

- **GitHub Issues**: Report bugs or ask questions
- **Documentation**: Check README.md for details
- **Community**: Join bioinformatics forums and Discord channels
- **Papers**: Read referenced papers for deeper understanding

---

Happy Learning! 🎉🧬🔬
