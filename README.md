# Deep Learning Biology Codebook

A comprehensive collection of Jupyter notebooks demonstrating deep learning techniques applied to biological data. Each notebook includes detailed explanations, code examples, and visualizations designed for learners with little to medium coding knowledge.

## 📚 Notebooks Overview

### 1. CNN for DNA Sequence Classification
**File:** `notebooks/01_CNN_DNA_Sequence_Classification.ipynb`

Learn how Convolutional Neural Networks (CNNs) can identify patterns in DNA sequences, specifically for promoter region detection.

**Topics Covered:**
- One-hot encoding of DNA sequences
- CNN architecture for sequence data
- Motif detection and visualization
- Binary classification with biological sequences

**Key Techniques:**
- Conv1D layers for sequence patterns
- MaxPooling for position invariance
- Filter visualization to understand learned motifs

---

### 2. RNN/LSTM for Protein Sequence Analysis
**File:** `notebooks/02_RNN_LSTM_Protein_Sequence.ipynb`

Explore Recurrent Neural Networks and LSTMs for protein family classification.

**Topics Covered:**
- Integer encoding and embeddings for amino acids
- Bidirectional LSTM architecture
- Handling variable-length sequences
- Multi-class protein classification

**Key Techniques:**
- Sequence padding and packing
- Bidirectional processing for context
- Gradient clipping for stable training
- Comparison of RNN vs LSTM

---

### 3. Autoencoder for Gene Expression Analysis
**File:** `notebooks/03_Autoencoder_Gene_Expression.ipynb`

Discover unsupervised learning with autoencoders for dimensionality reduction of high-dimensional gene expression data.

**Topics Covered:**
- Unsupervised dimensionality reduction
- Latent space representation
- Clustering in reduced dimensions
- Comparison with PCA

**Key Techniques:**
- Encoder-decoder architecture
- Batch normalization and dropout
- t-SNE visualization
- Reconstruction quality analysis

---

### 4. Transfer Learning for Cell Image Classification
**File:** `notebooks/04_Transfer_Learning_Cell_Images.ipynb`

Master transfer learning using pre-trained ResNet models for cell type identification.

**Topics Covered:**
- Pre-trained model utilization
- Fine-tuning strategies
- Data augmentation techniques
- Image classification with limited data

**Key Techniques:**
- Loading pre-trained ResNet18
- Layer freezing and unfreezing
- ImageNet normalization
- Model evaluation and visualization

---

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose (for GPU support, install NVIDIA Container Toolkit)
- NVIDIA GPU (optional, but recommended for faster training)
- At least 8GB RAM
- 10GB free disk space

### Quick Start with Docker

1. **Clone the repository:**
```bash
git clone https://github.com/tonyliang19/deep-learning-biology-codebook.git
cd deep-learning-biology-codebook
```

2. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

3. **Access Jupyter:**
Open your browser and navigate to:
```
http://localhost:8888
```

The Jupyter interface will open with all notebooks ready to run!

### Alternative: Manual Installation

If you prefer not to use Docker:

1. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start Jupyter:**
```bash
jupyter notebook
```

---

## 🐳 Docker Setup Details

### GPU Support

The provided Docker setup includes GPU support via NVIDIA CUDA. To use GPU:

1. **Install NVIDIA Container Toolkit:**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. **Verify GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Building the Docker Image

```bash
# Build the image
docker build -t deep-learning-biology .

# Run with GPU support
docker run --gpus all -p 8888:8888 -v $(pwd)/notebooks:/workspace/notebooks deep-learning-biology
```

### Docker Compose Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up --build
```

---

## 📖 Learning Path

### For Beginners:
1. Start with **Notebook 1 (CNN for DNA)** - introduces basic deep learning concepts
2. Move to **Notebook 4 (Transfer Learning)** - learn about pre-trained models
3. Try **Notebook 3 (Autoencoder)** - explore unsupervised learning
4. Finally, **Notebook 2 (LSTM)** - understand sequential models

### For Intermediate Users:
- Work through notebooks in order (1 → 2 → 3 → 4)
- Experiment with hyperparameters
- Modify architectures to see effects
- Try with your own biological data

---

## 🛠️ Customization

### Using Your Own Data

Each notebook includes data generation functions. To use your own data:

1. **Sequence Data (DNA/Protein):**
   - Load from FASTA files using BioPython
   - Adapt encoding functions for your sequences
   - Adjust sequence lengths and vocabulary

2. **Gene Expression:**
   - Load from CSV/TSV files with pandas
   - Ensure samples are rows, genes are columns
   - Normalize appropriately for your platform (RNA-seq, microarray)

3. **Images:**
   - Organize images in class folders
   - Use `torchvision.datasets.ImageFolder` for easy loading
   - Adjust image size to match your data

### Modifying Architectures

All models are defined as PyTorch `nn.Module` classes. To modify:

```python
# Example: Add more layers to CNN
self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6)
self.pool3 = nn.MaxPool1d(kernel_size=2)
```

### Hyperparameter Tuning

Key hyperparameters to experiment with:
- **Learning rate**: 0.0001 to 0.01
- **Batch size**: 16, 32, 64
- **Number of layers**: 2-5 for most tasks
- **Hidden dimensions**: 64, 128, 256, 512
- **Dropout rate**: 0.1 to 0.5

---

## 📊 Expected Results

### Training Times (with GPU):
- **CNN (Notebook 1)**: ~5-10 minutes
- **LSTM (Notebook 2)**: ~10-15 minutes
- **Autoencoder (Notebook 3)**: ~10-15 minutes
- **Transfer Learning (Notebook 4)**: ~5-10 minutes

### Typical Accuracies (on synthetic data):
- **DNA Classification**: >90%
- **Protein Classification**: >85%
- **Cell Image Classification**: >90%

*Note: Results vary based on data complexity and randomness*

---

## 🔧 Troubleshooting

### Common Issues:

**1. Out of Memory Errors:**
```python
# Reduce batch size
batch_size = 16  # or 8
```

**2. CUDA Errors:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

**3. Slow Training on CPU:**
- Consider using Google Colab for free GPU access
- Or reduce model size and dataset size

**4. Module Import Errors:**
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

---

## 📚 Additional Resources

### Deep Learning:
- [Deep Learning Book](https://www.deeplearningbook.org/) by Goodfellow et al.
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Fast.ai Course](https://course.fast.ai/)

### Computational Biology:
- [Bioinformatics Algorithms](http://bioinformaticsalgorithms.com/)
- [Deep Learning in Life Sciences](https://mitpress.mit.edu/books/deep-learning-life-sciences)

### Datasets:
- **DNA/Protein**: UniProt, NCBI GenBank, Pfam
- **Gene Expression**: TCGA, GEO, GTEx
- **Images**: Broad Bioimage Benchmark Collection, Cell Image Library

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add new notebooks for different biological applications
- Improve existing explanations and code
- Fix bugs or typos
- Add support for additional frameworks (TensorFlow, JAX)

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- BioPython community for biological data tools
- Scientific Python ecosystem (NumPy, Pandas, Matplotlib)

---

## 📧 Contact

For questions, suggestions, or collaborations, please open an issue on GitHub.

---

## 🎯 Next Steps

After completing these notebooks, consider:
1. **Advanced architectures**: Transformers, Graph Neural Networks
2. **Real datasets**: Work with published biological datasets
3. **Production deployment**: Model serving with TorchServe or ONNX
4. **Interpretability**: SHAP values, attention visualization
5. **Multi-modal learning**: Combine sequences, images, and expression data