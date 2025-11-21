# Deep Learning Biology Codebook 📚

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Quarto](https://img.shields.io/badge/Made%20with-Quarto-blue.svg)](https://quarto.org/)

A comprehensive, beginner-friendly guide to deep learning with a focus on biological applications. This codebook provides in-depth explanations and practical examples covering everything from basic neural networks to advanced architectures like Transformers and Vision Transformers. All content is designed to be accessible to university year 2 students with basic programming knowledge.

## 🎯 Purpose

This codebook is designed to be:
- **Beginner-Friendly**: Written for university year 2 students with basic programming knowledge
- **Educational**: Step-by-step explanations with mathematical foundations made accessible
- **Practical**: Working code examples you can run and modify
- **Comprehensive**: Covers fundamental to advanced topics with detailed explanations
- **Biology-focused**: Examples and applications relevant to biological data

## 📖 Learning Track: Fundamentals

### [1. Introduction to Neural Networks](notebooks/01_neural_networks_basics.ipynb)
Learn the basics with enhanced explanations designed for beginners:
- What are Neural Networks? (Intuitive explanations with real-world analogies)
- Perceptrons and Multi-Layer Perceptrons (MLPs)
- Activation Functions (Step-by-step mathematical explanations)
- Forward and Backward Propagation
- Loss Functions and Optimization
- Training Your First Neural Network
- **Biology Application**: Gene Expression Classification

### [2. Convolutional Neural Networks (CNNs)](notebooks/02_convolutional_networks.ipynb)
Understand CNNs with detailed visual explanations:
- Understanding Convolutions (Why they're needed for images)
- CNN Architecture Components
  - Convolutional Layers
  - Pooling Layers
  - Fully Connected Layers
- Popular CNN Architectures (LeNet, AlexNet, VGG, ResNet)
- Transfer Learning
- **Biology Application**: Cell Image Classification

### [3. Transformers](notebooks/03_transformers.ipynb)
Demystify transformers with accessible explanations:
- Attention Mechanism (The key innovation explained simply)
- Self-Attention and Multi-Head Attention
- Positional Encoding
- Transformer Architecture
- BERT and GPT Overview
- **Biology Application**: Protein Sequence Analysis

### [4. Vision Transformers (ViT)](notebooks/04_vision_transformers.ipynb)
Learn cutting-edge computer vision:
- From CNN to ViT
- Patch Embedding (Breaking images into pieces)
- ViT Architecture
- Comparing ViT with CNNs
- Hybrid Architectures
- **Biology Application**: Medical Image Analysis

## 📊 Applied Notebooks: Real-World Biology Examples

These notebooks provide complete, end-to-end examples of applying deep learning to real biological problems, with enhanced step-by-step explanations:

### [5. CNN for DNA Sequence Classification](notebooks/05_CNN_DNA_Sequence_Classification.ipynb)
Learn how CNNs identify patterns in DNA sequences for promoter region detection.

**Topics**: One-hot encoding, Conv1D layers, motif visualization, binary classification  
**New explanations**: Detailed introduction to promoters, DNA representation, and biological context

### [6. RNN/LSTM for Protein Sequence Analysis](notebooks/06_RNN_LSTM_Protein_Sequence.ipynb)
Explore RNNs and LSTMs for protein family classification.

**Topics**: Embeddings, bidirectional LSTM, variable-length sequences, multi-class classification  
**New explanations**: What proteins are, why sequence classification matters, RNN advantages

### [7. Autoencoder for Gene Expression Analysis](notebooks/07_Autoencoder_Gene_Expression.ipynb)
Discover unsupervised learning with autoencoders for dimensionality reduction.

**Topics**: Encoder-decoder architecture, latent space, clustering, comparison with PCA  
**New explanations**: Gene expression data basics, dimensionality reduction need, autoencoder intuition

### [8. Transfer Learning with Cell Images](notebooks/08_Transfer_Learning_Cell_Images.ipynb)
Master transfer learning for cell image classification with pre-trained models.

**Topics**: Fine-tuning, feature extraction, data augmentation, model comparison  
**New explanations**: Small data problem in biology, transfer learning strategy, feature extraction vs fine-tuning

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8 or higher
python --version
```

### Installation
```bash
# Clone the repository
git clone https://github.com/tonyliang19/deep-learning-biology-codebook.git
cd deep-learning-biology-codebook

# Install dependencies (PyTorch-based)
pip install -r requirements.txt
```

### Running the Notebooks
```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Then navigate to the `notebooks/` directory and start with `00_index.ipynb`.

### Docker Setup (Alternative)

For a consistent environment across different systems:

```bash
# Build and run with Docker Compose
docker-compose up

# Access Jupyter at http://localhost:8888
# Token will be displayed in the terminal
```

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for detailed setup instructions.

## 📚 How to Use This Codebook

1. **Sequential Learning**: Start from Chapter 1 and progress through each chapter
2. **Quick Reference**: Use the index notebook to jump to specific topics
3. **Hands-on Practice**: Run all code cells and experiment with parameters
4. **Biology Focus**: Pay special attention to the biology application sections
5. **Applied Examples**: Work through the applied notebooks for complete projects

### Using as an Interactive Book

This repository is built as a beautiful interactive book using [Quarto](https://quarto.org/)!

**Build the book locally:**
```bash
# Install Quarto from https://quarto.org/docs/get-started/

# Render the book
quarto render

# Preview the book (opens in browser)
quarto preview
```

The rendered book includes:
- **All code outputs**: See plots, tables, and results directly in the book
- **Syntax highlighting**: Beautiful code formatting
- **Interactive navigation**: Table of contents, search functionality
- **Mobile-friendly**: Responsive design works on all devices

See [docs/BUILD_BOOK.md](docs/BUILD_BOOK.md) for more details on:
- Publishing to GitHub Pages
- Customizing the appearance
- Adding custom styling

## 🔧 Requirements

See [requirements.txt](requirements.txt) for all dependencies.

Main libraries used:
- PyTorch for deep learning
- NumPy and Pandas for data manipulation
- Matplotlib and Seaborn for visualization
- Transformers library for pre-trained models
- Biopython and scikit-bio for biology-specific operations

## 🌟 Features

✨ **Comprehensive Coverage**: From basics to advanced topics  
📖 **In-depth Explanations**: Theory + Math + Code  
🧬 **Biology-Focused**: Real biological applications  
💻 **Working Code**: All examples tested and runnable  
📊 **Rich Visualizations**: Plots, diagrams, and illustrations  
📚 **Book Format**: Can be built as interactive online book  
🐳 **Docker Support**: Reproducible environment  
🚀 **Easy Setup**: Clear installation and usage instructions  

## 📚 Additional Resources

For more information, check out the [docs/](docs/) directory:
- **[RESOURCES.md](docs/RESOURCES.md)**: Books, courses, papers, tools, and datasets
- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)**: Guidelines for contributing
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)**: Quick reference guide
- **[QUICKSTART.md](docs/QUICKSTART.md)**: Quick start guide
- **[USAGE.md](docs/USAGE.md)**: Detailed usage instructions
- **[SUMMARY.md](docs/SUMMARY.md)**: Content summary

## 🙏 Acknowledgments

This codebook is inspired by various deep learning resources and adapted for biological applications.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 📄 License

This project is open source and available under the MIT License for educational purposes.

---

**Happy Learning! 🧬🤖**
