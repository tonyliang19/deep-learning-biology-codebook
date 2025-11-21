# Deep Learning Biology Codebook 📚

A comprehensive, user-friendly guide to deep learning with a focus on biological applications. This codebook provides in-depth explanations and practical examples covering everything from basic neural networks to advanced architectures like Transformers and Vision Transformers.

## 🎯 Purpose

This codebook is designed to be:
- **Educational**: Step-by-step explanations with mathematical foundations
- **Practical**: Working code examples you can run and modify
- **Comprehensive**: Covers fundamental to advanced topics
- **Biology-focused**: Examples and applications relevant to biological data

## 📖 Table of Contents

### [0. Index & Quick Start](notebooks/00_index.ipynb)
Start here for an overview and quick navigation guide

### [1. Introduction to Neural Networks](notebooks/01_neural_networks_basics.ipynb)
- What are Neural Networks?
- Perceptrons and Multi-Layer Perceptrons (MLPs)
- Activation Functions
- Forward and Backward Propagation
- Loss Functions and Optimization
- Training Your First Neural Network
- **Biology Application**: Gene Expression Classification

### [2. Convolutional Neural Networks (CNNs)](notebooks/02_convolutional_networks.ipynb)
- Understanding Convolutions
- CNN Architecture Components
  - Convolutional Layers
  - Pooling Layers
  - Fully Connected Layers
- Popular CNN Architectures (LeNet, AlexNet, VGG, ResNet)
- Transfer Learning
- **Biology Application**: Cell Image Classification

### [3. Transformers](notebooks/03_transformers.ipynb)
- Attention Mechanism
- Self-Attention and Multi-Head Attention
- Positional Encoding
- Transformer Architecture
- BERT and GPT Overview
- **Biology Application**: Protein Sequence Analysis

### [4. Vision Transformers (ViT)](notebooks/04_vision_transformers.ipynb)
- From CNN to ViT
- Patch Embedding
- ViT Architecture
- Comparing ViT with CNNs
- Hybrid Architectures
- **Biology Application**: Medical Image Analysis

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

# Install dependencies
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

## 📚 How to Use This Codebook

1. **Sequential Learning**: Start from Chapter 1 and progress through each chapter
2. **Quick Reference**: Use the index notebook to jump to specific topics
3. **Hands-on Practice**: Run all code cells and experiment with parameters
4. **Biology Focus**: Pay special attention to the biology application sections

## 🔧 Requirements

See [requirements.txt](requirements.txt) for the complete list of dependencies.

Main libraries used:
- PyTorch / TensorFlow for deep learning
- NumPy and Pandas for data manipulation
- Matplotlib and Seaborn for visualization
- Transformers library for pre-trained models

## 📝 Contributing

Contributions are welcome! If you find errors or have suggestions for improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

This codebook is inspired by various deep learning resources and adapted for biological applications.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy Learning! 🧬🤖**