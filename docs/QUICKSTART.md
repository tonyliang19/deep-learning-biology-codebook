# Quick Start Guide

Get started with the Deep Learning Biology Codebook in 5 minutes!

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/tonyliang19/deep-learning-biology-codebook.git
cd deep-learning-biology-codebook
```

### Step 2: Set Up Python Environment

**Option A: Using conda (Recommended)**

```bash
# Create environment
conda create -n dl-bio python=3.8

# Activate environment
conda activate dl-bio

# Install dependencies
pip install -r requirements.txt
```

**Option B: Using venv**

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Launch Jupyter

```bash
# Option 1: Jupyter Notebook
jupyter notebook

# Option 2: JupyterLab (Recommended)
jupyter lab
```

A browser window will open. Navigate to `notebooks/` and start with `00_index.ipynb`!

## 📚 Learning Path

### Complete Beginner Path
Follow chapters sequentially:
1. **00_index.ipynb** - Orientation and setup
2. **01_neural_networks_basics.ipynb** - Foundation (2-3 hours)
3. **02_convolutional_networks.ipynb** - Computer Vision (3-4 hours)
4. **03_transformers.ipynb** - Sequence Modeling (4-5 hours)
5. **04_vision_transformers.ipynb** - Advanced Vision (3-4 hours)

**Total time**: ~15-20 hours

### Quick Reference Path
Jump directly to topics you need:
- Need CNNs? → Chapter 2
- Working with sequences? → Chapter 3
- Modern vision models? → Chapter 4

## ✅ First Steps

### 1. Test Your Installation

Open `notebooks/00_index.ipynb` and run the first code cell:

```python
import sys
print(f"Python version: {sys.version}")

libraries = ['torch', 'numpy', 'pandas', 'matplotlib']
for lib in libraries:
    try:
        module = __import__(lib)
        print(f"✓ {lib}: installed")
    except ImportError:
        print(f"✗ {lib}: NOT INSTALLED")
```

All libraries should show "✓ installed".

### 2. Run a Quick Example

Try this simple neural network:

```python
import torch
import torch.nn as nn

# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Test it
x = torch.randn(5, 10)
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print("✓ Everything works!")
```

### 3. Start Learning!

Open **Chapter 1** and begin your journey:
```bash
# In Jupyter, open:
notebooks/01_neural_networks_basics.ipynb
```

## 🎯 Tips for Success

### Study Habits
- 📖 **Read actively**: Don't just read, run the code
- 🔄 **Experiment**: Change parameters and see what happens
- ✏️ **Take notes**: Add your own markdown cells
- 🏃 **Practice**: Try the exercises at the end of each chapter

### Code Execution
- **Run cells in order**: Top to bottom for best results
- **Restart kernel**: If things break, Kernel → Restart & Clear Output
- **Save often**: Ctrl+S (Cmd+S on Mac)

### Getting Help
- Check the explanations in markdown cells
- Look at code comments
- Google error messages
- Open an issue on GitHub

## 🧠 Key Concepts by Chapter

### Chapter 1: Neural Networks
- Perceptrons and MLPs
- Activation functions (ReLU, Sigmoid)
- Forward and backward propagation
- Loss functions and optimizers

### Chapter 2: CNNs
- Convolution operations
- Pooling layers
- CNN architectures (ResNet, VGG)
- Transfer learning

### Chapter 3: Transformers
- Attention mechanism
- Self-attention and multi-head attention
- Positional encoding
- BERT and GPT

### Chapter 4: Vision Transformers
- Patch embeddings
- ViT architecture
- ViT vs CNNs
- Hybrid models

## 🧬 Biology Applications

Each chapter includes real biological applications:

- **Chapter 1**: Gene expression classification
- **Chapter 2**: Cell image classification
- **Chapter 3**: Protein sequence analysis
- **Chapter 4**: Medical image analysis

## ⚡ Troubleshooting

### Common Issues

**"Module not found"**
```bash
pip install <module-name>
```

**"CUDA not available"**
- Not a problem! All examples work on CPU
- For GPU support, install PyTorch with CUDA

**"Kernel died"**
- Restart kernel and run cells again
- Check available memory (notebooks may need 4-8GB RAM)

**Notebooks won't open**
- Make sure you're in the correct directory
- Check file permissions

## 📖 Additional Resources

### Books
- "Deep Learning" by Goodfellow, Bengio, Courville
- "Dive into Deep Learning" (d2l.ai) - Free online!

### Online Courses
- fast.ai - Practical Deep Learning
- Stanford CS231n - CNNs
- Stanford CS224n - NLP/Transformers

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## 🎓 After Completion

Once you finish all chapters:

1. **Build something**: Apply to your own data
2. **Share**: Publish your results
3. **Contribute**: Help improve this codebook
4. **Keep learning**: Deep learning evolves rapidly!

## 🤝 Community

- **Issues**: Report bugs or ask questions on GitHub
- **Discussions**: Share your projects and ideas
- **Contributions**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Ready? Let's Go! 🚀

```bash
jupyter lab
# Open notebooks/00_index.ipynb
# Start learning!
```

**Happy Learning!** 🧬🤖
