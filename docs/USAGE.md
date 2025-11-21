# Usage Guide

This comprehensive guide will help you get started with the Deep Learning for Biology Codebook, from setting up your environment to running your first notebook.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Installing Dependencies](#installing-dependencies)
3. [Running Jupyter Notebooks](#running-jupyter-notebooks)
4. [Working with Notebooks](#working-with-notebooks)
5. [Data Management](#data-management)
6. [GPU Configuration](#gpu-configuration)
7. [Common Workflows](#common-workflows)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

## Environment Setup

### Option 1: Using Virtual Environment (venv)

Recommended for most users who work with Python regularly.

#### Step 1: Create Virtual Environment

```bash
# Navigate to the project directory
cd deep-learning-biology-codebook

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Step 2: Verify Activation

Your terminal prompt should now show `(venv)` at the beginning, indicating the virtual environment is active.

```bash
# Check Python location
which python  # On Linux/Mac
where python  # On Windows

# Should point to the venv directory
```

### Option 2: Using Conda

Recommended for users who need better dependency management or work with multiple languages.

#### Step 1: Create Conda Environment

```bash
# Create environment with specific Python version
conda create -n bio-dl python=3.9

# Activate the environment
conda activate bio-dl
```

#### Step 2: Verify Environment

```bash
# Check active environment
conda info --envs

# Current environment should have an asterisk (*)
```

### Option 3: Using Docker (Advanced)

For users who want isolated, reproducible environments.

```bash
# Build Docker image (when Dockerfile is provided)
docker build -t bio-dl-codebook .

# Run container with Jupyter
docker run -p 8888:8888 -v $(pwd):/workspace bio-dl-codebook

# Access Jupyter at http://localhost:8888
```

## Installing Dependencies

### Using pip (with venv)

```bash
# Make sure your virtual environment is activated
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list
```

### Using conda

```bash
# Install from requirements file
conda install --file requirements.txt

# Or install from environment.yml if provided
conda env create -f environment.yml

# Update existing environment
conda env update -f environment.yml
```

### Core Dependencies

The main packages you'll be using include:

**Deep Learning Framework:**
- `torch` (PyTorch) - Deep learning framework
- `torchvision` - Computer vision utilities
- `torchaudio` - Audio processing utilities

**Biology & Bioinformatics:**
- `biopython` - Sequence analysis
- `scikit-bio` - Bioinformatics algorithms
- `pysam` - SAM/BAM file handling
- `scanpy` - Single-cell analysis

**Data Science:**
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scipy` - Scientific computing
- `scikit-learn` - Machine learning utilities

**Visualization:**
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive plots

**Jupyter:**
- `jupyter` - Notebook interface
- `jupyterlab` - Advanced notebook environment
- `ipywidgets` - Interactive widgets

## Running Jupyter Notebooks

### Starting Jupyter Notebook

```bash
# Activate your environment first
source venv/bin/activate  # or: conda activate bio-dl

# Start Jupyter Notebook
jupyter notebook

# Specify port (if 8888 is occupied)
jupyter notebook --port 8889

# Allow external connections (use with caution)
jupyter notebook --ip 0.0.0.0
```

Your browser should automatically open to `http://localhost:8888`. If not, copy the URL with the token from the terminal.

### Starting JupyterLab (Recommended)

JupyterLab provides a more powerful interface:

```bash
# Start JupyterLab
jupyter lab

# With specific settings
jupyter lab --port 8890 --no-browser
```

### Using VS Code

If you prefer VS Code:

1. Install the "Jupyter" extension from the VS Code marketplace
2. Open a `.ipynb` file
3. Select your Python interpreter (the one from your virtual environment)
4. Run cells directly in VS Code

### Using Google Colab

For cloud-based execution (free GPU access):

1. Upload notebooks to Google Drive
2. Open with Google Colab
3. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
4. Install additional packages: `!pip install package-name`

## Working with Notebooks

### Notebook Basics

#### Cell Types

- **Code Cell**: Python code that can be executed
- **Markdown Cell**: Documentation, explanations, equations
- **Raw Cell**: Unformatted text (rarely used)

#### Keyboard Shortcuts

In **Command Mode** (press `Esc`):
- `A` - Insert cell above
- `B` - Insert cell below
- `D, D` - Delete cell
- `M` - Convert to Markdown
- `Y` - Convert to Code
- `Shift + Enter` - Run cell and move to next
- `Ctrl + Enter` - Run cell and stay

In **Edit Mode** (press `Enter`):
- `Tab` - Code completion
- `Shift + Tab` - Show documentation
- `Ctrl + /` - Comment/uncomment
- `Esc` - Return to command mode

### Running a Complete Notebook

#### Option 1: Cell by Cell (Recommended for Learning)

1. Start at the first cell
2. Read the explanations
3. Run each cell with `Shift + Enter`
4. Observe outputs and understand what happened
5. Modify and re-run to experiment

#### Option 2: Run All Cells

```python
# In Jupyter menu: Cell > Run All
# Or use: Runtime > Run All (in Colab)
```

**Warning**: Some notebooks may take significant time or require user input. Running all cells is best for notebooks you've already understood.

### Modifying Notebooks

#### Experiment Safely

```python
# Original code
model = create_model(layers=3)

# Try different parameters
model = create_model(layers=5)  # Experiment!

# Use cells to compare
print("3 layers:", results_3_layers)
print("5 layers:", results_5_layers)
```

#### Add Your Own Cells

- Add cells to try variations
- Insert markdown cells for your notes
- Keep original cells for reference

#### Save Checkpoints

- Manually: `File > Save and Checkpoint`
- Auto-save is enabled by default
- Consider version control: `git commit -m "Experimented with hyperparameters"`

## Data Management

### Understanding Data Paths

Notebooks reference data in several ways:

```python
# Relative path (from notebook location)
data_path = "../data/sequences.fasta"

# Absolute path
data_path = "/home/user/data/sequences.fasta"

# Using Path from pathlib (recommended)
from pathlib import Path
data_path = Path(__file__).parent / "data" / "sequences.fasta"
```

### Downloading Sample Data

Many notebooks include data download cells:

```python
# Example: Download from URL
import urllib.request
url = "http://example.com/dataset.csv"
urllib.request.urlretrieve(url, "data/dataset.csv")

# Example: Using provided scripts
!python data/download_scripts/get_protein_data.py
```

### Working with Large Datasets

For datasets too large to include in the repository:

#### Option 1: Download Scripts

```bash
# Run provided download scripts
cd data/download_scripts
python download_genomics_data.py --dataset ENCODE

# Or from within notebook
!python data/download_scripts/download_genomics_data.py --dataset ENCODE
```

#### Option 2: Manual Download

1. Follow links provided in the notebook
2. Download to the `data/` directory
3. Verify file paths match notebook expectations

#### Option 3: Use Symbolic Links

```bash
# If data exists elsewhere on your system
cd data
ln -s /path/to/large/dataset ./dataset_name
```

### Data Preprocessing

#### Caching Preprocessed Data

```python
import os
import pickle

# Check if preprocessed data exists
cache_file = "data/cache/preprocessed_data.pkl"
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    print("Loaded from cache")
else:
    # Preprocess data (time-consuming)
    data = preprocess_raw_data()
    
    # Save to cache
    os.makedirs("data/cache", exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print("Saved to cache")
```

## GPU Configuration

### Checking GPU Availability

#### PyTorch

```python
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
```

### Using GPU in Notebooks

All notebooks use PyTorch and automatically use GPU if available:

```python
# PyTorch - models and tensors are moved to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = inputs.to(device)
```

### GPU Memory Management

```python
# PyTorch - Clear cache
torch.cuda.empty_cache()

# Reduce batch size if OOM (Out Of Memory)
batch_size = 16  # Try smaller: 8, 4, 2

# Enable gradient checkpointing (saves memory)
model.gradient_checkpointing_enable()
```

### CPU-Only Execution

To force CPU execution (useful for debugging):

```python
# PyTorch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

## Common Workflows

### Workflow 1: Exploring a New Topic

1. **Read the introduction** - Understand the biological problem
2. **Review the background** - Grasp the methodology
3. **Run data loading cells** - See the data structure
4. **Execute model definition** - Understand the architecture
5. **Skip training** (initially) - Use pre-trained weights if provided
6. **Analyze results** - Focus on interpretation
7. **Return to training** - When ready to experiment

### Workflow 2: Adapting to Your Data

1. **Find similar example** - Locate relevant notebook
2. **Copy the notebook** - Create your own version
3. **Modify data loading** - Point to your data
4. **Adjust preprocessing** - Match your data format
5. **Update model if needed** - Adapt architecture
6. **Retrain** - Fit to your data
7. **Evaluate** - Assess performance

### Workflow 3: Hyperparameter Tuning

1. **Baseline run** - Use default parameters
2. **Identify bottlenecks** - Analyze performance
3. **Systematic search** - Try variations
4. **Log results** - Track experiments
5. **Select best** - Based on validation metrics
6. **Final evaluation** - Test on held-out data

### Workflow 4: Building a Pipeline

```python
# Example: Create a processing pipeline
from pathlib import Path

def run_analysis_pipeline(input_file, output_dir):
    """Complete analysis pipeline."""
    
    # 1. Load data
    data = load_data(input_file)
    
    # 2. Preprocess
    processed = preprocess(data)
    
    # 3. Feature extraction
    features = extract_features(processed)
    
    # 4. Model prediction
    predictions = model.predict(features)
    
    # 5. Post-processing
    results = postprocess(predictions)
    
    # 6. Save outputs
    save_results(results, output_dir)
    
    return results

# Run pipeline
results = run_analysis_pipeline(
    "data/my_data.fasta",
    "results/my_analysis"
)
```

## Troubleshooting

### Issue: Module Not Found

```bash
# Error: ModuleNotFoundError: No module named 'some_package'

# Solution 1: Install the missing package
pip install some_package

# Solution 2: Check if you're in the right environment
which python  # Should point to your venv/conda

# Solution 3: Restart kernel after installation
# Kernel > Restart in Jupyter
```

### Issue: Kernel Crashes

```python
# Common causes:
# 1. Out of memory - Reduce batch size
# 2. Infinite loop - Check code logic
# 3. Conflicting versions - Update packages

# Solutions:
# - Restart kernel: Kernel > Restart
# - Clear outputs: Cell > All Output > Clear
# - Restart Jupyter server
# - Check system resources: htop or Activity Monitor
```

### Issue: CUDA Out of Memory

```python
# Error: RuntimeError: CUDA out of memory

# Solution 1: Reduce batch size
batch_size = 16  # Try 8, 4, or 2

# Solution 2: Use gradient accumulation
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Solution 3: Use CPU for large inputs
device = "cpu"
```

### Issue: Slow Training

```python
# Check:
# 1. Are you using GPU?
print(next(model.parameters()).device)  # Should show 'cuda'

# 2. Are you using efficient data loading?
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

# 3. Are you using mixed precision?
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 4. Profile your code
import cProfile
cProfile.run('train_model()')
```

### Issue: Results Don't Match

```python
# Set random seeds for reproducibility
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Issue: Package Version Conflicts

```bash
# Check installed versions
pip list | grep package-name

# Install specific version
pip install package-name==1.2.3

# Update requirements
pip freeze > requirements.txt

# Create fresh environment if needed
conda create -n bio-dl-fresh python=3.9
conda activate bio-dl-fresh
pip install -r requirements.txt
```

## Best Practices

### Code Organization

```python
# Group imports at the top
import numpy as np
import pandas as pd
from pathlib import Path

# Set configurations early
RANDOM_SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Define helper functions
def load_data(path):
    """Load and validate data."""
    pass

# Main analysis in sequential cells
```

### Documentation

```python
# Add markdown cells to explain your reasoning
"""
## Experiment: Testing Larger Model

I'm increasing the model size to see if it improves
performance on this dataset. Previous model had 3 layers,
trying 5 layers now.
"""

# Add comments for complex operations
# Extract features using sliding window (size=100, step=50)
features = extract_features(sequence, window=100, step=50)
```

### Version Control

```bash
# Track notebook changes with git
git add notebooks/my_experiment.ipynb
git commit -m "Add experiment with 5-layer model"

# Clear outputs before committing (optional)
jupyter nbconvert --clear-output --inplace notebook.ipynb

# Use .gitignore for large files
echo "data/*.csv" >> .gitignore
echo "models/*.pth" >> .gitignore
```

### Experiment Tracking

```python
# Keep a log of experiments
import json
from datetime import datetime

experiment_log = {
    "timestamp": datetime.now().isoformat(),
    "model": "ResNet-18",
    "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50
    },
    "results": {
        "accuracy": 0.87,
        "loss": 0.23
    }
}

# Save to file
with open("experiments/log.json", "a") as f:
    f.write(json.dumps(experiment_log) + "\n")
```

### Resource Management

```python
# Clean up after experiments
import gc
import torch

# Clear variables
del large_dataset
gc.collect()

# Clear GPU cache
torch.cuda.empty_cache()

# Close file handles
with open("data.txt") as f:
    data = f.read()
# File automatically closed after 'with' block
```

## Getting Help

If you encounter issues not covered here:

1. **Check the FAQ** in the main README
2. **Search existing issues** on GitHub
3. **Ask in discussions** for general questions
4. **Open an issue** for bugs or feature requests
5. **Consult RESOURCES.md** for learning materials

## Next Steps

- **Start with fundamentals**: Begin with `notebooks/01_fundamentals/`
- **Follow your interests**: Jump to relevant topics
- **Join the community**: Share your experiences and learn from others
- **Contribute back**: Improve notebooks and documentation

---

Happy coding! 🧬🤖

For more information, see:
- [README.md](README.md) - Project overview
- [RESOURCES.md](RESOURCES.md) - Learning resources
- [GitHub Issues](https://github.com/tonyliang19/deep-learning-biology-codebook/issues) - Support
