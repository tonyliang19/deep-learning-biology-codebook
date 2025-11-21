# Building the Book with Quarto

This codebook is built as an interactive online book using [Quarto](https://quarto.org/). Quarto provides excellent support for Jupyter notebooks with beautiful rendering of code outputs, plots, and mathematical equations.

## Why We Switched from Jupyter Book to Quarto

**Advantages of Quarto:**
- Better rendering of notebook outputs and plots
- More modern and actively developed
- Excellent support for multiple output formats (HTML, PDF, ePub)
- Better integration with Python, R, and Julia
- More flexible customization options
- Faster build times
- Better mobile responsiveness

## Prerequisites

### Install Quarto

**Linux:**
```bash
# Download and install the latest version
wget https://github.com/quarto-dev/quarto-cli/releases/download/v1.4.549/quarto-1.4.549-linux-amd64.deb
sudo dpkg -i quarto-1.4.549-linux-amd64.deb
```

**macOS:**
```bash
# Using Homebrew
brew install quarto

# Or download the installer from https://quarto.org/docs/get-started/
```

**Windows:**
Download and run the installer from [https://quarto.org/docs/get-started/](https://quarto.org/docs/get-started/)

Verify installation:
```bash
quarto --version
```

## Building the HTML Book

### 1. Render the book

From the repository root:

```bash
quarto render
```

This will:
- Execute or use cached outputs from all Jupyter notebooks
- Generate HTML from notebooks and Quarto markdown files
- Create a table of contents and navigation
- Apply the custom theme
- Build the search index

The output will be in the `_book/` directory.

### 2. Preview the book

For live preview with auto-reload:

```bash
quarto preview
```

This will:
- Start a local web server
- Open your browser automatically
- Watch for file changes and auto-reload

Or manually serve the built book:

```bash
cd _book
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

## Building a PDF

Quarto can generate PDF output:

```bash
quarto render --to pdf
```

Requirements:
- LaTeX distribution (TeX Live recommended)
- Install with: `quarto install tinytex` (Quarto's lightweight LaTeX)

## Publishing to GitHub Pages

### 1. Install ghp-import

```bash
pip install ghp-import
```

### 2. Build and publish

```bash
# Build the book
jupyter-book build .

# Publish to gh-pages branch
ghp-import -n -p -f _build/html
```

Your book will be available at: `https://tonyliang19.github.io/deep-learning-biology-codebook/`

## Customization

### Modify _config.yml

Edit `_config.yml` to customize:
- Book title and author
- Theme colors and layout
- Execution settings
- Repository links

### Modify _toc.yml

Edit `_toc.yml` to:
- Change chapter order
- Add new chapters
- Organize sections
- Add parts/sections

### Add Custom CSS

Create `_static/custom.css` and reference it in `_config.yml`:

```yaml
html:
  extra_css:
    - _static/custom.css
```

## Advanced Features

### Enable Execution

To execute notebooks during build (useful for ensuring all code runs):

```yaml
# In _config.yml
execute:
  execute_notebooks: force
  timeout: 600
```

### Add Binder Integration

Allow readers to run notebooks in the cloud:

```yaml
# In _config.yml
launch_buttons:
  binderhub_url: "https://mybinder.org"
```

### Add Google Colab Links

```yaml
# In _config.yml
launch_buttons:
  colab_url: "https://colab.research.google.com"
```

## Troubleshooting

### Build fails

```bash
# Clean previous builds
jupyter-book clean .

# Rebuild
jupyter-book build .
```

### Notebook execution errors

Set execution to 'off' in `_config.yml` if notebooks require special environments:

```yaml
execute:
  execute_notebooks: off
```

### Missing dependencies

```bash
# Install all dependencies needed for building
pip install -r requirements.txt
pip install jupyter-book ghp-import
```

## Continuous Integration

### GitHub Actions

Create `.github/workflows/deploy-book.yml`:

```yaml
name: Deploy Book

on:
  push:
    branches: [main]

jobs:
  deploy-book:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install jupyter-book ghp-import
    
    - name: Build book
      run: jupyter-book build .
    
    - name: Deploy to GitHub Pages
      run: ghp-import -n -p -f _build/html
```

## Resources

- [Jupyter Book Documentation](https://jupyterbook.org/)
- [MyST Markdown Guide](https://myst-parser.readthedocs.io/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)

---

Happy building! 📚✨
