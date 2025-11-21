# Building the Book

This codebook can be converted into an interactive online book using [Jupyter Book](https://jupyterbook.org/).

## Prerequisites

```bash
pip install jupyter-book
```

## Building the HTML Book

### 1. Build the book

```bash
jupyter-book build .
```

This will:
- Generate HTML from notebooks and markdown files
- Create a table of contents
- Add navigation elements
- Apply the theme

### 2. View the book

Open `_build/html/index.html` in your browser, or serve it locally:

```bash
# Option 1: Python's built-in server
cd _build/html
python -m http.server 8000

# Option 2: Using Jupyter Book's serve command
jupyter-book build . --builder dirhtml
```

Then open http://localhost:8000 in your browser.

## Building a PDF

```bash
jupyter-book build . --builder pdflatex
```

Requirements:
- LaTeX distribution (e.g., TeX Live, MiKTeX)
- Additional packages may be needed

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
pip install -r requirements.txt
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
