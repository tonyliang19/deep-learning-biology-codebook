# Migration from Jupyter Book to Quarto

## Summary of Changes

This document outlines the major changes made during the migration from Jupyter Book to Quarto.

## Why Quarto?

Quarto offers several advantages over Jupyter Book:

1. **Better notebook rendering**: Code outputs, plots, and mathematical equations render more beautifully
2. **Modern technology**: Active development with regular updates and improvements
3. **Better performance**: Faster build times and better caching
4. **Enhanced features**: Better support for cross-references, callouts, and interactive elements
5. **Multi-format output**: Easy generation of HTML, PDF, and other formats
6. **Mobile-friendly**: Responsive design that works well on all devices

## Notebook Reorganization

### Old Structure
```
notebooks/
├── 00_index.ipynb
├── 01_neural_networks_basics.ipynb
├── 02_convolutional_networks.ipynb
├── 03_transformers.ipynb
├── 04_vision_transformers.ipynb
├── 01_CNN_DNA_Sequence_Classification.ipynb
├── 02_RNN_LSTM_Protein_Sequence.ipynb
├── 03_Autoencoder_Gene_Expression.ipynb
└── 04_Transfer_Learning_Cell_Images.ipynb
```

### New Structure
```
notebooks/
├── 01_neural_networks_basics.ipynb          (Chapter 1)
├── 02_convolutional_networks.ipynb          (Chapter 2)
├── 03_transformers.ipynb                    (Chapter 3)
├── 04_vision_transformers.ipynb             (Chapter 4)
├── 05_CNN_DNA_Sequence_Classification.ipynb (Chapter 5)
├── 06_RNN_LSTM_Protein_Sequence.ipynb       (Chapter 6)
├── 07_Autoencoder_Gene_Expression.ipynb     (Chapter 7)
└── 08_Transfer_Learning_Cell_Images.ipynb   (Chapter 8)
```

The index notebook (00_index.ipynb) was replaced with `index.qmd`, a Quarto markdown file that serves as the book homepage.

## Documentation Enhancements

All notebooks have been significantly enhanced with more detailed, beginner-friendly explanations:

### Enhanced Sections

1. **Neural Networks Basics (Chapter 1)**
   - Added comprehensive introduction explaining neural networks with real-world analogies
   - Expanded activation functions section with detailed explanations of sigmoid, tanh, ReLU, and Leaky ReLU
   - Included "Don't Panic" sections to demystify mathematical notation
   - Added step-by-step breakdowns of how neurons process information

2. **Convolutional Networks (Chapter 2)**
   - Detailed explanation of why CNNs are necessary for images
   - Broke down the parameter explosion problem with concrete examples
   - Added intuitive explanations of convolution operations

3. **Transformers (Chapter 3)**
   - Added welcoming introduction positioning transformers in AI history
   - Explained attention mechanism with accessible language

4. **Vision Transformers (Chapter 4)**
   - Connected ViT to previous CNN knowledge
   - Explained patch embedding concept clearly

5. **Applied Notebooks (Chapters 5-8)**
   - Each now includes comprehensive biological context
   - "What You'll Learn" and "Skills You'll Practice" sections
   - Detailed problem statements explaining real-world applications
   - Step-by-step explanations of biological concepts

## New Files Added

- `_quarto.yml`: Main Quarto configuration file
- `index.qmd`: Book homepage (replaces 00_index.ipynb)
- `custom.scss`: Custom styling for the book
- `.github/workflows/publish-book.yml`: GitHub Actions workflow for automated deployment

## Updated Files

- `README.md`: Updated to reflect Quarto migration and beginner-friendly focus
- `docs/BUILD_BOOK.md`: Complete rewrite with Quarto instructions
- `.gitignore`: Added `_book/` to ignore Quarto build output

## Building the Book

### Local Development

```bash
# Render the book
quarto render

# Preview with live reload
quarto preview
```

### Deployment

The book can be automatically deployed to GitHub Pages using the included GitHub Actions workflow. The workflow:

1. Triggers on push to main branch
2. Sets up Quarto and Python environment
3. Installs dependencies
4. Renders the book
5. Deploys to GitHub Pages

## Configuration Highlights

### Book Structure

The book is organized into two parts:

1. **Fundamentals** (Chapters 1-4): Core deep learning concepts
2. **Applied Examples** (Chapters 5-8): Real-world biological applications

### Rendering Options

- **Execute**: `freeze: auto` - Notebooks are executed only when changed
- **Cache**: Enabled for faster subsequent builds
- **Code**: Visible by default with copy buttons
- **Theme**: Light (cosmo) and dark (darkly) modes available

## Target Audience

The enhanced documentation targets university year 2 students with:

- Basic Python programming knowledge
- High school level mathematics
- Limited statistics/machine learning background
- Interest in biology applications

## Key Improvements for Beginners

1. **Real-world analogies**: Complex concepts explained through everyday examples
2. **Mathematical notation guides**: "Plain English" translations of formulas
3. **Step-by-step breakdowns**: Detailed explanations of algorithms
4. **Biological context**: Clear motivation for why each technique matters
5. **Visual aids**: Emphasis on diagrams and visualizations
6. **Progressive complexity**: Concepts build naturally from simple to advanced

## Backward Compatibility

The old Jupyter Book configuration files (`_config.yml`, `_toc.yml`) remain in the repository but are no longer used. They can be removed in a future cleanup if desired.

## Future Enhancements

Potential improvements for future iterations:

1. Add interactive widgets using Quarto's Observable integration
2. Include video tutorials embedded in chapters
3. Add exercise notebooks with solutions
4. Create a glossary of terms
5. Add more code examples with detailed comments
6. Include links to external resources and datasets
