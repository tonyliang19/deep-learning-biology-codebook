# Data Directory

This directory is for storing datasets used in the biology applications throughout the notebooks.

## Suggested Datasets

### For Neural Networks (Chapter 1)
- **Gene Expression Data**: Classification of cancer types
  - Source: [TCGA](https://www.cancer.gov/tcga)
  - Format: CSV with gene expression values
  - Task: Binary or multi-class classification

### For CNNs (Chapter 2)
- **Cell Image Data**: Cell type classification
  - Source: [HPA Cell Atlas](https://www.proteinatlas.org/humanproteome/cell)
  - Format: Images (PNG/JPEG)
  - Task: Image classification
  
- **Microscopy Images**: Organelle detection
  - Source: [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/)
  - Format: TIFF images
  - Task: Object detection/segmentation

### For Transformers (Chapter 3)
- **Protein Sequences**: Function prediction
  - Source: [UniProt](https://www.uniprot.org/)
  - Format: FASTA
  - Task: Sequence classification
  
- **DNA/RNA Sequences**: Promoter prediction
  - Source: [NCBI GenBank](https://www.ncbi.nlm.nih.gov/genbank/)
  - Format: FASTA
  - Task: Sequence classification

### For Vision Transformers (Chapter 4)
- **Medical Images**: Disease diagnosis
  - Source: [NIH Chest X-ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
  - Format: PNG images
  - Task: Multi-label classification
  
- **Histopathology Images**: Cancer detection
  - Source: [TCGA](https://www.cancer.gov/tcga) or [CAMELYON](https://camelyon17.grand-challenge.org/)
  - Format: Whole slide images (WSI)
  - Task: Classification/segmentation

## Data Structure

Organize your data as follows:

```
data/
├── README.md (this file)
├── gene_expression/
│   ├── train.csv
│   ├── test.csv
│   └── metadata.json
├── cell_images/
│   ├── train/
│   │   ├── class_1/
│   │   └── class_2/
│   └── test/
│       ├── class_1/
│       └── class_2/
├── protein_sequences/
│   ├── train.fasta
│   ├── test.fasta
│   └── labels.csv
└── medical_images/
    ├── train/
    └── test/
```

## Data Preprocessing

Each notebook includes code for:
- Loading data
- Preprocessing
- Splitting train/validation/test sets
- Creating data loaders

## Privacy and Ethics

⚠️ **Important**: When working with biological and medical data:

1. **Privacy**: Ensure data is de-identified
2. **Consent**: Only use data with proper consent
3. **Ethics**: Follow institutional review board (IRB) guidelines
4. **Attribution**: Cite data sources properly
5. **Licensing**: Respect data usage restrictions

## Downloading Data

Most datasets require registration or agreements. Follow the links above to:
1. Visit the data source
2. Read the terms of use
3. Register if required
4. Download the data
5. Place in appropriate subdirectory

## Synthetic Data

For quick testing without downloading large datasets, many notebooks include code to generate synthetic data that mimics biological data characteristics.

## Questions?

If you have questions about data sources or preprocessing, please open an issue on GitHub.

---

**Note**: Large datasets are not included in this repository. Please download them separately and place in this directory.
