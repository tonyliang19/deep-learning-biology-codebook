#!/usr/bin/env python3
"""
Test script to validate the deep learning environment setup.
Run this to ensure all dependencies are properly installed.
"""

import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    tests = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("sklearn", "Scikit-learn"),
        ("Bio", "BioPython"),
        ("tqdm", "tqdm"),
    ]
    
    failed = []
    for module, name in tests:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)
    
    return len(failed) == 0

def test_torch():
    """Test PyTorch functionality."""
    print("\nTesting PyTorch...")
    
    try:
        import torch
        
        # Check version
        print(f"  PyTorch version: {torch.__version__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        
        # Test basic operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.matmul(x, y)
        print(f"  ✓ Basic tensor operations work")
        
        # Test GPU operations if available
        if cuda_available:
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            print(f"  ✓ GPU operations work")
        
        return True
    except Exception as e:
        print(f"  ✗ PyTorch test failed: {e}")
        return False

def test_notebook_structure():
    """Test if notebooks are properly structured."""
    print("\nTesting notebook structure...")
    
    import os
    import json
    
    notebook_dir = 'notebooks'
    if not os.path.exists(notebook_dir):
        print(f"  ✗ Notebook directory not found")
        return False
    
    notebooks = [f for f in os.listdir(notebook_dir) if f.endswith('.ipynb')]
    
    if len(notebooks) == 0:
        print(f"  ✗ No notebooks found")
        return False
    
    print(f"  Found {len(notebooks)} notebooks:")
    for nb in sorted(notebooks):
        path = os.path.join(notebook_dir, nb)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cells = len(data.get('cells', []))
                print(f"    ✓ {nb}: {cells} cells")
        except Exception as e:
            print(f"    ✗ {nb}: {e}")
            return False
    
    return True

def test_data_directories():
    """Test if data directories exist."""
    print("\nTesting data directories...")
    
    import os
    
    dirs = ['data/raw', 'data/processed']
    for d in dirs:
        if os.path.exists(d):
            print(f"  ✓ {d} exists")
        else:
            print(f"  ! {d} not found (will be created when needed)")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Deep Learning Biology Environment Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("PyTorch", test_torch),
        ("Notebook Structure", test_notebook_structure),
        ("Data Directories", test_data_directories),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! Environment is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
