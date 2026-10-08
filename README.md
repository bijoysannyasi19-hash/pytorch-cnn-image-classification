# Deep Learning Application in Data Science

This project builds a Convolutional Neural Network (CNN) from scratch in PyTorch to classify images. It compares an MLP, a Plain CNN, and an Improved CNN.

## Problem Statement
The goal is multi-class image classification. Given a 28x28 grayscale image, the model predicts one of 10 digits. Deep learning (CNNs) are ideal for this as they capture spatial features directly. (Note: I fell back to MNIST instead of CIFAR-10 because downloading CIFAR-10 was too slow on this machine).

## Dataset
- **Name**: MNIST
- **Source**: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- **Size**: 70,000 images (28x28)
- **Classes**: 10 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
- *Note: Downloads automatically via torchvision.*

## Key Results
- Improved CNN Test Accuracy: **0.9198**
- Mean Test Accuracy (3 seeds): **0.6932 ± 0.1641**

![Model Comparison](figures/model_comp.png)
![Confusion Matrix](figures/cm.png)

## ARCHITECTURE

### Pipeline Flowchart
```mermaid
flowchart TD
    A[Dataset Download] --> B[Train/Val/Test Split]
    B --> C[Normalization and Augmentation]
    C --> D[DataLoaders]
    D --> E[Model: MLP, Plain CNN, Improved CNN]
    E --> F[Training: loss, optimizer, scheduler, early stopping]
    F --> G[Validation and Tuning]
    G --> H[Final Test Evaluation]
    H --> I[Ablation and Error Analysis]
    I --> J[Report]
```

![Pipeline](docs/architecture.png)

### Improved CNN Network Diagram
![Improved CNN](docs/network_arch.png)
- **Conv 64, BN, ReLU**: Initial feature extraction and normalization.
- **ResB blocks**: Residual blocks with skip connections to allow gradient flow.
- **AdaptiveAvgPool**: Reduces spatial dimensions to 1x1, saving parameters.
- **Dropout**: Regularization to prevent overfitting.
- **Linear**: Final classification layer.

### Source Code
- `src/data.py`: Loads and transforms MNIST, creates DataLoaders. Data moves from `/data` to RAM.
- `src/nets.py`: Defines neural network PyTorch models.
- `src/train.py`: Contains the training and validation loop, saves to `/models` and `/results`.
- `src/evaluate.py`: Computes test metrics.
- `src/plots.py`: Generates all figures and saves to `/figures`.
- `src/run.py`: Master script to run everything.
- `src/report_gen.py`: Generates the Word document into `/report`.

## Project Structure
```
/data                # dataset 
/notebooks           # jupyter notebooks
/src                 # source code
/figures             # output plots
/results             # output metrics and csv files
/models              # saved weights
/report              # generated DOCX
/docs                # documentation images
README.md            # this file
requirements.txt     # pip requirements
.gitignore           # git ignore file
LICENSE              # MIT license
```

## Installation
1. Requires Python 3.
2. Create a virtual environment: `python -m venv venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`.
4. *Note: Training will automatically use GPU if available, else CPU.*

## How to Run
Run the full pipeline (takes ~15-30 mins on CPU):
```bash
python src/run.py
python src/report_gen.py
```
To open the notebook:
```bash
jupyter notebook notebooks/deep_learning.ipynb
```

## Methodology Summary
- **Preprocessing**: Mean/std normalization (from train set), random crop and flip.
- **Models**: Explored MLP, Plain CNN, and Improved CNN (with BN and skip).
- **Training**: AdamW optimizer, CrossEntropyLoss, plateau scheduler, early stopping.
- **Evaluation**: Accuracy, Macro F1, Per-class metrics, multiple seeds.
- **Ablation**: Removed augmentation, BN, skip connections to measure impact.

## Strengths, Limitations and Future Work
- **Strengths**: Robust evaluation across seeds, thorough ablation, clear insights.
- **Limitations**: Model is small and trained for few epochs.
- **Future Work**: Use larger ResNet architectures, train longer, use GPU, apply MixUp.

## Credits
- Dataset: MNIST (Yann LeCun, Corinna Cortes, Christopher J.C. Burges)
- License: MIT
- Author: [Your Name]
