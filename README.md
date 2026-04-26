# NepScript Genesis - Neural Architecture Search for GAN

## Project Overview
The project focuses on generating synthetic Nepali/Devanagari handwritten digits using automatically discovered optimal GAN architectures through NAS. 

The project focuses on improving Random Neural Architecture Search for GAN architecture for synthetic Nepali/Devanagari handwritten digits.


##  Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd NepScript-Genesis-Framework

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup (Required if dataset is missing)
```bash
# Download the dataset from Kaggle:
# https://www.kaggle.com/datasets/anurags397/hindi-mnist-data/data

# Place hindi-mnist-data.zip in the data/ folder
# Extract the dataset:
python data/extraction_script.py
```

### 3. Run the Web App
```bash
# Open the Gradio web interface
python app.py  
```

## Supported NAS Strategies
- Random Search
- Progressive Search
- Adaptive Search
- Multi-fidelity Search
- Adversarial (Gradient-based) Search

## GAN Evaluation Metrics
- **Stability:** Measures training convergence
- **Balance:** Final generator/discriminator loss proximity
- **Quality:** Pixel intensity and variation
- **Diversity:** Average pairwise sample distance
- **Consistency:** Intra-class variance reliability
- **Efficiency:** Training and model compactness
- **Nepali Quality:** Sharpness and stroke integrity for Nepali digits

## 🚀 Training & NAS Search

NepScript Genesis provides multiple ways to discover and train optimal GAN architectures.

### ⚡ Batch Training (All Algorithms)
Use this script to run the full search and training pipeline for all supported strategies automatically:
```bash
chmod +x scripts/run_all_experiments.sh
./scripts/run_all_experiments.sh
```

### 🛠️ Individual Model Training
To train an architecture found by a specific strategy, follow this two-step process:

1. **Find Architecture**: `python scripts/nas_search.py --strategy <strategy> --epochs 4`
2. **Final Training**: `python scripts/train_model.py --arch-config <results_path>.json --epochs 400`

**Quick Reference Table:**

| Algorithm | Training Command (Step 2) |
|-----------|---------------------------|
| **Random** | `python scripts/train_model.py --arch-config experiments/gan_run_models_and_images/nas_results/best_architecture_random.json --epochs 400` |
| **Adaptive** | `python scripts/train_model.py --arch-config experiments/gan_run_models_and_images/nas_results/best_architecture_adaptive.json --epochs 400` |
| **Progressive** | `python scripts/train_model.py --arch-config experiments/gan_run_models_and_images/nas_results/best_architecture_progressive.json --epochs 400` |
| **Multi-Fidelity** | `python scripts/train_model.py --arch-config experiments/gan_run_models_and_images/nas_results/best_architecture_multifidelity.json --epochs 400` |
| **Adversarial** | `python scripts/train_model.py --arch-config experiments/gan_run_models_and_images/nas_results/best_architecture_adversarial.json --epochs 400` |

> [!TIP]
> Hardware (CUDA/MPS/CPU) is automatically detected. No manual `--device` flag is required.

### Evaluation & Generation
After training, evaluate your model or generate samples:
```bash
# Calculate FID, IS, Precision, and Recall
python scripts/fidscore.py --model path/to/best_generator.pth --config path/to/arch_config.json

# Generate synthetic images
python scripts/generate.py --model path/to/best_generator.pth --arch-config path/to/arch_config.json --num-samples 100 --grid
```

## 📖 Basic Script Usage

### Run NAS Search
```bash
python scripts/nas_search.py --config configs/nas_config.yaml --strategy <strategy> --epochs <num_epochs>
```

### Train GAN Model
```bash
python scripts/train_model.py --config configs/default_training.yaml --arch-config <config_path> --epochs <num_epochs>
```

### Evaluate GAN Model
```bash
python scripts/fidscore.py --model <generator_path> --config <config_path> --device <device>
```

## Documentation & Guides
- See `docs/algorithm_and_evaluation.md` for search strategy algorithms
- See `docs/model_evaluation_metric.md` for evaluation metrics
- See `docs/gan_evaluation_metric.md` for GAN specific metrics (FID, IS, etc.)
- See `docs/AdversarialNAS_Implementation_Guide.md` for algorithm details

## Project Structure
- `app.py`: Gradio web interface
- `scripts/`: Search, training, and evaluation scripts
- `nepscript/`: Core engine and model factory
- `configs/`: YAML configuration files
- `data/`: Dataset management
- `experiments/`: Results and generated samples

## ⚖️ License
This project is part of academic research. Please cite appropriately if you use it in your work.

---
