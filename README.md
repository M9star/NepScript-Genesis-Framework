# NepScript Genesis - Neural Architecture Search for GAN

## Project Overview
The project focuses on generating synthetic Nepali/Devanagari handwritten digits using automatically discovered optimal GAN architectures through NAS. 

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
python data/extraction_script.py
```

> **Dataset:** Devanagari Handwritten Character Dataset (DHCD) — 20,000 digit
> samples across 10 classes (0–9), 2,000 per class, resized to 32×32 and
> normalized to [-1, 1]. The raw images are **not committed** to the repo.
> After extraction, the configs expect:
> - images under `data/DevanagariHandwrittenDigitDataset/`
> - labels at `data/hindi_mnist.csv`
>
> The committed `data/hindi_mnist_with_synthetic_10k.csv` and
> `data/synthetic_digits_10k/` are the GAN-augmented set used for the
> downstream CNN experiments, not the base training data.

### 3. Run the Web App
```bash
# Open the Gradio web interface
python app.py  
```
### Download Pre-trained Models
All pre-trained models are available here:

**[ Download All Models from Google Drive](https://drive.google.com/drive/folders/1YbGtc-QbDNzC4F6L0swKh4URSQhPwa1i?usp=sharing)**

The folder contains `.pth` generator files and `.json` architecture configs for:
- Random Search
- Adaptive Search
- Progressive Search
- Multi-Fidelity Search
- Adversarial Search
- Manual DCGAN Baseline

### Setup Downloaded Models
```bash
# Create models directory
mkdir -p models/

# After downloading, place files in models/ folder:
# models/
# ├── best_generator_random.pth
# ├── best_architecture_random.json
# ├── best_generator_adaptive.pth
# ├── best_architecture_adaptive.json
# └── ... (other models)
```


## Supported NAS Strategies
-  **Random Search** - random architecture sampling
-  **Adaptive Search** - Learns from successful architectures
- **Progressive Search** - Phased coarse-to-fine search
-  **Multi-Fidelity Search** - Efficient 3-stage screening
- **Adversarial (Gradient-based)** - Direct architecture parameter optimization

Model tha is compare with 
- **Manual DCGAN** - Radford et al. (2015) baseline



### Run NAS Search
```bash
python scripts/nas_search.py --strategy adaptive --num-archs <num_arch_search> --epochs <num_epoch>

for adverserial: 
python scripts/nas_search.py --strategy adversarial --epochs <num_epoch> 

```

### Train GAN Model
```bash
python scripts/train_model.py --strategy <strategy_name> --epochs <num_epoch>

```

- use manual_dcgan for baseline manual training
- we enable augmentation while training


### Evaluation & Generation
After training, evaluate your model or generate samples:
```bash
# Calculate FID, IS, Precision, and Recall
python scripts/evaluate_model_scores.py --strategy <strategy_name>

# Generate synthetic images
python scripts/generate.py --model path/to/best_generator.pth --arch-config path/to/arch_config.json --num-samples 100 --grid
```


## ⚖️ License
This project is part of academic research. Please cite appropriately if you use it in your work.

---
