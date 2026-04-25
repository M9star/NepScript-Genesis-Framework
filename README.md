# NepScript Genesis - Neural Architecture Search for GAN

## Project Overview
The project focuses on generating synthetic Nepali/Devanagari handwritten digits using automatically discovered optimal GAN architectures through NAS. 

The project focuses on improving Random Neural Architecture Search for GAN architecture for synthetic Nepali/Devanagari handwritten digits.




##  Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd NepScript-Genesis

# Install dependencies
pip install -r requirements.txt

```

### 2. Data Setup   (Required only if the data set is deleted from the repo)
```bash
# Download the dataset from Kaggle:
https://www.kaggle.com/datasets/anurags397/hindi-mnist-data/data

# Place hindi-mnist-data.zip in the data/ folder
# Extract the dataset:
python data/extraction_script.py
```

### 3. Run NAS Experiments
```bash
# Open the main app.py ( web interface: used gradio)
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

## Example Scripts

### Run NAS Search
```bash
python scripts/nas_search.py --config configs/nas_config.yaml --strategy <strategy> --epochs <num_epochs>
```

### Train GAN Model
```bash
python scripts/train_model.py --config configs/default_training.yaml
```

### Evaluate GAN Model
```bash
python scripts/fidscore.py --gen_path <generator_path> --disc_path <discriminator_path> --arch_config <config_path>
```

## Documentation & Guides
- See `docs/algorithm_and_evaluation.md` for search strategy algorithms
- See `docs/model_evaluation_metric.md` for evaluation of model 
- See `docs/gan_evaluation_metric.md` to learn about GAN evaluation metrics (FID, IS, Precision, Recall)
- See `docs/AdversarialNAS_Implementation_Guide.md` for advanced strategy details

## Project Structure (Short Overview)
- app.py — Gradio web interface for GAN evaluation and visualization
- scripts — NAS search, training, and evaluation scripts
- nepscript — Core NAS and GAN implementation (models, training, utils)
- configs — YAML configuration files for experiments
- data — Dataset and extraction scripts
- docs — Documentation and guides
- experiments — Results, metrics, and generated images


##  License

This project is part of academic research. Please cite appropriately if you use it in your work.


---





