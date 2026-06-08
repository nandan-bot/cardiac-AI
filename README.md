# Cardiac VisionAI — VLM Fine-Tuning Pipeline

Fine-tune a Vision Language Model (Qwen2-VL-2B) on cardiac CT data for automated CCTA analysis. 

## Overview

This pipeline fine-tunes a **Qwen2-VL-2B** model using **QLoRA** (4-bit quantization + LoRA adapters) to analyze Coronary CT Angiography scans. The model learns to identify coronary artery anatomy, plaque, stenosis, and generate structured clinical reports.

### Architecture

```
Raw Data (NIfTI/DICOM) → Data Pipeline → VLM JSON Dataset → QLoRA Training → Fine-tuned Model → Inference
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install base dependencies
pip install -r requirements.txt

# Install Unsloth (NVIDIA GPU required for training)
pip install unsloth
```

### 2. Prepare Data

```bash
# Option A: Generate synthetic demo data (no datasets needed)
python scripts/prepare_data.py --demo

# Option B: Process real datasets (download from Kaggle first)
#   - ImageCAS: https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas
#   - Heart CT: https://www.kaggle.com/datasets/abbymorgan/heart-ct
# Place in data/raw/imagecas/ and data/raw/heart_ct/
python scripts/prepare_data.py

# Validate the dataset
python scripts/validate_dataset.py
```

### 3. Train

```bash
# Smoke test (10 steps, verifies everything works)
python train_vlm.py --smoke-test

# Full training
python train_vlm.py

# Custom step count
python train_vlm.py --max-steps 100
```

### 4. Inference

```bash
# Run on a demo image
python inference.py --demo

# Run on a specific image
python inference.py --image path/to/cardiac_ct.png

# Custom question
python inference.py --image scan.png --question "Is there coronary calcification?"
```

### 5. Export

```bash
# Export LoRA adapters only (small, fast)
python export_model.py --format lora

# Export merged HuggingFace model
python export_model.py --format huggingface

# Export GGUF for Ollama/llama.cpp
python export_model.py --format gguf --quant q4_k_m
```

## Hardware Requirements

| Hardware | Status | Notes |
|----------|--------|-------|
| **NVIDIA RTX 3050 Ti (4GB)** | ✅ Current | Uses Qwen2-VL-2B + QLoRA, batch_size=1 |
| **NVIDIA RTX 3060+ (12GB)** | ✅ Better | Can use Qwen2-VL-7B, larger batch |
| **Intel Arc / Gaudi** | 🔮 Planned | Change `backend` in config.py |
| **CPU only** | ⚠️ Slow | Works for inference, not practical for training |

## Project Structure

```
cardiac-vlm-finetune/
├── config.py                    # All hyperparameters and paths
├── train_vlm.py                 # Main training script
├── inference.py                 # Run model on new images
├── export_model.py              # Export/merge model
├── requirements.txt             # Python dependencies
├── scripts/
│   ├── prepare_data.py          # Data conversion pipeline
│   └── validate_dataset.py      # Dataset sanity checks
├── data/
│   ├── raw/                     # Place Kaggle downloads here
│   │   ├── imagecas/
│   │   └── heart_ct/
│   └── processed/               # Generated: JSON + PNGs
│       ├── images/
│       ├── dataset.json
│       ├── train_dataset.json
│       └── val_dataset.json
├── outputs/                     # Training checkpoints
│   └── lora_adapters/
└── exported_model/              # Final exported model
```

## Configuration

All settings are in `config.py`. Key parameters:

```python
# Switch hardware backend (for Intel VM migration)
hardware.backend = "nvidia_unsloth"  # or "intel_ipex" in the future

# Adjust for your VRAM
hardware.vram_gb = 4.0
model_cfg.max_seq_length = 1024       # Reduce to 512 if OOM
data_cfg.image_size = (384, 384)      # Reduce to (256, 256) if OOM
model_cfg.lora_r = 16                 # Reduce to 8 if OOM
```

## Data Format

Each training sample follows this JSON structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "path/to/slice.png"},
        {"type": "text", "text": "Analyze this cardiac CT..."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "This axial CCTA slice shows..."}
      ]
    }
  ]
}
```

## Switching to Intel VM

When your Intel VM is ready:

1. Install Intel Extension for PyTorch: `pip install intel-extension-for-pytorch`
2. In `config.py`, change: `hardware.backend = "intel_ipex"`
3. Implement the `load_model_intel()` function in `train_vlm.py`
4. The data pipeline, dataset format, and inference API remain identical

## Datasets

| Dataset | Source | Format | Content |
|---------|--------|--------|---------|
| ImageCAS | [Kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas) | NIfTI (.nii.gz) | 1000 3D CTA volumes + coronary artery masks |
| Heart CT | [Kaggle](https://www.kaggle.com/datasets/abbymorgan/heart-ct) | DICOM | Sequential cardiac CT slices |

## License

Internal use — Aventyn Confidential.
