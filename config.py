"""
Cardiac VisionAI — VLM Fine-Tuning Configuration
=================================================
Central configuration for the entire pipeline.
Change HARDWARE_BACKEND to switch between NVIDIA (Unsloth) and Intel (IPEX/OpenVINO).
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGES_DIR = PROCESSED_DATA_DIR / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXPORT_DIR = PROJECT_ROOT / "exported_model"

# Raw dataset paths (symlink or copy your Kaggle downloads here)
IMAGECAS_DIR = RAW_DATA_DIR / "imagecas"      # Contains case_id/img.nii.gz + label.nii.gz
HEART_CT_DIR = RAW_DATA_DIR / "heart_ct"      # Contains DICOM files or PNG slices

# Processed dataset output
DATASET_JSON = PROCESSED_DATA_DIR / "dataset.json"
TRAIN_DATASET_JSON = PROCESSED_DATA_DIR / "train_dataset.json"
VAL_DATASET_JSON = PROCESSED_DATA_DIR / "val_dataset.json"


@dataclass
class HardwareConfig:
    """
    Hardware backend configuration.
    
    Supported backends:
      - "nvidia_unsloth" : NVIDIA GPU + Unsloth (current, default)
      - "intel_ipex"     : Intel GPU/CPU + Intel Extension for PyTorch (future)
      - "intel_openvino" : Intel + OpenVINO for inference only (future)
    """
    backend: Literal["nvidia_unsloth", "intel_ipex", "intel_openvino"] = "nvidia_unsloth"
    
    # GPU memory constraint (GB) — used to auto-select model size and batch params
    vram_gb: float = 4.0
    
    # Device override (None = auto-detect)
    device: Optional[str] = None
    
    def get_device(self) -> str:
        if self.device:
            return self.device
        if self.backend == "nvidia_unsloth":
            return "cuda"
        elif self.backend.startswith("intel"):
            return "xpu"  # Intel GPU via IPEX
        return "cpu"


@dataclass
class ModelConfig:
    """
    Model selection based on VRAM constraints.
    
    VRAM Guide (4-bit quantized):
      - 2B model: ~2-3 GB  → fits in 4 GB VRAM ✅
      - 3B model: ~3-4 GB  → tight fit in 4 GB VRAM ⚠️
      - 7B model: ~6-8 GB  → needs 12+ GB VRAM ❌
    """
    # Model identifiers
    # For Unsloth (NVIDIA): use the bnb-4bit variant
    # For Intel IPEX: use the base HF model + manual quantization
    model_name_unsloth: str = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"
    model_name_base: str = "Qwen/Qwen2-VL-2B-Instruct"  # For Intel/non-Unsloth path
    
    # Quantization
    load_in_4bit: bool = True
    
    # Sequence length — shorter saves VRAM
    max_seq_length: int = 1024
    
    # LoRA / QLoRA configuration
    lora_r: int = 16                        # Rank — higher = more capacity, more VRAM
    lora_alpha: int = 16                    # Scaling factor, typically == r
    lora_dropout: float = 0.0              # Unsloth recommends 0
    lora_bias: str = "none"
    lora_target_modules: str = "all-linear"
    
    # Which parts of the VLM to fine-tune
    finetune_vision_layers: bool = True     # True for medical imaging (domain shift)
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    
    # Gradient checkpointing (critical for low VRAM)
    use_gradient_checkpointing: str = "unsloth"


@dataclass
class DataConfig:
    """Data processing configuration."""
    # Pilot size — how many images to use for initial validation
    pilot_size: int = 300
    
    # Image processing
    image_size: tuple = (384, 384)    # Resize target — 384px balances quality vs VRAM
    image_format: str = "png"
    
    # CT windowing (for cardiac CT angiography)
    ct_window_width: int = 400        # Hounsfield units
    ct_window_level: int = 40         # Hounsfield units (center)
    
    # Slice extraction from 3D volumes
    slices_per_volume: int = 3        # How many 2D slices to extract per 3D case
    slice_axes: list = field(default_factory=lambda: ["axial"])  # axial, coronal, sagittal
    
    # Train/val split
    val_ratio: float = 0.1
    
    # System instruction for the VLM
    system_instruction: str = (
        "You are an expert cardiac radiologist specializing in Coronary CT Angiography (CCTA) analysis. "
        "Analyze the provided cardiac CT image and describe clinically relevant findings including: "
        "coronary artery anatomy, any visible plaque (calcified or non-calcified), stenosis, "
        "and other cardiovascular abnormalities. Provide structured, actionable observations."
    )


@dataclass
class TrainingConfig:
    """Training hyperparameters optimized for 4GB VRAM."""
    # Batch size — must be 1 for 4GB VRAM
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8   # Effective batch = 1 × 8 = 8
    
    # Learning rate
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    
    # Schedule
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # Precision — auto-detect bf16 support
    fp16: bool = True     # Will be overridden if bf16 is supported
    bf16: bool = False    # Preferred if GPU supports it
    
    # Optimizer — 8-bit Adam saves ~30% VRAM
    optim: str = "adamw_8bit"
    
    # Logging
    logging_steps: int = 1
    save_steps: int = 50
    save_total_limit: int = 3
    
    # Output
    output_dir: str = str(OUTPUT_DIR)
    
    # Critical VLM flags
    remove_unused_columns: bool = False                                     # MUST be False for VLM
    dataset_kwargs: dict = field(default_factory=lambda: {
        "skip_prepare_dataset": True                                        # MUST be True for VLM
    })
    
    # Seed
    seed: int = 42
    
    # Max steps override (for quick smoke tests, set to e.g. 10)
    # Set to -1 to use num_train_epochs instead
    max_steps: int = -1


# ─────────────────────────────────────────────
#  Global config instances
# ─────────────────────────────────────────────
hardware = HardwareConfig()
model_cfg = ModelConfig()
data_cfg = DataConfig()
training_cfg = TrainingConfig()


def get_model_name() -> str:
    """Return the appropriate model name based on hardware backend."""
    if hardware.backend == "nvidia_unsloth":
        return model_cfg.model_name_unsloth
    else:
        return model_cfg.model_name_base


def ensure_dirs():
    """Create all required directories."""
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGES_DIR, 
              OUTPUT_DIR, EXPORT_DIR, IMAGECAS_DIR, HEART_CT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def print_config():
    """Print current configuration summary."""
    print("=" * 60)
    print("  Cardiac VisionAI — Configuration Summary")
    print("=" * 60)
    print(f"  Hardware Backend : {hardware.backend}")
    print(f"  VRAM Available   : {hardware.vram_gb} GB")
    print(f"  Device           : {hardware.get_device()}")
    print(f"  Model            : {get_model_name()}")
    print(f"  Quantization     : {'4-bit (QLoRA)' if model_cfg.load_in_4bit else 'FP16 (LoRA)'}")
    print(f"  LoRA Rank        : {model_cfg.lora_r}")
    print(f"  Max Seq Length   : {model_cfg.max_seq_length}")
    print(f"  Batch Size       : {training_cfg.per_device_train_batch_size} × {training_cfg.gradient_accumulation_steps} = {training_cfg.per_device_train_batch_size * training_cfg.gradient_accumulation_steps}")
    print(f"  Learning Rate    : {training_cfg.learning_rate}")
    print(f"  Epochs           : {training_cfg.num_train_epochs}")
    print(f"  Pilot Size       : {data_cfg.pilot_size} images")
    print(f"  Image Size       : {data_cfg.image_size}")
    print(f"  Output Dir       : {training_cfg.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
