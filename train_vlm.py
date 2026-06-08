"""
Cardiac VisionAI — VLM Fine-Tuning Training Script
====================================================
Fine-tunes a Vision Language Model on cardiac CT data using QLoRA.

Hardware-aware: automatically adjusts for your GPU.
Backend-aware: supports NVIDIA (Unsloth) now, Intel (IPEX) in the future.

Usage:
  python train_vlm.py                    # Full training
  python train_vlm.py --smoke-test       # Quick 10-step validation
  python train_vlm.py --max-steps 50     # Custom step count
  python train_vlm.py --resume outputs   # Resume from checkpoint

Prerequisites:
  1. pip install -r requirements.txt
  2. pip install unsloth
  3. python scripts/prepare_data.py --demo   (or with real data)
"""
import os
# Disable Triton compilation / Dynamo to avoid PyTorch 2.5 + Triton Windows compatibility crashes
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch

# Run comprehensive compatibility patch for low-bit integers in PyTorch < 2.6
for bits in range(1, 8):
    for prefix in ('int', 'uint'):
        attr = f"{prefix}{bits}"
        if not hasattr(torch, attr):
            fallback = 'bool' if bits == 1 else ('int8' if prefix == 'int' else 'uint8')
            setattr(torch, attr, getattr(torch, fallback))

# Windows DLL loading workarounds: import bitsandbytes and peft early
import bitsandbytes
import peft

import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

# Suppress noisy warnings during import
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*")

from config import (
    hardware, model_cfg, data_cfg, training_cfg,
    get_model_name, print_config, ensure_dirs,
    TRAIN_DATASET_JSON, VAL_DATASET_JSON, DATASET_JSON,
    OUTPUT_DIR,
)


# ─────────────────────────────────────────────────────────────
#  Backend-Specific Model Loading
# ─────────────────────────────────────────────────────────────

def load_model_nvidia():
    """
    Load model using Unsloth (NVIDIA GPU path).
    Returns (model, tokenizer).
    """
    from unsloth import FastVisionModel
    import torch
    
    print(f"\n[Model] Loading {get_model_name()} via Unsloth...")
    print(f"[Model] 4-bit quantization: {model_cfg.load_in_4bit}")
    print(f"[Model] Max sequence length: {model_cfg.max_seq_length}")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=get_model_name(),
        load_in_4bit=model_cfg.load_in_4bit,
        use_gradient_checkpointing=model_cfg.use_gradient_checkpointing,
        max_seq_length=model_cfg.max_seq_length,
    )
    
    # Print VRAM usage after loading
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
    
    # Configure LoRA
    print(f"\n[LoRA] Configuring adapters (rank={model_cfg.lora_r}, alpha={model_cfg.lora_alpha})...")
    print(f"[LoRA] Vision layers: {model_cfg.finetune_vision_layers}")
    print(f"[LoRA] Language layers: {model_cfg.finetune_language_layers}")
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=model_cfg.finetune_vision_layers,
        finetune_language_layers=model_cfg.finetune_language_layers,
        finetune_attention_modules=model_cfg.finetune_attention_modules,
        finetune_mlp_modules=model_cfg.finetune_mlp_modules,
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        bias=model_cfg.lora_bias,
        random_state=training_cfg.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"[VRAM] After LoRA: {allocated:.2f} GB")
    
    return model, tokenizer


def load_model_intel():
    """
    Load model using Intel Extension for PyTorch (future Intel VM path).
    
    TODO: Implement when Intel VM is available.
    This is a placeholder that shows the architecture for Intel support.
    """
    raise NotImplementedError(
        "Intel IPEX backend is not yet implemented.\n"
        "To use this backend, you will need:\n"
        "  1. pip install intel-extension-for-pytorch\n"
        "  2. pip install optimum[openvino]\n"
        "  3. Update this function with IPEX model loading\n\n"
        "For now, use hardware.backend = 'nvidia_unsloth' in config.py\n\n"
        "Reference implementation pattern:\n"
        "  import intel_extension_for_pytorch as ipex\n"
        "  from transformers import AutoModelForCausalLM, AutoProcessor\n"
        "  model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name_base)\n"
        "  model = ipex.optimize(model, dtype=torch.bfloat16)\n"
    )


def load_model():
    """Load model using the configured backend."""
    if hardware.backend == "nvidia_unsloth":
        return load_model_nvidia()
    elif hardware.backend.startswith("intel"):
        return load_model_intel()
    else:
        raise ValueError(f"Unknown backend: {hardware.backend}")


# ─────────────────────────────────────────────────────────────
#  Dataset Loading
# ─────────────────────────────────────────────────────────────

def load_dataset_from_json(filepath: Path) -> List[Dict[str, Any]]:
    """Load and convert dataset from our JSON format to Unsloth-compatible format."""
    from PIL import Image
    
    print(f"\n[Data] Loading dataset from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Convert image paths to PIL Image objects
    converted = []
    skipped = 0
    
    for sample in raw_data:
        try:
            messages = sample["messages"]
            new_messages = []
            
            for msg in messages:
                new_content = []
                for block in msg["content"]:
                    if block["type"] == "image":
                        img_path = block["image"]
                        if Path(img_path).exists():
                            pil_image = Image.open(img_path).convert("RGB")
                            # Resize if needed (saves VRAM during training)
                            if pil_image.size != data_cfg.image_size:
                                pil_image = pil_image.resize(
                                    data_cfg.image_size, Image.LANCZOS
                                )
                            new_content.append({
                                "type": "image",
                                "image": pil_image
                            })
                        else:
                            raise FileNotFoundError(f"Image not found: {img_path}")
                    else:
                        new_content.append(block)
                
                new_messages.append({
                    "role": msg["role"],
                    "content": new_content
                })
            
            converted.append({"messages": new_messages})
            
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  [WARNING] Skipped sample: {e}")
    
    print(f"[Data] Loaded {len(converted)} samples ({skipped} skipped)")
    return converted


# ─────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────

def create_trainer(model, tokenizer, train_dataset, val_dataset=None, 
                   max_steps: int = -1):
    """Create the SFT trainer with proper VLM configuration."""
    import torch
    from trl import SFTTrainer, SFTConfig
    from unsloth.trainer import UnslothVisionDataCollator
    
    # Auto-detect bf16 support
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    
    print(f"\n[Trainer] Creating SFT trainer...")
    print(f"[Trainer] Precision: {'bf16' if use_bf16 else 'fp16'}")
    print(f"[Trainer] Batch: {training_cfg.per_device_train_batch_size} × "
          f"{training_cfg.gradient_accumulation_steps} = "
          f"{training_cfg.per_device_train_batch_size * training_cfg.gradient_accumulation_steps}")
    print(f"[Trainer] LR: {training_cfg.learning_rate}")
    
    # Determine steps vs epochs
    effective_max_steps = max_steps if max_steps > 0 else training_cfg.max_steps
    num_epochs = training_cfg.num_train_epochs if effective_max_steps <= 0 else 1
    
    sft_config = SFTConfig(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_ratio=training_cfg.warmup_ratio,
        num_train_epochs=num_epochs,
        max_steps=effective_max_steps if effective_max_steps > 0 else -1,
        learning_rate=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=training_cfg.logging_steps,
        save_steps=training_cfg.save_steps,
        save_total_limit=training_cfg.save_total_limit,
        optim=training_cfg.optim,
        seed=training_cfg.seed,
        output_dir=training_cfg.output_dir,
        remove_unused_columns=False,              # CRITICAL for VLM
        dataset_kwargs={"skip_prepare_dataset": True},  # CRITICAL for VLM
        report_to="none",  # Change to "wandb" if using W&B
    )
    
    # Create vision data collator
    data_collator = UnslothVisionDataCollator(model, tokenizer)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        args=sft_config,
    )
    
    return trainer


def train(args):
    """Main training function."""
    import torch
    
    print_config()
    ensure_dirs()
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n[GPU] {gpu_name} — {gpu_mem:.1f} GB total VRAM")
        
        if gpu_mem < 3.5:
            print("[WARNING] Very low VRAM detected. Training may fail.")
            print("[TIP] Try reducing image_size in config.py or use --smoke-test first.")
    else:
        print("\n[WARNING] No CUDA GPU detected! Training will be very slow on CPU.")
    
    # Load model
    model, tokenizer = load_model()
    
    # Load dataset
    dataset_path = TRAIN_DATASET_JSON if TRAIN_DATASET_JSON.exists() else DATASET_JSON
    if not dataset_path.exists():
        print(f"\n[ERROR] No dataset found at {dataset_path}")
        print("[TIP] Run: python scripts/prepare_data.py --demo")
        sys.exit(1)
    
    train_dataset = load_dataset_from_json(dataset_path)
    
    val_dataset = None
    if VAL_DATASET_JSON.exists():
        val_dataset = load_dataset_from_json(VAL_DATASET_JSON)
    
    if not train_dataset:
        print("[ERROR] Training dataset is empty!")
        sys.exit(1)
    
    # Determine max_steps
    max_steps = -1
    if args.smoke_test:
        max_steps = 10
        print("\n[SMOKE TEST] Running for 10 steps only...")
    elif args.max_steps > 0:
        max_steps = args.max_steps
    
    # Create trainer
    trainer = create_trainer(
        model, tokenizer, train_dataset, val_dataset,
        max_steps=max_steps
    )
    
    # Print VRAM before training
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n[VRAM] Pre-training — Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
    
    # Train!
    print(f"\n{'='*50}")
    print(f"  Starting training...")
    print(f"  Dataset: {len(train_dataset)} samples")
    if max_steps > 0:
        print(f"  Max steps: {max_steps}")
    else:
        print(f"  Epochs: {training_cfg.num_train_epochs}")
    print(f"{'='*50}\n")
    
    try:
        trainer_stats = trainer.train(
            resume_from_checkpoint=args.resume if args.resume else None
        )
        
        print(f"\n{'='*50}")
        print(f"  Training Complete!")
        print(f"{'='*50}")
        print(f"  Total steps: {trainer_stats.global_step}")
        print(f"  Training loss: {trainer_stats.training_loss:.4f}")
        print(f"  Runtime: {trainer_stats.metrics.get('train_runtime', 0):.1f}s")
        
    except torch.cuda.OutOfMemoryError:
        print("\n[ERROR] CUDA Out of Memory!")
        print("Try these fixes in config.py:")
        print("  1. Reduce image_size to (256, 256)")
        print("  2. Reduce max_seq_length to 512")
        print("  3. Reduce lora_r to 8")
        print("  4. Set finetune_vision_layers = False")
        sys.exit(1)
    
    # Save model
    print(f"\n[Save] Saving LoRA adapters to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR / "lora_adapters"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "lora_adapters"))
    print(f"[Save] Done! Adapters saved to {OUTPUT_DIR / 'lora_adapters'}")
    
    # Print final VRAM
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n[VRAM] Peak memory usage: {peak:.2f} GB")
    
    return trainer_stats


# ─────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cardiac VisionAI — VLM Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_vlm.py --smoke-test          # Quick 10-step test
  python train_vlm.py --max-steps 50        # Run 50 steps
  python train_vlm.py                       # Full training (3 epochs)
  python train_vlm.py --resume outputs      # Resume from checkpoint
        """
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick 10-step smoke test")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Override max training steps (-1 = use epochs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint directory")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
