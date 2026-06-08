"""
Cardiac VisionAI — Model Export
=================================
Merge LoRA adapters back into the base model and export.

Supported export formats:
  1. HuggingFace (full model)  — for HF Hub or local loading
  2. GGUF (quantized)          — for llama.cpp / Ollama
  3. vLLM-compatible           — for high-throughput serving

Usage:
  python export_model.py --format huggingface
  python export_model.py --format gguf --quant q4_k_m
  python export_model.py --format all
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
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    hardware, model_cfg, get_model_name,
    OUTPUT_DIR, EXPORT_DIR,
)


def export_huggingface(model, tokenizer, output_dir: Path):
    """Export as merged HuggingFace model."""
    print(f"\n[Export] Saving merged model to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method="merged_16bit",  # Full precision merged model
    )
    print(f"[Export] HuggingFace model saved to {output_dir}")


def export_gguf(model, tokenizer, output_dir: Path, quantization: str = "q4_k_m"):
    """Export as GGUF for llama.cpp / Ollama."""
    print(f"\n[Export] Saving GGUF ({quantization}) to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=quantization,
    )
    print(f"[Export] GGUF model saved to {output_dir}")


def export_lora_only(model, tokenizer, output_dir: Path):
    """Export just the LoRA adapters (smallest output)."""
    print(f"\n[Export] Saving LoRA adapters to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[Export] LoRA adapters saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Cardiac VisionAI — Model Export")
    parser.add_argument("--format", type=str, default="lora",
                        choices=["lora", "huggingface", "gguf", "all"],
                        help="Export format")
    parser.add_argument("--adapter-path", type=str,
                        default=str(OUTPUT_DIR / "lora_adapters"),
                        help="Path to trained LoRA adapters")
    parser.add_argument("--output-dir", type=str,
                        default=str(EXPORT_DIR),
                        help="Output directory for exported model")
    parser.add_argument("--quant", type=str, default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="GGUF quantization method")
    
    args = parser.parse_args()
    
    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)
    
    if not adapter_path.exists():
        print(f"[ERROR] LoRA adapters not found at {adapter_path}")
        print("[TIP] Train the model first: python train_vlm.py")
        sys.exit(1)
    
    # Load model with adapters
    if hardware.backend == "nvidia_unsloth":
        from unsloth import FastVisionModel
        
        print(f"[Model] Loading from {adapter_path}...")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=str(adapter_path),
            load_in_4bit=model_cfg.load_in_4bit,
        )
    else:
        raise NotImplementedError(f"Export not yet implemented for {hardware.backend}")
    
    # Export
    if args.format in ("lora", "all"):
        export_lora_only(model, tokenizer, output_dir / "lora")
    
    if args.format in ("huggingface", "all"):
        export_huggingface(model, tokenizer, output_dir / "huggingface")
    
    if args.format in ("gguf", "all"):
        export_gguf(model, tokenizer, output_dir / "gguf", args.quant)
    
    print(f"\n[Done] Export complete! Files in {output_dir}")


if __name__ == "__main__":
    main()
