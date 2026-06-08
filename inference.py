"""
Cardiac VisionAI — Inference Script
=====================================
Run the fine-tuned VLM on new cardiac CT images.

Usage:
  python inference.py --image path/to/cardiac_ct.png
  python inference.py --image path/to/image.png --question "What do you see?"
  python inference.py --demo   # Run on a demo image from the dataset
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
    hardware, model_cfg, data_cfg,
    get_model_name, OUTPUT_DIR, IMAGES_DIR,
)


def load_model_for_inference(adapter_path: str = None):
    """Load model with optional LoRA adapters for inference."""
    import torch
    
    if hardware.backend == "nvidia_unsloth":
        from unsloth import FastVisionModel
        
        if adapter_path and Path(adapter_path).exists():
            print(f"[Model] Loading fine-tuned model from {adapter_path}...")
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=adapter_path,
                load_in_4bit=model_cfg.load_in_4bit,
            )
        else:
            print(f"[Model] Loading base model {get_model_name()}...")
            print("[INFO] No fine-tuned adapters found. Using base model.")
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=get_model_name(),
                load_in_4bit=model_cfg.load_in_4bit,
            )
        
        # Set to inference mode
        FastVisionModel.for_inference(model)
        
    else:
        raise NotImplementedError(f"Inference not yet implemented for {hardware.backend}")
    
    return model, tokenizer


def run_inference(model, tokenizer, image_path: str, question: str = None,
                  max_tokens: int = 512, temperature: float = 0.7):
    """Run inference on a single image."""
    import torch
    from PIL import Image
    from transformers import TextStreamer
    
    # Load and prepare image
    img = Image.open(image_path).convert("RGB")
    if img.size != data_cfg.image_size:
        img = img.resize(data_cfg.image_size, Image.LANCZOS)
    
    # Default question
    if not question:
        question = data_cfg.system_instruction
    
    # Prepare input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    
    inputs = tokenizer(
        img,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    
    print(f"\n{'─'*50}")
    print(f"  Image: {Path(image_path).name}")
    print(f"  Question: {question[:80]}...")
    print(f"{'─'*50}")
    print("\n  Model Response:\n")
    
    # Generate with streaming
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=max_tokens,
            use_cache=True,
            temperature=temperature,
            min_p=0.1,
        )
    
    # Decode full response (streamer already printed it)
    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    print(f"\n{'─'*50}")
    
    return response


def main():
    parser = argparse.ArgumentParser(
        description="Cardiac VisionAI — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, 
                        help="Path to cardiac CT image (PNG/JPG)")
    parser.add_argument("--question", type=str, default=None,
                        help="Custom question about the image")
    parser.add_argument("--adapter-path", type=str, 
                        default=str(OUTPUT_DIR / "lora_adapters"),
                        help="Path to LoRA adapters directory")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--demo", action="store_true",
                        help="Run on a random demo image from the dataset")
    
    args = parser.parse_args()
    
    # Find image
    if args.demo:
        if IMAGES_DIR.exists():
            images = list(IMAGES_DIR.glob("*.png"))
            if images:
                import random
                args.image = str(random.choice(images))
                print(f"[Demo] Using random image: {args.image}")
            else:
                print("[ERROR] No images found. Run: python scripts/prepare_data.py --demo")
                sys.exit(1)
        else:
            print("[ERROR] Images directory not found. Run: python scripts/prepare_data.py --demo")
            sys.exit(1)
    
    if not args.image:
        parser.print_help()
        print("\n[ERROR] Provide --image path or use --demo flag")
        sys.exit(1)
    
    if not Path(args.image).exists():
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)
    
    # Load model
    model, tokenizer = load_model_for_inference(args.adapter_path)
    
    # Run inference
    response = run_inference(
        model, tokenizer,
        image_path=args.image,
        question=args.question,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    return response


if __name__ == "__main__":
    main()
