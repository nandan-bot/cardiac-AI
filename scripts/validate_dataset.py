"""
Cardiac VisionAI — Dataset Validation
======================================
Validates that the processed dataset is correctly formatted for VLM fine-tuning.

Usage:
  python scripts/validate_dataset.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_JSON, TRAIN_DATASET_JSON, VAL_DATASET_JSON, IMAGES_DIR


def validate_sample(sample: dict, idx: int) -> list:
    """Validate a single training sample. Returns list of errors."""
    errors = []
    prefix = f"Sample {idx}"
    
    # Check top-level structure
    if "messages" not in sample:
        errors.append(f"{prefix}: Missing 'messages' key")
        return errors
    
    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append(f"{prefix}: 'messages' must be a list with at least 2 entries (user + assistant)")
        return errors
    
    # Check user message
    user_msg = messages[0]
    if user_msg.get("role") != "user":
        errors.append(f"{prefix}: First message role must be 'user', got '{user_msg.get('role')}'")
    
    user_content = user_msg.get("content", [])
    if not isinstance(user_content, list):
        errors.append(f"{prefix}: User content must be a list of content blocks")
    else:
        has_image = any(c.get("type") == "image" for c in user_content)
        has_text = any(c.get("type") == "text" for c in user_content)
        
        if not has_image:
            errors.append(f"{prefix}: User content missing image block")
        if not has_text:
            errors.append(f"{prefix}: User content missing text block")
        
        # Check image path exists
        for c in user_content:
            if c.get("type") == "image":
                img_path = c.get("image", "")
                if img_path and not Path(img_path).exists():
                    errors.append(f"{prefix}: Image file not found: {img_path}")
    
    # Check assistant message
    asst_msg = messages[1]
    if asst_msg.get("role") != "assistant":
        errors.append(f"{prefix}: Second message role must be 'assistant', got '{asst_msg.get('role')}'")
    
    asst_content = asst_msg.get("content", [])
    if not isinstance(asst_content, list):
        errors.append(f"{prefix}: Assistant content must be a list")
    else:
        has_text = any(c.get("type") == "text" for c in asst_content)
        if not has_text:
            errors.append(f"{prefix}: Assistant content missing text response")
        
        # Check text is non-trivial
        for c in asst_content:
            if c.get("type") == "text" and len(c.get("text", "")) < 20:
                errors.append(f"{prefix}: Assistant text too short ({len(c.get('text', ''))} chars)")
    
    return errors


def validate_dataset(filepath: Path) -> bool:
    """Validate an entire dataset file."""
    print(f"\nValidating: {filepath}")
    print("-" * 50)
    
    if not filepath.exists():
        print(f"  [FAIL] File not found: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"  ❌ Root must be a JSON array, got {type(data).__name__}")
        return False
    
    print(f"  Total samples: {len(data)}")
    
    all_errors = []
    role_counts = Counter()
    text_lengths = []
    image_count = 0
    
    for idx, sample in enumerate(data):
        errors = validate_sample(sample, idx)
        all_errors.extend(errors)
        
        # Collect stats
        for msg in sample.get("messages", []):
            role_counts[msg.get("role", "unknown")] += 1
            for c in msg.get("content", []):
                if c.get("type") == "text":
                    text_lengths.append(len(c.get("text", "")))
                if c.get("type") == "image":
                    image_count += 1
    
    # Print results
    if all_errors:
        print(f"\n  [FAIL] Found {len(all_errors)} errors:")
        for err in all_errors[:20]:  # Show first 20
            print(f"    - {err}")
        if len(all_errors) > 20:
            print(f"    ... and {len(all_errors) - 20} more")
    else:
        print(f"  [OK] All {len(data)} samples are valid!")
    
    # Stats
    print(f"\n  Statistics:")
    print(f"    Roles: {dict(role_counts)}")
    print(f"    Images: {image_count}")
    if text_lengths:
        print(f"    Text lengths: min={min(text_lengths)}, max={max(text_lengths)}, "
              f"avg={sum(text_lengths)//len(text_lengths)}")
    
    return len(all_errors) == 0


def main():
    print("=" * 50)
    print("  Cardiac VisionAI — Dataset Validation")
    print("=" * 50)
    
    all_valid = True
    
    for filepath in [DATASET_JSON, TRAIN_DATASET_JSON, VAL_DATASET_JSON]:
        if filepath.exists():
            valid = validate_dataset(filepath)
            all_valid = all_valid and valid
    
    # Check images directory
    print(f"\nImages directory: {IMAGES_DIR}")
    if IMAGES_DIR.exists():
        image_files = list(IMAGES_DIR.glob("*.*"))
        print(f"  Total image files: {len(image_files)}")
        
        # Check a few random images
        from PIL import Image
        for img_path in image_files[:3]:
            try:
                img = Image.open(img_path)
                print(f"  [OK] {img_path.name}: {img.size} {img.mode}")
            except Exception as e:
                print(f"  [FAIL] {img_path.name}: {e}")
    else:
        print("  ⚠️ Images directory does not exist yet")
    
    print(f"\n{'='*50}")
    if all_valid:
        print("  [OK] All datasets are valid and ready for training!")
    else:
        print("  [FAIL] Some issues found. Fix them before training.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
