"""
Cardiac VisionAI — Data Preparation Pipeline
=============================================
Converts raw medical imaging datasets into VLM fine-tuning format.

Supports:
  1. ImageCAS (NIfTI .nii.gz) — 3D CTA volumes → 2D PNG slices
  2. Heart CT (DICOM / PNG)   — DICOM series → 2D PNG slices
  3. Demo mode                — Creates synthetic samples if no raw data exists

Output format (per sample):
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "image": "path/to/slice.png"},
      {"type": "text",  "text": "Analyze this cardiac CT image..."}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "This axial CCTA slice shows..."}
    ]}
  ]
}

Usage:
  python scripts/prepare_data.py
  python scripts/prepare_data.py --demo    # Generate synthetic demo data
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    data_cfg, IMAGECAS_DIR, HEART_CT_DIR,
    IMAGES_DIR, PROCESSED_DATA_DIR, DATASET_JSON,
    TRAIN_DATASET_JSON, VAL_DATASET_JSON,
)
from config import ensure_dirs


# ─────────────────────────────────────────────────────────────
#  CT Windowing
# ─────────────────────────────────────────────────────────────

def apply_ct_window(image_array: np.ndarray, 
                    window_width: int = 400, 
                    window_level: int = 40) -> np.ndarray:
    """
    Apply CT windowing to convert Hounsfield Units to display range [0, 255].
    
    For cardiac CT (CCTA):
      - Soft tissue:  W=400, L=40
      - Lung:         W=1500, L=-600
      - Bone:         W=2000, L=300
      - Mediastinum:  W=350, L=50
    """
    lower = window_level - window_width // 2
    upper = window_level + window_width // 2
    
    windowed = np.clip(image_array, lower, upper)
    windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
    return windowed


# ─────────────────────────────────────────────────────────────
#  ImageCAS Processing (NIfTI)
# ─────────────────────────────────────────────────────────────

def process_imagecas(max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Process ImageCAS NIfTI volumes into 2D PNG slices + text pairs.
    
    Each case folder contains:
      - img.nii.gz   : 3D CTA volume
      - label.nii.gz : 3D segmentation mask (coronary arteries)
    
    Returns list of dataset samples.
    """
    try:
        import nibabel as nib
    except ImportError:
        print("[WARNING] nibabel not installed. Run: pip install nibabel")
        print("[INFO] Skipping ImageCAS processing.")
        return []
    
    if not IMAGECAS_DIR.exists():
        print(f"[INFO] ImageCAS directory not found at {IMAGECAS_DIR}")
        print("[INFO] Download from: https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas")
        return []
    
    case_dirs = sorted([d for d in IMAGECAS_DIR.iterdir() if d.is_dir()])
    if max_cases:
        case_dirs = case_dirs[:max_cases]
    
    if not case_dirs:
        print(f"[INFO] No case directories found in {IMAGECAS_DIR}")
        return []
    
    samples = []
    print(f"\n[ImageCAS] Processing {len(case_dirs)} cases...")
    
    for case_dir in tqdm(case_dirs, desc="ImageCAS cases"):
        img_path = case_dir / "img.nii.gz"
        label_path = case_dir / "label.nii.gz"
        
        if not img_path.exists():
            continue
        
        try:
            # Load 3D volume
            nii_img = nib.load(str(img_path))
            volume = nii_img.get_fdata()
            
            # Load segmentation mask if available
            has_mask = label_path.exists()
            if has_mask:
                mask_vol = nib.load(str(label_path)).get_fdata()
            
            # Extract slices from the volume
            n_slices = volume.shape[2]  # axial slices along z-axis
            
            # Select representative slices (middle region where heart is)
            center = n_slices // 2
            spread = n_slices // 6
            slice_indices = np.linspace(
                max(0, center - spread),
                min(n_slices - 1, center + spread),
                data_cfg.slices_per_volume,
                dtype=int
            )
            
            for idx in slice_indices:
                slice_2d = volume[:, :, idx]
                
                # Apply cardiac CT windowing
                slice_windowed = apply_ct_window(
                    slice_2d,
                    data_cfg.ct_window_width,
                    data_cfg.ct_window_level
                )
                
                # Resize
                img = Image.fromarray(slice_windowed, mode='L').convert('RGB')
                img = img.resize(data_cfg.image_size, Image.LANCZOS)
                
                # Save PNG
                case_id = case_dir.name
                filename = f"imagecas_{case_id}_slice{idx:04d}.{data_cfg.image_format}"
                filepath = IMAGES_DIR / filename
                img.save(filepath)
                
                # Generate clinical description from mask
                description = _generate_imagecas_description(
                    case_id, idx, n_slices,
                    mask_vol[:, :, idx] if has_mask else None
                )
                
                # Create VLM training sample
                sample = _create_sample(
                    image_path=str(filepath),
                    question=_get_clinical_question("ccta_analysis"),
                    answer=description
                )
                samples.append(sample)
                
        except Exception as e:
            print(f"[WARNING] Failed to process {case_dir.name}: {e}")
            continue
    
    print(f"[ImageCAS] Generated {len(samples)} samples")
    return samples


def _generate_imagecas_description(case_id: str, slice_idx: int, 
                                     total_slices: int,
                                     mask_slice: Optional[np.ndarray] = None) -> str:
    """
    Generate a clinical description for a CTA slice.
    Uses segmentation mask metadata when available.
    """
    # Determine anatomical level
    relative_pos = slice_idx / total_slices
    if relative_pos < 0.3:
        level = "inferior cardiac level, near the diaphragm"
    elif relative_pos < 0.5:
        level = "mid-cardiac level, at the level of the coronary arteries"
    elif relative_pos < 0.7:
        level = "superior cardiac level, near the aortic arch"
    else:
        level = "supra-cardiac level"
    
    parts = [
        f"This axial CCTA slice (case {case_id}, slice {slice_idx}/{total_slices}) "
        f"is obtained at the {level}."
    ]
    
    if mask_slice is not None:
        artery_pixels = np.sum(mask_slice > 0)
        total_pixels = mask_slice.size
        artery_fraction = artery_pixels / total_pixels if total_pixels > 0 else 0
        
        if artery_fraction > 0.01:
            parts.append(
                f"Coronary artery segments are visible in this slice, "
                f"occupying approximately {artery_fraction*100:.1f}% of the cross-section. "
                f"The coronary arteries appear patent at this level."
            )
        elif artery_fraction > 0:
            parts.append(
                "Minimal coronary artery cross-sections are visible at this level. "
                "Small-caliber vessels or distal branches may be present."
            )
        else:
            parts.append(
                "No major coronary artery segments are identified at this particular slice level. "
                "The cardiac chambers and mediastinal structures are visualized."
            )
    else:
        # No mask — provide generic description
        descriptions = [
            "The cardiac silhouette and mediastinal structures are visualized. "
            "Assessment of coronary artery calcification and luminal patency is recommended.",
            
            "The cardiac chambers are within normal size limits at this level. "
            "Coronary artery evaluation requires correlation with adjacent slices.",
            
            "The great vessels and cardiac structures are visualized. "
            "No gross abnormality is identified at this slice level, though comprehensive "
            "review of the full volume is necessary for definitive assessment.",
        ]
        parts.append(random.choice(descriptions))
    
    parts.append(
        "Note: This is an automated preliminary description. "
        "All findings should be verified by a qualified radiologist and correlated "
        "with the patient's clinical history."
    )
    
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────
#  Heart CT Processing (DICOM / PNG)
# ─────────────────────────────────────────────────────────────

def process_heart_ct(max_images: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Process Heart CT dataset (DICOM or PNG images).
    
    Handles two common structures:
      1. Flat DICOM directory
      2. train/images + train/masks structure
    """
    if not HEART_CT_DIR.exists():
        print(f"[INFO] Heart CT directory not found at {HEART_CT_DIR}")
        print("[INFO] Download from: https://www.kaggle.com/datasets/abbymorgan/heart-ct")
        return []
    
    samples = []
    
    # Try structure 1: train/images + train/masks
    train_images = HEART_CT_DIR / "train" / "images"
    if train_images.exists():
        samples.extend(_process_heart_ct_segmentation(train_images, max_images))
    
    # Try structure 2: flat DICOM directory
    dicom_files = list(HEART_CT_DIR.glob("**/*.dcm"))
    if dicom_files:
        samples.extend(_process_heart_ct_dicom(dicom_files, max_images))
    
    # Try structure 3: any PNG/JPG images in the directory
    if not samples:
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_files.extend(HEART_CT_DIR.glob(f"**/{ext}"))
        if image_files:
            samples.extend(_process_generic_images(image_files, max_images))
    
    print(f"[Heart CT] Generated {len(samples)} samples")
    return samples


def _process_heart_ct_segmentation(images_dir: Path, 
                                    max_images: Optional[int]) -> List[Dict[str, Any]]:
    """Process heart CT with segmentation mask structure."""
    image_files = sorted(list(images_dir.glob("*.*")))
    if max_images:
        image_files = image_files[:max_images]
    
    masks_dir = images_dir.parent / "masks"
    
    samples = []
    print(f"\n[Heart CT] Processing {len(image_files)} images from segmentation dataset...")
    
    for img_path in tqdm(image_files, desc="Heart CT images"):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(data_cfg.image_size, Image.LANCZOS)
            
            filename = f"heartct_{img_path.stem}.{data_cfg.image_format}"
            filepath = IMAGES_DIR / filename
            img.save(filepath)
            
            # Check for corresponding mask
            has_mask = False
            if masks_dir.exists():
                mask_path = masks_dir / img_path.name
                has_mask = mask_path.exists()
            
            description = _generate_heart_ct_description(
                img_path.stem, has_mask
            )
            
            sample = _create_sample(
                image_path=str(filepath),
                question=_get_clinical_question("cardiac_assessment"),
                answer=description
            )
            samples.append(sample)
            
        except Exception as e:
            print(f"[WARNING] Failed to process {img_path}: {e}")
            continue
    
    return samples


def _process_heart_ct_dicom(dicom_files: List[Path],
                             max_images: Optional[int]) -> List[Dict[str, Any]]:
    """Process DICOM files into PNG + text pairs."""
    try:
        import pydicom
    except ImportError:
        print("[WARNING] pydicom not installed. Run: pip install pydicom")
        return []
    
    if max_images:
        dicom_files = dicom_files[:max_images]
    
    samples = []
    print(f"\n[Heart CT] Processing {len(dicom_files)} DICOM files...")
    
    for dcm_path in tqdm(dicom_files, desc="DICOM files"):
        try:
            dcm = pydicom.dcmread(str(dcm_path))
            pixel_array = dcm.pixel_array.astype(np.float64)
            
            # Apply rescale slope/intercept if available
            slope = getattr(dcm, 'RescaleSlope', 1)
            intercept = getattr(dcm, 'RescaleIntercept', 0)
            pixel_array = pixel_array * slope + intercept
            
            # Apply cardiac windowing
            windowed = apply_ct_window(
                pixel_array,
                data_cfg.ct_window_width,
                data_cfg.ct_window_level
            )
            
            img = Image.fromarray(windowed, mode='L').convert('RGB')
            img = img.resize(data_cfg.image_size, Image.LANCZOS)
            
            filename = f"heartct_dcm_{dcm_path.stem}.{data_cfg.image_format}"
            filepath = IMAGES_DIR / filename
            img.save(filepath)
            
            description = _generate_heart_ct_description(dcm_path.stem, False)
            
            sample = _create_sample(
                image_path=str(filepath),
                question=_get_clinical_question("cardiac_assessment"),
                answer=description
            )
            samples.append(sample)
            
        except Exception as e:
            print(f"[WARNING] Failed to process DICOM {dcm_path}: {e}")
            continue
    
    return samples


def _process_generic_images(image_files: List[Path],
                             max_images: Optional[int]) -> List[Dict[str, Any]]:
    """Process generic image files."""
    if max_images:
        image_files = image_files[:max_images]
    
    samples = []
    print(f"\n[Generic] Processing {len(image_files)} image files...")
    
    for img_path in tqdm(image_files, desc="Images"):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(data_cfg.image_size, Image.LANCZOS)
            
            filename = f"generic_{img_path.stem}.{data_cfg.image_format}"
            filepath = IMAGES_DIR / filename
            img.save(filepath)
            
            description = _generate_heart_ct_description(img_path.stem, False)
            
            sample = _create_sample(
                image_path=str(filepath),
                question=_get_clinical_question("cardiac_assessment"),
                answer=description
            )
            samples.append(sample)
            
        except Exception as e:
            continue
    
    return samples


def _generate_heart_ct_description(image_id: str, has_mask: bool) -> str:
    """Generate clinical description for a heart CT image."""
    descriptions = [
        (
            f"This cardiac CT image (ID: {image_id}) demonstrates the cardiac anatomy. "
            "The cardiac chambers and great vessels are visualized. "
            "The myocardial wall thickness appears within normal limits. "
            "No pericardial effusion is identified. "
            "Further assessment with full volumetric analysis is recommended for comprehensive evaluation."
        ),
        (
            f"Cardiac CT slice (ID: {image_id}) shows the cross-sectional anatomy of the heart. "
            "The left and right ventricles are demonstrated with normal morphology at this level. "
            "The interventricular septum is intact. "
            "Assessment of coronary artery calcium scoring requires dedicated analysis."
        ),
        (
            f"This CT image (ID: {image_id}) displays cardiac structures at the current slice level. "
            "The ascending aorta and pulmonary trunk are within normal caliber. "
            "The cardiac silhouette is not enlarged. "
            "Clinical correlation with patient symptoms and additional imaging is advised."
        ),
    ]
    
    desc = random.choice(descriptions)
    
    if has_mask:
        desc += (
            " Segmentation data is available for this slice, indicating annotated cardiac "
            "structures including myocardium and chamber boundaries."
        )
    
    desc += (
        " Note: This is an automated preliminary description. All findings should be verified "
        "by a qualified radiologist."
    )
    
    return desc


# ─────────────────────────────────────────────────────────────
#  Demo Data Generation (no real data needed)
# ─────────────────────────────────────────────────────────────

def generate_demo_data(num_samples: int = 300) -> List[Dict[str, Any]]:
    """
    Generate synthetic demo cardiac CT images for pipeline validation.
    Creates circular/elliptical shapes mimicking cardiac cross-sections.
    Useful when you don't have the Kaggle datasets downloaded yet.
    """
    print(f"\n[Demo] Generating {num_samples} synthetic cardiac CT images...")
    samples = []
    
    for i in tqdm(range(num_samples), desc="Generating demo data"):
        # Create a synthetic "cardiac CT slice"
        img = _create_synthetic_cardiac_slice(i)
        
        filename = f"demo_cardiac_{i:04d}.{data_cfg.image_format}"
        filepath = IMAGES_DIR / filename
        img.save(filepath)
        
        # Generate varied clinical descriptions
        description = _generate_demo_description(i)
        question = _get_clinical_question(random.choice([
            "ccta_analysis", "cardiac_assessment", "findings_summary"
        ]))
        
        sample = _create_sample(
            image_path=str(filepath),
            question=question,
            answer=description
        )
        samples.append(sample)
    
    print(f"[Demo] Generated {len(samples)} synthetic samples")
    return samples


def _create_synthetic_cardiac_slice(seed: int) -> Image.Image:
    """Create a synthetic image resembling a cardiac CT cross-section."""
    rng = random.Random(seed)
    w, h = data_cfg.image_size
    
    # Dark background (like CT)
    bg_val = rng.randint(5, 25)
    img = Image.new('RGB', (w, h), (bg_val, bg_val, bg_val))
    draw = ImageDraw.Draw(img)
    
    # Chest wall (bright ring)
    cx, cy = w // 2, h // 2
    chest_r = int(min(w, h) * 0.42)
    wall_color = rng.randint(140, 180)
    draw.ellipse(
        [cx - chest_r, cy - chest_r, cx + chest_r, cy + chest_r],
        fill=(wall_color, wall_color, wall_color)
    )
    
    # Lung fields (dark regions)
    lung_color = rng.randint(10, 30)
    inner_r = int(chest_r * 0.85)
    draw.ellipse(
        [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
        fill=(lung_color, lung_color, lung_color)
    )
    
    # Heart (central bright structure)
    heart_w = int(chest_r * rng.uniform(0.5, 0.7))
    heart_h = int(chest_r * rng.uniform(0.45, 0.65))
    heart_cx = cx + rng.randint(-15, 5)  # Heart slightly left of center
    heart_cy = cy + rng.randint(-5, 10)
    heart_color = rng.randint(100, 160)
    draw.ellipse(
        [heart_cx - heart_w, heart_cy - heart_h,
         heart_cx + heart_w, heart_cy + heart_h],
        fill=(heart_color, heart_color, heart_color)
    )
    
    # Cardiac chambers (darker regions within heart)
    for _ in range(rng.randint(2, 4)):
        ch_r = int(heart_w * rng.uniform(0.15, 0.35))
        ch_cx = heart_cx + rng.randint(-heart_w // 2, heart_w // 2)
        ch_cy = heart_cy + rng.randint(-heart_h // 2, heart_h // 2)
        ch_color = rng.randint(30, 70)
        draw.ellipse(
            [ch_cx - ch_r, ch_cy - ch_r, ch_cx + ch_r, ch_cy + ch_r],
            fill=(ch_color, ch_color, ch_color)
        )
    
    # Coronary arteries (small bright dots/lines)
    for _ in range(rng.randint(3, 8)):
        ar = rng.randint(2, 5)
        ax = heart_cx + int(heart_w * rng.uniform(-0.8, 0.8))
        ay = heart_cy + int(heart_h * rng.uniform(-0.8, 0.8))
        ar_color = rng.randint(180, 240)
        draw.ellipse(
            [ax - ar, ay - ar, ax + ar, ay + ar],
            fill=(ar_color, ar_color, ar_color)
        )
    
    # Optional: calcification spots
    if rng.random() > 0.5:
        for _ in range(rng.randint(1, 3)):
            cr = rng.randint(1, 4)
            ccx = heart_cx + int(heart_w * rng.uniform(-0.6, 0.6))
            ccy = heart_cy + int(heart_h * rng.uniform(-0.6, 0.6))
            draw.ellipse(
                [ccx - cr, ccy - cr, ccx + cr, ccy + cr],
                fill=(240, 240, 240)
            )
    
    # Apply slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    return img


def _generate_demo_description(idx: int) -> str:
    """Generate varied clinical descriptions for demo data."""
    rng = random.Random(idx)
    
    findings_pool = [
        # Normal findings
        (
            "The cardiac chambers are within normal size limits. The myocardial wall "
            "thickness is preserved. No pericardial effusion is identified. The coronary "
            "arteries appear patent without significant calcification at this level."
        ),
        (
            "Normal cardiac morphology is demonstrated. The left ventricular cavity size "
            "is within normal range. The aortic root dimensions are unremarkable. "
            "No coronary artery calcification is detected."
        ),
        (
            "The cardiac silhouette is normal in size and configuration. The great vessels "
            "demonstrate normal caliber. The mediastinal fat planes are preserved. "
            "No pleural effusion or pericardial abnormality is noted."
        ),
        # Mild findings
        (
            "Mild coronary artery calcification is noted in the left anterior descending "
            "artery (LAD) territory. The calcification is predominantly focal and non-obstructive. "
            "Agatston score estimation suggests low-to-moderate cardiovascular risk. "
            "Clinical correlation and risk factor assessment are recommended."
        ),
        (
            "Scattered coronary calcifications are identified. The right coronary artery (RCA) "
            "shows a small focus of calcified plaque. Left main coronary artery appears "
            "patent. No hemodynamically significant stenosis is suggested at this level."
        ),
        # Moderate findings
        (
            "Moderate coronary artery calcification is present involving the proximal LAD "
            "and left circumflex artery (LCx). The calcium distribution suggests diffuse "
            "atherosclerotic disease. Correlation with functional assessment (stress testing) "
            "may be warranted."
        ),
        (
            "The left ventricular wall shows subtle areas of reduced attenuation, which "
            "may represent prior ischemic changes versus artifact. Coronary artery calcification "
            "is identified in multiple vessels. Clinical correlation recommended."
        ),
        # Cardiomegaly
        (
            "The cardiac silhouette appears mildly enlarged, suggesting cardiomegaly. "
            "The left ventricular dimensions are at the upper limits of normal. "
            "Pericardial fat is unremarkable. Correlation with echocardiography is suggested."
        ),
    ]
    
    finding = rng.choice(findings_pool)
    
    return (
        f"Synthetic cardiac CT analysis (sample {idx}): {finding} "
        "Note: This is a synthetically generated description for pipeline validation. "
        "Not for clinical use."
    )


# ─────────────────────────────────────────────────────────────
#  Shared Utilities
# ─────────────────────────────────────────────────────────────

def _get_clinical_question(question_type: str) -> str:
    """Get a varied clinical question for the user prompt."""
    questions = {
        "ccta_analysis": [
            (
                "You are an expert cardiac radiologist analyzing a Coronary CT Angiography (CCTA) scan. "
                "Describe the clinically relevant findings in this image, including coronary artery "
                "anatomy, any visible plaque, stenosis, or other cardiovascular abnormalities."
            ),
            (
                "Analyze this CCTA image as an expert cardiac radiologist. Report on: "
                "1) Coronary artery visibility and patency "
                "2) Presence and type of any calcification or plaque "
                "3) Cardiac chamber sizes "
                "4) Any additional findings"
            ),
            (
                "Review this Coronary CT Angiography slice and provide a structured report. "
                "Focus on coronary artery assessment, myocardial evaluation, and any incidental findings."
            ),
        ],
        "cardiac_assessment": [
            (
                "You are an expert cardiac radiologist. Analyze this cardiac CT image and describe "
                "the visible cardiac structures, any abnormalities, and relevant clinical findings."
            ),
            (
                "Provide a professional radiological assessment of this cardiac CT scan. "
                "Include observations about cardiac morphology, great vessels, and any pathology."
            ),
        ],
        "findings_summary": [
            (
                "Summarize the key clinical findings visible in this cardiac CT image. "
                "Prioritize findings by clinical significance."
            ),
            (
                "What are the most clinically relevant observations in this cardiac CT scan? "
                "Provide a structured summary suitable for a clinical report."
            ),
        ],
    }
    
    pool = questions.get(question_type, questions["cardiac_assessment"])
    return random.choice(pool)


def _create_sample(image_path: str, question: str, answer: str) -> Dict[str, Any]:
    """Create a single VLM training sample in the correct format."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
    }


def split_dataset(samples: List[Dict[str, Any]], 
                  val_ratio: float = 0.1) -> Tuple[List, List]:
    """Split samples into train and validation sets."""
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - val_ratio))
    return samples[:split_idx], samples[split_idx:]


def save_dataset(samples: List[Dict[str, Any]], filepath: Path):
    """Save dataset as JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {len(samples)} samples -> {filepath}")


# ─────────────────────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cardiac VisionAI — Data Preparation")
    parser.add_argument("--demo", action="store_true",
                        help="Generate synthetic demo data (no real datasets needed)")
    parser.add_argument("--num-samples", type=int, default=data_cfg.pilot_size,
                        help=f"Number of samples to generate (default: {data_cfg.pilot_size})")
    parser.add_argument("--val-ratio", type=float, default=data_cfg.val_ratio,
                        help=f"Validation split ratio (default: {data_cfg.val_ratio})")
    args = parser.parse_args()
    
    # Create directories
    ensure_dirs()
    
    all_samples = []
    
    if args.demo:
        # Demo mode — generate synthetic data
        all_samples = generate_demo_data(num_samples=args.num_samples)
    else:
        # Process real datasets
        # Calculate how many images to take from each source
        per_source = args.num_samples // 2
        
        imagecas_samples = process_imagecas(max_cases=per_source // data_cfg.slices_per_volume)
        heartct_samples = process_heart_ct(max_images=per_source)
        
        all_samples = imagecas_samples + heartct_samples
        
        if not all_samples:
            print("\n[WARNING] No real data found. Falling back to demo mode.")
            print("[TIP] Place your datasets in:")
            print(f"  ImageCAS: {IMAGECAS_DIR}")
            print(f"  Heart CT: {HEART_CT_DIR}")
            print("[TIP] Or run with --demo flag for synthetic data.\n")
            all_samples = generate_demo_data(num_samples=args.num_samples)
    
    # Limit to pilot size
    if len(all_samples) > args.num_samples:
        random.shuffle(all_samples)
        all_samples = all_samples[:args.num_samples]
    
    # Split into train/val
    train_samples, val_samples = split_dataset(all_samples, args.val_ratio)
    
    # Save
    save_dataset(all_samples, DATASET_JSON)
    save_dataset(train_samples, TRAIN_DATASET_JSON)
    save_dataset(val_samples, VAL_DATASET_JSON)
    
    print(f"\n{'='*50}")
    print(f"  Dataset Summary")
    print(f"{'='*50}")
    print(f"  Total samples : {len(all_samples)}")
    print(f"  Train samples : {len(train_samples)}")
    print(f"  Val samples   : {len(val_samples)}")
    print(f"  Image dir     : {IMAGES_DIR}")
    print(f"  Dataset JSON  : {DATASET_JSON}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
