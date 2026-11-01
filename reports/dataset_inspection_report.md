# Phase 1: Dataset Inspection Report

**Project:** WasteWise Recycling Assistant  
**Dataset Path:** `data/raw/DATASET/`  
**Date of Inspection:** November 1, 2026  
**Inspection Status:** Completed Successfully  

---

## 1. Directory Structure & Folder Verification

The dataset directory was verified at `c:\Users\pavan\WasteWise-Recycling-Assistant\data\raw\DATASET`. All required folders exist and are directly accessible.

```
data/raw/DATASET/
├── TRAIN/
│   ├── O/  (Organic)
│   └── R/  (Recyclable)
└── TEST/
    ├── O/  (Organic)
    └── R/  (Recyclable)
```

> [!NOTE]
> An extra nested directory was observed at `data/raw/DATASET/DATASET/` containing duplicate copies of `TRAIN` and `TEST` (likely from archive extraction). In accordance with instructions, **no files or folders were modified, moved, or deleted**. The primary dataset under `data/raw/DATASET/` was inspected.

---

## 2. Image Count Verification

All four folders were verified by inspecting each file individually:

| Split | Class Label | Class Name | Verified Count | Expected Count | Status |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **TRAIN** | `O` | Organic | **12,565** | 12,565 | Match |
| **TRAIN** | `R` | Recyclable | **9,999** | 9,999 | Match |
| **TEST** | `O` | Organic | **1,401** | 1,401 | Match |
| **TEST** | `R` | Recyclable | **1,112** | 1,112 | Match |
| **Total** | — | — | **25,077** | **25,077** | **100% Match** |

### Split & Class Distribution
- **Total Dataset Size:** 25,077 images (211.94 MB / 0.21 GB)
- **Train Split:** 22,564 images (89.98%)
- **Test Split:** 2,513 images (10.02%)
- **Class Balance (Overall):**
  - Organic (`O`): 13,966 images (55.69%)
  - Recyclable (`R`): 11,111 images (44.31%)
- **Class Balance (Train Split):**
  - Organic (`O`): 12,565 images (55.69%)
  - Recyclable (`R`): 9,999 images (44.31%)
- **Class Balance (Test Split):**
  - Organic (`O`): 1,401 images (55.75%)
  - Recyclable (`R`): 1,112 images (44.25%)

The class distribution is consistent across both train and test splits (~55.7% Organic to ~44.3% Recyclable).

---

## 3. Image Integrity & Corruption Check

Every single image file (25,077 / 25,077) was decoded and its raw pixel buffer loaded using PIL (`Image.open()` followed by `img.verify()` and `img.load()`):

- **Corrupted Images:** `0`
- **Unreadable / Missing Files:** `0`
- **Truncated Files:** `0`
- **Integrity Pass Rate:** `100.0%`

---

## 4. Image Formats, Color Modes & Dimensions

### File Extensions vs. Internal Encoding
All 25,077 files end with the `.jpg` extension on disk. However, internal header decoding revealed:
- **JPEG Format:** 24,705 images (98.52%)
- **PNG Format:** 372 images (1.48%) — encoded internally as PNG despite having `.jpg` filename extension.

### Color Modes
- **RGB (3 channels):** 24,705 images (98.52%)
- **P (Palette / Indexed 8-bit color):** 372 images (1.48%) — corresponding to the 372 PNG files.

> [!IMPORTANT]
> **Data Loader Requirement:** Image loaders in PyTorch/TensorFlow (e.g., `PIL.Image.open(path).convert('RGB')` or `torchvision.io.read_image`) must explicitly convert images to RGB mode to normalize the 372 palette-based images to standard 3-channel RGB tensors.

### Dimensions Statistics
- **Minimum Dimensions:** 51 x 69 pixels
- **Maximum Dimensions:** 1460 x 1500 pixels
- **Average Dimensions:** 252.42 x 203.78 pixels
- **Top 5 Most Common Resolutions:**
  1. `225 x 225`: 4,895 images (19.52%)
  2. `275 x 183`: 2,837 images (11.31%)
  3. `259 x 194`: 1,911 images (7.62%)
  4. `300 x 168`: 1,055 images (4.21%)
  5. `183 x 275`: 588 images (2.34%)

A target input size of `224 x 224` or `256 x 256` standard for vision backbones (ResNet, EfficientNet, MobileNet) is well-aligned with the median image resolutions in this dataset.

---

## 5. Class Label Confirmation

- **`O`**: Organic waste (e.g. food scraps, fruit peels, biological items)
- **`R`**: Recyclable waste (e.g. plastic bottles, metal cans, paper, glass)
- Verified naming convention:
  - `TRAIN/O`: `O_1.jpg`, `O_2.jpg`, ... up to `O_12565.jpg`
  - `TRAIN/R`: `R_1.jpg`, `R_2.jpg`, ... up to `R_9999.jpg`
  - `TEST/O`: `O_12568.jpg`, ...
  - `TEST/R`: `R_10000.jpg`, ...

---

## 6. Python Accessibility

- Python environment: Python 3.12 (executable at `C:\Users\pavan\AppData\Local\Programs\Python\Python312\python.exe`)
- Direct read access verified with Python's standard `os`, `pathlib`, and `PIL` (Pillow 12.0.0).
- File permissions and read performance: Full read access, 25,077 files read and validated in ~282 seconds.

---

## 7. Summary & Phase 1 Conclusion

| Requirement | Result |
| :--- | :--- |
| **Inspect dataset directory** | Completed (`data/raw/DATASET`) |
| **Verify folder counts** | Completed (25,077 verified against expected counts) |
| **Check corruption** | Completed (0 corrupted, 0 unreadable) |
| **Check formats & dimensions** | Completed (Identified 372 palette-based PNGs disguised as `.jpg`; mapped resolutions) |
| **Confirm labels** | Confirmed: `O` = Organic, `R` = Recyclable |
| **Python accessibility** | Confirmed (100% accessible) |
| **No dataset modification** | Complied (0 files modified, deleted, or moved) |
| **No model training** | Complied (0 models trained) |
| **No fabricated metrics** | Complied (all figures generated from actual file checks) |
