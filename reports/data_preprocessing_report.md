# Phase 2: Data Loading and Preprocessing Report

**Project:** WasteWise Recycling Assistant  
**Dataset Root:** `data/raw/DATASET/`  
**Pipeline Source:** `src/data_loader.py`  
**Test Suite:** `tests/test_data_loader.py`  
**Date:** November 1, 2026  
**Status:** Complete & Fully Validated  

---

## 1. Executive Summary

Phase 2 implemented and validated a modular, high-performance data loading and preprocessing pipeline for the WasteWise Recycling Assistant. The pipeline ingests the 25,077 verified images, encodes labels canonically, converts all images explicitly to 3-channel RGB (resolving the 372 palette-encoded PNG images discovered in Phase 1), resizes inputs to 224 × 224 pixels, normalizes pixel intensities to `[-1.0, 1.0]` for MobileNetV2, and partitions the training set into training and validation subsets using stratified sampling with a fixed random seed (`42`). The provided test set remains completely isolated.

All 9 unit tests passed successfully.

---

## 2. Dataset Split Counts and Stratification

The 22,564 images in `TRAIN/` were partitioned using a stratified 80/20 train/validation split (`val_ratio=0.20`, `random_state=42`). The test set (`TEST/`) of 2,513 images was kept strictly untouched and isolated.

| Split | Total Images | Organic (`O`, 0) | Recyclable (`R`, 1) | % Organic | % Recyclable |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Subset** | **18,051** | 10,052 | 7,999 | 55.69% | 44.31% |
| **Validation Subset** | **4,513** | 2,513 | 2,000 | 55.68% | 44.32% |
| **Total Train (Combined)** | **22,564** | 12,565 | 9,999 | 55.69% | 44.31% |
| **Test Set (Isolated)** | **2,513** | 1,401 | 1,112 | 55.75% | 44.25% |
| **Grand Total** | **25,077** | **13,966** | **11,111** | **55.69%** | **44.31%** |

### Stratification Accuracy
- The baseline training proportion is **55.69% Organic** to **44.31% Recyclable**.
- The validation subset exhibits **55.68% Organic** to **44.32% Recyclable** (deviation < 0.01%).
- Stratification guarantees balanced evaluation during model validation without sampling bias.

---

## 3. Preprocessing Specifications

### 3.1. Label Mapping
- **Organic (`O`)**: `0`
- **Recyclable (`R`)**: `1`

### 3.2. Color Normalization (RGB Conversion)
- Phase 1 revealed 372 files with `.jpg` extensions that are internally encoded as PNG files using palette/indexed color mode (`P`).
- The pipeline explicitly calls `.convert('RGB')` in PIL and uses `tf.image.decode_image(..., channels=3, expand_animations=False)` in TensorFlow.
- Verified: Both JPEG and PNG images are loaded as standard 3-channel RGB tensors with shape `(224, 224, 3)`.

### 3.3. Spatial Resizing
- Target resolution: **224 × 224 pixels** (height × width).
- Interpolation: Bilinear resampling.

### 3.4. Intensity Scaling for MobileNetV2
- MobileNetV2 requires input pixel values scaled to the range `[-1.0, 1.0]`:
  $$\text{Pixel}_{\text{norm}} = \frac{\text{Pixel}}{127.5} - 1.0$$
- Verified minimum and maximum pixel bounds: `min = -1.0`, `max = 1.0`, dtype = `float32`.

---

## 4. Leakage Prevention and Dataset Isolation

To ensure scientific validity and avoid data contamination:
1. **Zero Overlap**: Filepath sets for Train, Validation, and Test were compared using set intersections:
   - $\text{Train} \cap \text{Val} = \emptyset$ (0 overlapping files)
   - $\text{Train} \cap \text{Test} = \emptyset$ (0 overlapping files)
   - $\text{Val} \cap \text{Test} = \emptyset$ (0 overlapping files)
2. **Zero Dropped Files**: All 25,077 filepaths are accounted for with explicit length assertions.
3. **No Modification**: Original files in `data/raw/DATASET/` remain untouched and read-only.
4. **Reproducibility**: Splitting is deterministic using `random_state=42`.

---

## 5. Automated Unit Test Results

The test suite in `tests/test_data_loader.py` was executed with `python -m unittest tests/test_data_loader.py`:

```
Ran 9 tests in 9.627s

OK
```

### Detailed Test Case Coverage:
1. `test_label_mappings`: Validates bidirectional mappings (`O` $\leftrightarrow$ `0`, `R` $\leftrightarrow$ `1`).
2. `test_record_counts_and_no_missing_files`: Verifies exact count matching across all folders without omissions.
3. `test_stratified_split_proportions`: Confirms exact 18,051 / 4,513 split and stratification ratio retention.
4. `test_dataset_isolation_and_no_leakage`: Confirms pairwise disjointness of splits.
5. `test_stratified_split_reproducibility`: Confirms identical output across runs with fixed seed.
6. `test_pil_image_preprocessing_standard_jpeg`: Confirms PIL load, shape `(224, 224, 3)`, float32, range `[-1.0, 1.0]`.
7. `test_pil_image_preprocessing_palette_png`: Confirms RGB conversion for palette PNGs (`O_10083.jpg`).
8. `test_tf_preprocessing_function`: Tests native TensorFlow graph preprocessing.
9. `test_tf_dataset_pipeline_batching`: Validates `tf.data.Dataset` batch generation, prefetching, and labels.

---

## 6. Artifacts Created

1. `src/__init__.py`: Package initialization.
2. `src/data_loader.py`: Production-ready data loading and preprocessing module.
3. `tests/test_data_loader.py`: Comprehensive test suite for data ingestion and transforms.
4. `reports/data_preprocessing_report.md`: Formal Phase 2 documentation.
