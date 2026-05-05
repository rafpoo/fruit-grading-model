# Fruit Quality Grading — Project Context for Claude Code

## Project Overview

This project builds an automated **fruit quality grading system** using a two-phase machine learning pipeline:

1. **Phase 1 — Unsupervised Labeling**: Extract visual features from fresh fruit images and use K-Means clustering to automatically assign quality grades (A, B, C) — replacing manual labeling.
2. **Phase 2 — Supervised Classification**: Train a CNN (EfficientNetB0) on the K-Means-labeled data to predict the grade of new fruit images.

### Key Design Decisions

- **No spoiled/diseased fruit detection** — the lecturer confirmed this is too visible to the human eye; focus is on grading _fresh_ fruit quality only.
- **Cluster-then-classify approach** — K-Means acts as an automatic labeler (pseudo-labeling), removing human labeling bias and scaling to large datasets.
- **Per-fruit clustering** — K-Means is run separately for each fruit type to ensure it finds quality variation _within_ a fruit, not differences _between_ fruit types.
- **Apple color-variety handling** — apple clustering excludes strong color-identity features (`hue_mean`, `r_mean`, `g_mean`, `b_mean`) so green apples are not automatically treated as lower quality. This reduces the chance that K-Means separates red vs green apple varieties instead of quality.
- **Clustering validation added** — Step 1 now includes a validation cell that reports Silhouette score, Davies-Bouldin index, an elbow plot, and a PASS/CHECK summary per fruit.

---

## Fruits in Scope

| Fruit     | Dataset Folder  |
| --------- | --------------- |
| 🍎 Apple  | `freshapples/`  |
| 🍊 Orange | `freshoranges/` |
| 🍌 Banana | `freshbananas/` |

---

## Grading System

| Grade | Meaning         | Visual Characteristics                                          |
| ----- | --------------- | --------------------------------------------------------------- |
| **A** | Premium quality | Vibrant color, smooth surface, uniform appearance, no blemishes |
| **B** | Average quality | Slight color variation, minor surface marks, mostly uniform     |
| **C** | Low quality     | Dull color, visible marks/bruises (not rot), irregular shape    |

Grades are assigned automatically by ranking K-Means clusters on a quality score based on brightness, saturation, color uniformity, texture homogeneity, blemish ratio, and edge density.

---

## Dataset

**Source**: Kaggle — "Fruits Fresh and Rotten for Classification"

- URL: `kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification`
- Only the **fresh** subfolders are used (`freshapples/`, `freshoranges/`, `freshbananas/`)
- Rotten images are ignored entirely

**Expected folder structure after download:**

```
dataset/
├── train/
│   ├── freshapples/
│   ├── freshoranges/
│   └── freshbananas/
└── test/
    ├── freshapples/
    ├── freshoranges/
    └── freshbananas/
```

### Download Dataset with Kaggle API

Run this once before Phase 1. It downloads the Kaggle dataset, extracts it, and places the files into the `dataset/` folder expected by the notebooks.

#### 1. Install Kaggle API

```bash
pip install kaggle
```

#### 2. Add Kaggle credentials

1. Go to Kaggle account settings: `https://www.kaggle.com/settings`
2. Create/download an API token. This downloads `kaggle.json`.
3. Put `kaggle.json` in:

```text
C:\Users\<your-username>\.kaggle\kaggle.json
```

For this machine, the expected path is:

```text
C:\Users\Rafael Po\.kaggle\kaggle.json
```

#### 3. Notebook download cell

Add/run this near the top of the notebook before feature extraction:

```python
from pathlib import Path
import zipfile
import shutil

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET_SLUG = "sriramr/fruits-fresh-and-rotten-for-classification"
PROJECT_ROOT = Path(r"C:\Users\Rafael Po\Kuliah\Semester 6\DeepLearning\fruit-grading")
RAW_DIR = PROJECT_ROOT / "raw"
DATASET_ROOT = PROJECT_ROOT / "dataset"

RAW_DIR.mkdir(exist_ok=True)
DATASET_ROOT.mkdir(exist_ok=True)

api = KaggleApi()
api.authenticate()

api.dataset_download_files(
    DATASET_SLUG,
    path=str(RAW_DIR),
    unzip=False,
)

zip_path = RAW_DIR / "fruits-fresh-and-rotten-for-classification.zip"
extract_dir = RAW_DIR / "fruits-fresh-and-rotten-for-classification"

if not extract_dir.exists():
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

# Kaggle zip usually contains this nested folder.
source_root = extract_dir / "dataset"
if not source_root.exists():
    source_root = extract_dir

for split in ["train", "test"]:
    for fruit_folder in ["freshapples", "freshoranges", "freshbananas"]:
        src = source_root / split / fruit_folder
        dst = DATASET_ROOT / split / fruit_folder
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            shutil.rmtree(dst)

        shutil.copytree(src, dst)

print("Dataset ready:")
for split in ["train", "test"]:
    for fruit_folder in ["freshapples", "freshoranges", "freshbananas"]:
        folder = DATASET_ROOT / split / fruit_folder
        count = len(list(folder.glob("*")))
        print(f"{split}/{fruit_folder}: {count} images")
```

After this cell succeeds, `DATASET_ROOT = "dataset"` can stay unchanged in the rest of the notebook.

---

## Project Structure

```
fruit-grading/
│
├── CLAUDE.md                          ← you are here
│
├── dataset/                           ← Kaggle dataset (download separately)
│   ├── train/
│   └── test/
│
├── output/                            ← auto-generated by notebooks
│   ├── labeled_dataset.csv            ← K-Means labels, input for CNN
│   ├── kmeans_features.csv            ← raw extracted features
│   ├── cluster_samples/               ← sample images per cluster for inspection
│   │   ├── apple_cluster_0/
│   │   ├── apple_cluster_1/
│   │   ├── apple_cluster_2/
│   │   └── ... (orange, banana too)
│   └── plots/
│       ├── apple_pca.png              ← PCA cluster visualization
│       ├── orange_pca.png
│       ├── banana_pca.png
│       ├── clustering_validation_elbow.png
│       ├── feature_distributions.png
│       ├── grade_distribution.png
│       └── *_cluster_samples.png
│
├── fruit_grading_step1_kmeans.ipynb   ← Phase 1: Feature extraction + K-Means
└── fruit_grading_step2_cnn.ipynb      ← Phase 2: CNN training (TODO)
```

---

## Phase 1 Notebook — `fruit_grading_step1_kmeans.ipynb`

### What it does

Extracts 18 visual features per image and runs K-Means (k=3) per fruit to generate grade labels.

### Features Extracted (18 total)

All 18 features are still extracted and saved to `output/kmeans_features.csv`.

For **apple K-Means clustering only**, the notebook excludes:

- `hue_mean`
- `r_mean`
- `g_mean`
- `b_mean`

Reason: the apple dataset can contain both red and green varieties. Green apples may be Granny Smith-style apples, not bad-quality apples. Removing these color-identity features helps the clustering focus more on quality signals such as brightness, saturation, color uniformity, texture smoothness, blemish ratio, edge density, and shape.

**Color (10 features)**

- `hue_mean`, `hue_std` — color tone and spread
- `sat_mean`, `sat_std` — color richness (high sat = vibrant/ripe)
- `val_mean`, `val_std` — brightness (high = fresh)
- `color_uniformity` — inverse of saturation std; high = consistent color
- `r_mean`, `g_mean`, `b_mean` — raw RGB channel means

**Texture (5 features)** — via Gray-Level Co-occurrence Matrix (GLCM)

- `texture_contrast` — local intensity variation
- `texture_dissimilarity` — similar to contrast, less sensitive to outliers
- `texture_homogeneity` — surface smoothness (high = smooth)
- `texture_energy` — texture uniformity
- `texture_correlation` — gray level linear dependency

**Shape (3 features)**

- `blemish_ratio` — fraction of dark pixels relative to mean brightness
- `edge_density` — Canny edge density (high = rough surface)
- `roundness` — contour circularity (1.0 = perfect circle)

### Auto Grade Assignment Logic

Clusters are ranked by a weighted quality score:

```
score = +0.25 * (val_mean / 255)           # brightness
      + 0.20 * (sat_mean / 255)            # color richness
      + 0.20 * (color_uniformity / 10)     # uniform color
      + 0.15 * texture_homogeneity         # smooth texture
      - 0.10 * blemish_ratio               # fewer blemishes
      - 0.10 * edge_density                # smoother surface
```

Highest score → Grade A, middle → Grade B, lowest → Grade C.

### Clustering Validation

Step 1 now includes a **Clustering Validation** cell immediately after the K-Means cell.

It automatically computes:

- **Silhouette score per fruit** — higher is better; the notebook uses `>= 0.20` as a practical PASS threshold for exploratory image pseudo-labeling.
- **Davies-Bouldin index per fruit** — lower is better; the notebook uses `<= 2.00` as a practical PASS threshold.
- **Elbow plot** — recomputes inertia for `k=2` through `k=8` for each fruit.
- **Summary table** — shows `PASS` or `CHECK` for each fruit based on the two validation metrics.

The elbow plot is saved to:

```text
output/plots/clustering_validation_elbow.png
```

Important: these metrics validate cluster separation, not true fruit quality. If a fruit is marked `CHECK`, inspect `output/cluster_samples/` before using its pseudo-labels for CNN training.

### Manual Override

If auto-assignment looks wrong after inspecting `output/cluster_samples/`, update `GRADE_OVERRIDE` in Cell 4:

```python
GRADE_OVERRIDE = {
    "apple":  {0: "B", 1: "A", 2: "C"},
    "orange": {0: "A", 1: "C", 2: "B"},
    "banana": {0: "C", 1: "B", 2: "A"},
}
```

Then re-run from Cell 10 (K-Means) onwards.

### Key Output

`output/labeled_dataset.csv` with columns:

- `image_path` — absolute path to image file
- `fruit` — apple / orange / banana
- `cluster_id` — raw K-Means cluster (0, 1, or 2)
- `grade` — A / B / C
- `label` — e.g. `apple_A`, `orange_B` (9 classes total)

### Plot Output Note

Any cell that saves plots into `output/plots/` should ensure the folder exists first:

```python
plots_dir = Path(OUTPUT_DIR) / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
```

This avoids `FileNotFoundError` when calling `plt.savefig(...)` before `output/plots/` has been created.

---

## Phase 2 Notebook — `fruit_grading_step2_cnn.ipynb` (TODO)

### Plan

- Load `output/labeled_dataset.csv`
- Split into train / validation / test sets (stratified by `label`)
- Fine-tune **EfficientNetB0** (pretrained on ImageNet) for 9-class classification
- Evaluate with accuracy, F1-score, confusion matrix
- Generate **Grad-CAM** visualizations to show which part of the fruit the model focuses on

### Output Classes (9 total)

`apple_A`, `apple_B`, `apple_C`, `orange_A`, `orange_B`, `orange_C`, `banana_A`, `banana_B`, `banana_C`

---

## Dependencies

```bash
pip install opencv-python scikit-learn scikit-image pandas tqdm matplotlib tensorflow
```

Or for PyTorch-based CNN:

```bash
pip install torch torchvision timm grad-cam
```

---

## Configuration Reference

All configurable values live at the top of each notebook (Cell 4 in Step 1):

| Variable         | Default      | Description                                     |
| ---------------- | ------------ | ----------------------------------------------- |
| `DATASET_ROOT`   | `"dataset"`  | Path to Kaggle dataset root                     |
| `OUTPUT_DIR`     | `"output"`   | Where results are saved                         |
| `IMAGE_SIZE`     | `(224, 224)` | Resize target before feature extraction         |
| `N_CLUSTERS`     | `3`          | Number of K-Means clusters (= number of grades) |
| `N_SAMPLES`      | `10`         | Sample images saved per cluster for inspection  |
| `RANDOM_STATE`   | `42`         | Seed for reproducibility                        |
| `APPLE_EXCLUDED_CLUSTER_FEATURES` | `{hue_mean, r_mean, g_mean, b_mean}` | Apple-only features excluded from K-Means to avoid clustering by red/green variety |
| `GRADE_OVERRIDE` | `{}`         | Manual cluster→grade mapping per fruit          |

---

## Academic Context

- **Methodology**: Unsupervised → Supervised (cluster-then-classify / pseudo-labeling)
- **Grading standard reference**: Codex Alimentarius / USDA fruit grading standards
- **Why clustering first**: Removes human labeling bias; scalable to large datasets without manual annotation effort
- **Why per-fruit clustering**: Prevents K-Means from grouping by fruit type instead of quality
- **Why apple color features are excluded from clustering**: Apple color can represent cultivar/variety, not quality. A clean green apple should not be downgraded just because it is green.
- **Phase 1 validation metrics**: Silhouette score, Davies-Bouldin index, and elbow plot are used to judge whether the pseudo-label clusters are reliable enough for supervised training.
- **Evaluation metrics to report**: Accuracy, F1-score (macro), confusion matrix, Grad-CAM visualizations
