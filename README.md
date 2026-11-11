# Skeleton Data Validation Tool

Toolkit for validating keypoint (skeleton) annotations in YOLO Pose datasets:

- keypoint quality validation,
- detection of mislabeled samples,
- a merged anomaly ranking built from three unsupervised signals: inter-keypoint distances, LBP texture around keypoints and segmentation masks,
- YOLO Pose dataset operations (merge, split, flatten, duplicate removal, synthetic keypoint noise),
- format conversion (including COCO keypoints).

The code is an installable Python package, `pose_data_validation`:

- `pose_data_validation.validation`: image- and dataset-level validation and the merged ranking,
- `pose_data_validation.datasets`: dataset operations, formats and converters.

Repository layout:

| Path | Contents |
|---|---|
| `pose_data_validation/` | the package |
| `scores_from_publication/` | notebooks with the experiments and results reported in the publication |
| `example_keypoint_formats_usage.ipynb` | loading datasets in other keypoint formats |
| `test_validation_smoke.py` | smoke tests |

**Author:** Rafał Wysocki, Poznań University of Technology

## 1. Requirements

- Python 3.10 or newer
- Windows, Linux or macOS
- Optional: an NVIDIA GPU (CUDA) for faster embedding, segmentation and YOLO training
- Internet access on first use: CLIP weights (`open_clip`) and Ultralytics YOLO weights are downloaded automatically

## 2. Installation

Windows (PowerShell):

```powershell
git clone https://github.com/Mevaco2000/SKELETON-DATA-VALIDATION-TOOL.git
cd SKELETON-DATA-VALIDATION-TOOL
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Linux / macOS:

```bash
git clone https://github.com/Mevaco2000/SKELETON-DATA-VALIDATION-TOOL.git
cd SKELETON-DATA-VALIDATION-TOOL
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the dependency versions. `pip install -e .` installs the `pose_data_validation` package in editable mode, so it can be imported from any folder (including the notebooks), and changes to the code take effect without reinstalling.

**GPU.** On Windows, `pip` installs a CPU-only `torch`. For CUDA, first install the `torch` version pinned in `requirements.txt` and the matching `torchvision` from the [PyTorch index](https://pytorch.org/get-started/locally/), then run the commands above.

**PowerShell blocks activation.** If activation fails with `running scripts is disabled on this system`, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This change affects only the current PowerShell session.

**ZIP download.** If you downloaded a ZIP instead of cloning, extract it and run the same commands (without `git clone`) from the folder that contains `pyproject.toml`.

### Check the installation

```powershell
python test_validation_smoke.py
```

Expected final line:

```text
All smoke tests passed.
```

### Publication notebooks

The notebooks in `scores_from_publication/` contain the saved outputs of the experiments. They use absolute paths to the datasets on the author's machine; change these paths before re-running a notebook.

## 3. Using the package in another project

The library is packaged as `pose-data-validation` (import name `pose_data_validation`).
Only the library is installed: notebooks and tests stay in this repository.

Install it into the other project's environment straight from GitHub:

```powershell
pip install "pose-data-validation @ git+https://github.com/Mevaco2000/SKELETON-DATA-VALIDATION-TOOL.git"
```

For reproducible results, pin a commit or tag by appending `@<commit-or-tag>` to the URL.

Or install it from a local clone in editable mode:

```powershell
pip install -e path\to\SKELETON-DATA-VALIDATION-TOOL
```

In the other project's `requirements.txt`:

```text
pose-data-validation @ git+https://github.com/Mevaco2000/SKELETON-DATA-VALIDATION-TOOL.git
```

All dependencies (including `torch`, `ultralytics`, `open-clip-torch` and `faiss-cpu`) are installed with the package. For GPU support, install `torch` and `torchvision` first, as described in the GPU note in section 2.

## 4. Quick API usage

```python
from pose_data_validation import YOLOPoseDataset, YPSetValidation, MergedRankingEvaluator

dataset = YOLOPoseDataset("path/to/train.txt")
validator = YPSetValidation(dataset)

evaluator = MergedRankingEvaluator(
    validator=validator,
    report_paths={},
    top_k=200,
    output_path="ranking_report.txt",
    distance_model_name="random_forest",
    segmentation_model_name=None,
)

results = evaluator.run(
    distance_weight=1.0,
    lbp_weight=1.0,
    segmentation_weight=0.0,
)

print("Ranking rows:", len(results.get("combined_ranking", [])))
```

## 5. Working with multiple input formats

Registered formats include:

- `yolo_pose`
- `coco_keypoints`

```python
from pose_data_validation.datasets import KeypointDatasetAdapter

with KeypointDatasetAdapter.from_coco(
    annotations_dir="path/to/coco/annotations",
    images_dir="path/to/coco/images",
    output_root="./converted_yolo_pose",
    split_name="train",
) as adapter:
    dataset = adapter.as_yolo_dataset()
    print("Samples:", len(dataset))
    print(adapter.info())
```
