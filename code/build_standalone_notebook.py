"""Build a flat, self-contained Colab notebook from the readable Python modules."""

from __future__ import annotations

import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "code" / "image_ensemble"
OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "H100_Complete_Ensemble_Code.ipynb"

MODULES = (
    "config.py",
    "ensemble.py",
    "data.py",
    "models.py",
    "training.py",
    "runner.py",
)

MODULE_TITLES = {
    "config.py": "Configuration",
    "ensemble.py": "OOF Ensembling",
    "data.py": "Dataset and Preprocessing",
    "models.py": "Model Creation",
    "training.py": "Training and Inference",
    "runner.py": "Experiment Runner",
}


def markdown_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def flattened_module_cell(module_name: str) -> dict[str, object]:
    """Remove package-only imports so a module can run as an ordinary cell."""

    source = (PACKAGE_DIR / module_name).read_text(encoding="utf-8")
    tree = ast.parse(source)
    excluded_lines: set[int] = set()
    for node in ast.walk(tree):
        is_future_import = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        is_relative_import = isinstance(node, ast.ImportFrom) and node.level > 0
        if is_future_import or is_relative_import:
            excluded_lines.update(range(node.lineno, node.end_lineno + 1))

    lines = source.splitlines(keepends=True)
    flattened = "".join(
        line
        for line_number, line in enumerate(lines, start=1)
        if line_number not in excluded_lines
    )
    return code_cell(flattened)


def build_notebook() -> dict[str, object]:
    cells = [
        markdown_cell(
            """# Improved H100 Image Ensemble

This is the self-contained Google Colab notebook. Every part of the complete
PyTorch training pipeline is visible and executable directly in the notebook
cells. It trains ConvNeXt, Swin, and two EVA02 variants.

Only the image dataset must already exist in Google Drive:

```text
MyDrive/Kaggle/train/<class number>/*.jpg
MyDrive/Kaggle/test/*.jpg
```

Choose an H100 or A100 runtime before running the cells. A complete run is
computationally expensive.
"""
        ),
        markdown_cell(
            """## 1. Install Dependencies

The notebook uses PyTorch from the Colab runtime and installs the additional
libraries needed by the project.
"""
        ),
        code_cell("!pip -q install timm scikit-learn pandas Pillow\n"),
        markdown_cell(
            """## 2. Mount Google Drive

Run this cell directly inside the browser version of Google Colab. If Google
returns an authentication `400` error, disconnect the runtime, allow Google
authentication pop-ups, reconnect, and run the cell again.
"""
        ),
        code_cell(
            """from google.colab import drive
drive.mount("/content/drive")
"""
        ),
        markdown_cell(
            """## 3. Configure PyTorch Memory Allocation

Expandable CUDA segments can reduce failures caused by reserved-memory
fragmentation during long multi-model runs.
"""
        ),
        code_cell(
            """import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
"""
        ),
        markdown_cell(
            """## 4. Define the Training Pipeline

The following cells contain the full implementation directly. No Python package
files or source-code upload are required.
"""
        ),
    ]
    for module_name in MODULES:
        cells.append(markdown_cell(f"### {MODULE_TITLES[module_name]}\n"))
        cells.append(flattened_module_cell(module_name))
    cells.extend(
        [
            markdown_cell(
                """## 5. Configure the Run

Leave `QUICK_RUN = False` for the full experiment. Set it to `True` only when
checking that the dataset paths and pipeline are wired correctly.

The defaults are intentionally memory-conscious: smaller micro-batches are
paired with gradient accumulation, EMA is opt-in, and DataLoader prefetching is
limited. This is slower than an aggressive H100-only setup but much less likely
to exhaust GPU or host memory.
"""
            ),
            code_cell(
                """from dataclasses import replace
from pathlib import Path

KAGGLE_DIR = Path("/content/drive/MyDrive/Kaggle")
OUTPUT_DIR = KAGGLE_DIR / "improved_oof_ensemble"
QUICK_RUN = False

config = default_colab_config(KAGGLE_DIR)
config = replace(config, output_dir=OUTPUT_DIR)
if QUICK_RUN:
    config = replace(
        config,
        num_folds=2,
        final_epochs=2,
        blend_search_iterations=300,
    )

print("Train directory:", config.train_dir)
print("Test directory:", config.test_dir)
print("Output directory:", config.output_dir)
print("Variants:", [variant.name for variant in config.variants if variant.enabled])
for variant in config.variants:
    if variant.enabled:
        effective_batch = variant.batch_size * variant.accum_steps
        print(
            f"{variant.name}: micro-batch={variant.batch_size}, "
            f"accumulation={variant.accum_steps}, effective batch={effective_batch}, "
            f"gradient checkpointing={variant.gradient_checkpointing}, EMA={variant.use_ema}"
        )
"""
            ),
            markdown_cell(
                """## 6. Check Dataset Paths

This fast check catches missing Drive folders before GPU training begins.
"""
            ),
            code_cell(
                """assert config.train_dir.exists(), f"Missing training directory: {config.train_dir}"
assert config.test_dir.exists(), f"Missing test directory: {config.test_dir}"

train_class_directories = [
    path for path in config.train_dir.iterdir()
    if path.is_dir() and path.name.isdigit()
]
test_images = [
    path for path in config.test_dir.iterdir()
    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
]

print("Training class folders:", len(train_class_directories))
print("Test images:", len(test_images))
assert len(train_class_directories) == config.num_classes
assert test_images
"""
            ),
            markdown_cell(
                """## 7. Train and Create the Submission

This cell trains all enabled folds, generates out-of-fold predictions, selects
TTA behavior, learns ensemble weights, and writes the final CSV to Google Drive.
"""
            ),
            code_cell(
                """submission_path = run_experiment(config)
print("Submission ready:", submission_path)
"""
            ),
        ]
    )
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    OUTPUT_PATH.write_text(
        json.dumps(notebook, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Cells: {len(notebook['cells'])}")


if __name__ == "__main__":
    main()
