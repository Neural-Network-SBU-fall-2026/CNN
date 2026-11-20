# CIFAR-10 CNN assignment

**Sajjad Ranjbar — 405522012**  
Computer Science, Faculty of Mathematical Sciences, Shahid Beheshti University

The notebook compares four CNN architectures on CIFAR-10. It includes the
training curves, feature maps for an airplane and a cat, and a final test
evaluation of the model selected on validation data.

- [Assignment notebook](Noebooks/CIFAR10_CNN_Assignment.ipynb)
- [PDF report](report/report.pdf)

## Environment and reproduction

Use Python 3.12 and a virtual environment. For CPU PyTorch:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
python -m pip install -r requirements.txt
cd Noebooks
jupyter lab
```

Run the notebook cells in order to reproduce the experiments. Each configuration
uses the same 45,000 training examples and 5,000 validation examples, seed 42,
and ten epochs of AdamW with batch size 128 and a cosine learning-rate schedule.
The official 10,000-image test split is used for the final evaluation.

The notebook uses CUDA when available and defaults to four threads on CPU.
Set `CNN_NUM_THREADS` to change the thread count or `CIFAR_ROOT` to point to a
dataset cache. If the official CIFAR-10 download is unavailable, the notebook
uses a pinned GitHub mirror. All batch files and the metadata file must match
torchvision's checksums before loading. A fresh run replaces local checkpoints
and regenerates the results and plots.

## Report

Build the report from the saved results by running these commands from the
repository root.

```bash
python Latex/generate_results.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=report Latex/report.tex
```

The generator checks the saved test predictions and rebuilds the LaTeX tables
and results discussion. The experiment logs, split indices and data provenance
are in `report/results/`. The notebook and report cite the dataset and libraries.
