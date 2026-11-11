# CIFAR-10 CNN assignment

**Sajjad Ranjbar — 405522012**  
Computer Science, Faculty of Mathematical Sciences, Shahid Beheshti University

Open `Noebooks/CIFAR10_CNN_Assignment.ipynb` and run its cells in order.
It contains data inspection, four architecture experiments, intermediate
activations, validation-based model selection, one final test evaluation,
and a validation-only occlusion experiment. Results use real CIFAR-10 data.

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

The notebook uses seed 42, the full fixed 45,000/5,000/10,000 split,
10 epochs per configuration, batch size 128, AdamW, and the same cosine
learning-rate schedule for each run. Defaults can run on CPU or CUDA.
`CIFAR_ROOT` optionally points to a dataset cache; `CNN_NUM_THREADS`
optionally controls CPU threads (default 4). Run all cells for a fresh
experiment; checkpoints are overwritten, not silently reused. Local model
checkpoints are ignored by Git; measured logs and figures are versioned.
Runtime and exact architecture/settings are recorded in `report/results/`.

CIFAR-10 is obtained from its official source when accessible. A pinned
GitHub mirror is used as a fallback; every original batch and metadata file
must match torchvision's published MD5 before loading. The source commit
and SHA-256 digests are recorded. Downloaded data are ignored by Git.
Normalization uses only the 45,000 training examples, never validation or test.

## Report

LaTeX source: `Latex/report.tex`. Compiled PDF: `report/report.pdf`.
From the repository root:

```bash
python Latex/generate_results.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=report Latex/report.tex
```

The generation step audits the saved test predictions and regenerates the
LaTeX tables and numerical discussion; it does not run test inference again.

Figures and measured JSON/CSV logs are in `report/figures/` and
`report/results/`. The PDF and notebook report the limitations of the short
CPU experiments; these are not claims of a globally optimal architecture.

## Attribution

This completed implementation and report were prepared with OpenAI Codex
assistance at the student's request. The assignment template is from this
repository; the data-loading and model APIs are from PyTorch/torchvision,
and metrics are from scikit-learn. The named student should review the
code, measured results, and explanations before submission. No performance
numbers are invented or copied from other experiments.
