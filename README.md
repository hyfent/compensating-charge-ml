# Compensating-Charge-Informed Machine Learning

Compact data and code release supporting the manuscript
**“Compensating Charge-Informed Machine-Learning Model for Accurate and
Efficient Charge Prediction in Ionic Materials.”**

The repository is intentionally compact but covers two reproducibility levels:

1. **Manuscript results:** processed per-run tables, raw timing repetitions,
   Madelung and structural-shift data, statistical checks, and plotting code.
2. **Method workflow:** real labeled NaCl and CsPbI3 examples plus runnable
   Ewald-QEq labeling, conventional-ML training, CC-ML training, and prediction.

The small training examples verify the workflow; they are not substitutes for
the complete training sets used to obtain the manuscript statistics.

## Contents

```text
CCML_GitHub_release/
├── README.md
├── CITATION.cff
├── SHA256SUMS
├── requirements.txt
├── config/
│   ├── model_config.json
│   ├── material_nacl.json
│   ├── material_cspbi3.json
│   ├── qeq_nacl.json
│   └── qeq_cspbi3.json
├── data/
│   ├── README.md
│   ├── metrics/
│   │   ├── nacl_test_metrics.csv
│   │   ├── cspbi3_test_metrics.csv
│   │   ├── nacl_low_to_high_metrics.csv
│   │   └── cspbi3_low_to_high_metrics.csv
│   └── auxiliary/
│       ├── madelung_convergence.csv
│       ├── structural_shift_distances.csv
│       ├── structural_shift_summary.csv
│       ├── timing_raw.csv
│       └── timing_summary.csv
│   └── example/
│       ├── nacl_12_structures.fit.data
│       └── cspbi3_12_structures.fit.data
├── scripts/
│   ├── ccml_core.py
│   ├── data_io.py
│   ├── qeq_label.py
│   ├── train_example.py
│   ├── predict.py
│   ├── analysis.py
│   ├── plot_madelung.py
│   ├── plot_performance.py
│   ├── plot_structural_shift.py
│   ├── plot_timing.py
│   ├── validate_release.py
│   └── reproduce_all.py
└── figures/                 # generated locally; ignored by Git
```

## Quick start

Python 3.8 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce_all.py
```

The command performs four steps:

1. checks file completeness and expected row counts;
2. verifies one-to-one conventional-ML/CC-ML pairing;
3. recomputes pairwise gains, seed-averaged matrix cells, and effective-sample ratios;
4. regenerates compact versions of the numerical figures in `figures/`.

Tracked data and configuration files can be integrity-checked with
`sha256sum -c SHA256SUMS`.

PyTorch is needed for training and prediction. Statistical analysis and
plotting require only NumPy, pandas, and Matplotlib.

## End-to-end example

The following commands train and evaluate both predictors on the included NaCl
sample. They use the same descriptor definition, hidden-layer architecture,
AdamW settings, charge correction, and three-round CC-ML feedback specified in
`config/model_config.json`.

```bash
python scripts/train_example.py \
  --data data/example/nacl_12_structures.fit.data \
  --material config/material_nacl.json \
  --mode conventional --epochs 30 \
  --output example_outputs/nacl_conventional

python scripts/train_example.py \
  --data data/example/nacl_12_structures.fit.data \
  --material config/material_nacl.json \
  --mode ccml --epochs 30 \
  --output example_outputs/nacl_ccml

python scripts/predict.py \
  --data data/example/nacl_12_structures.fit.data \
  --artifact example_outputs/nacl_ccml
```

The same commands accept `data/example/cspbi3_12_structures.fit.data` together
with `config/material_cspbi3.json`.

For a transparent one-structure label-generation check:

```bash
python scripts/qeq_label.py \
  --data data/example/nacl_12_structures.fit.data \
  --qeq config/qeq_nacl.json \
  --limit 1 --output example_outputs/nacl_relabel.fit.data
```

`qeq_label.py` is a compact periodic Ewald-QEq reference implementation. It is
provided to expose the label-generation equations and parameters, not as an
optimized production solver.

## Method reference

`scripts/ccml_core.py` provides the method-defining operations:

- element-resolved radial structural descriptors;
- atom-centered neighborhoods under periodic boundary conditions;
- compensating charge
  `Delta q_i = -sum_{j in N_i} q_j`;
- uniform structure-level total-charge projection;
- fully connected SiLU charge network;
- fixed-depth compensating-charge feedback using corrected intermediate charges.

The released configuration uses a 6 Å cutoff, three feedback rounds, formal
initial charges, and the small/medium/large architectures listed in
`config/model_config.json`.

## Data scope

The four files in `data/metrics/` are immutable per-run numerical summaries
derived from the final charge-corrected predictions. They preserve the network,
training-set size, epoch budget, random seed, model type, MAE, RMSE, and
element-resolved errors needed to verify the manuscript statistics.

The compact release does **not** duplicate multi-gigabyte checkpoints, complete
MD trajectories, or every atom-level prediction file. The processed tables are
sufficient to reproduce the reported aggregate statistics and plots, while the
included labeled structures support an executable method demonstration.
Additional raw artifacts can be supplied by the corresponding author upon
reasonable request.

## Statistical definitions

For each matched setting,

```text
I_CC = 100 * (MAE_conventional - MAE_CCML) / MAE_conventional
```

For low-to-high tests, the MAE is first averaged over the three target
conditions within each deformation or temperature path for the same network,
training size, epoch budget, seed, and model type.

Effective sample efficiency is the CC-ML training-set size divided by the
smallest sampled conventional-ML training-set size reaching the same
seed-averaged MAE. If the conventional model does not reach the target within
the sampled range, the value based on its largest sampled set is marked as an
upper bound in the derived table.

## Reproducing individual components

```bash
python scripts/validate_release.py
python scripts/analysis.py
python scripts/qeq_label.py --help
python scripts/train_example.py --help
python scripts/predict.py --help
python scripts/plot_madelung.py
python scripts/plot_performance.py
python scripts/plot_structural_shift.py
python scripts/plot_timing.py
```

Derived statistics are written to `data/derived/`, and figures are written to
`figures/`. Both directories can be regenerated from the tracked source data.

## Suggested Data Availability Statement after GitHub publication

Replace the placeholder URL with the permanent repository URL or archive DOI:

```text
The processed data and scripts supporting the findings of this study are openly available at [repository URL or DOI]. Additional raw training artifacts are available from the corresponding author upon reasonable request.
```

For a citable release, archive a tagged GitHub release in Zenodo and use the
resulting DOI in both the manuscript statement and the data citation.

## Citation

Citation metadata are provided in `CITATION.cff`. Update the repository URL,
release version, publication DOI, and author ORCID identifiers before creating
the public release.

## License

No reuse license has been selected in this draft package. Before publication,
choose an explicit code and data license after confirming institutional and
funder requirements.
