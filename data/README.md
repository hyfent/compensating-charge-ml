# Data dictionary

## `metrics/`

The four metric tables contain one row per model/seed/evaluation condition.

Common columns:

| Column | Meaning |
|---|---|
| `network` | `small`, `medium`, or `large` architecture |
| `train_size` | Number of labeled training structures |
| `total_epochs` | Optimization budget |
| `seed` | Matched random seed |
| `mode` | `baseline` = conventional ML; `ic` = CC-ML |
| `mae_atom` | Atom-wise mean absolute charge error in elementary-charge units |
| `rmse_atom` | Atom-wise root-mean-square charge error |
| `structures` | Number of evaluated structures |
| `atoms` | Number of evaluated atomic predictions |
| `run_id` | Traceable identifier of the source run |

Low-to-high tables additionally contain:

| Column | Meaning |
|---|---|
| `task` | `deformation` or `temperature` |
| `stage` | Individual high-condition target |

Element-specific MAE and RMSE columns are retained where available.

Expected row counts:

| File | Rows |
|---|---:|
| `nacl_test_metrics.csv` | 432 |
| `cspbi3_test_metrics.csv` | 432 |
| `nacl_low_to_high_metrics.csv` | 1296 |
| `cspbi3_low_to_high_metrics.csv` | 1296 |

## `auxiliary/`

- `madelung_convergence.csv`: complete-shell direct and locally neutralized
  NaCl Madelung sums.
- `structural_shift_distances.csv`: per-structure distances from the
  low-condition descriptor centroid used in the structural-shift panels.
- `structural_shift_summary.csv`: median and interquartile summaries of those
  distances.
- `timing_raw.csv`: all recorded single-core wall-time repetitions.
- `timing_summary.csv`: precomputed timing summary retained for cross-checking.

All charge errors are reported in units of the elementary charge, `e`. Timing
values are in seconds unless the column name explicitly states otherwise.

## `example/`

The two fit-data files contain 12 complete, periodic, Ewald-QEq-labeled
structures for NaCl (216 atoms per structure) and CsPbI$_3$ (320 atoms per
structure). They are small subsets of the project data and support the runnable
labeling, training, and prediction examples. They are not used to recompute the
full manuscript learning curves.

Each structure is stored as a comment line, atom count, 3x3 cell matrix, and
one atom line per site:

```text
element  x  y  z  0  0  0  reference_charge
```
