# Compensating-Charge-Informed Machine Learning

Data and code supporting the manuscript **“Compensating Charge-Informed
Machine-Learning Model for Accurate and Efficient Charge Prediction in Ionic
Materials.”**

The repository contains processed results, model and material configurations,
example labeled structures, and scripts for statistical analysis, plotting,
charge labeling, training, and prediction for NaCl and CsPbI3.

## Contents

- `config/`: model, material, and QEq settings
- `data/metrics/`: processed test and low-to-high transfer results
- `data/auxiliary/`: Madelung, structural-shift, and timing data
- `data/example/`: compact labeled NaCl and CsPbI3 examples
- `scripts/`: analysis, plotting, labeling, training, and prediction code

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_all.py
```

This validates the released data, recomputes the reported aggregate
statistics, and generates the corresponding figures in `figures/`.

The included training and prediction examples can be inspected with:

```bash
python scripts/train_example.py --help
python scripts/predict.py --help
python scripts/qeq_label.py --help
```

## Data scope

The processed tables reproduce the aggregate statistics and plots reported in
the manuscript. The labeled example structures provide a compact demonstration
of the workflow; complete trajectories and model checkpoints are not included.

## Citation

Citation metadata are provided in `CITATION.cff`.

## License

No reuse license is currently provided.
