# Running Experiments and Evaluations

To generate uncertainty quantification (UQ) results and populate all evaluation metrics, run:

```bash
cd scripts

# Run model inference jobs
bash run_all_models_sbatch.sh

# Run evaluation pipeline
bash run_all_eval.sh
```

## Notes

* In this repository, `score` refers to **point-wise scoring**.
* All scripts are designed for batch execution on SLURM clusters.
* Evaluation outputs are automatically generated after inference completes.

# Repository Structure

All experiment outputs and logs are stored under:

```text
Experiments/
└── eval/
    └── <model-family-model-size>/
        └── <task-type>/              # score | pair | batch
            ├── reliability-diagram/
            ├── summary.json          # UQ evaluation metrics
            └── bias_summary.json     # Bias analysis metrics
```

## Directory Details

| File / Directory       | Description                                      |
| ---------------------- | ------------------------------------------------ |
| `reliability-diagram/` | Reliability calibration plots and visualizations |
| `summary.json`         | Aggregated uncertainty quantification metrics    |
| `bias_summary.json`    | Bias and fairness analysis outputs               |
| `score/`               | Point-wise scoring experiments                   |
| `pair/`                | Pairwise comparison experiments                  |
| `batch/`               | Batched evaluation experiments                   |

# Reproducing Results

1. Launch all model jobs using the provided SLURM scripts.
2. Wait for inference jobs to finish.
3. Run the evaluation pipeline.
4. Inspect generated metrics and calibration plots under `Experiments/eval/`.

# Output Convention

Results are organized hierarchically by:

1. **Model family**
2. **Model size**
3. **Task type**
4. **Evaluation artifact**

This layout is intended to simplify large-scale benchmarking and downstream analysis across multiple model configurations.
