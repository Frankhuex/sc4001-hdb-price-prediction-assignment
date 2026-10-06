# HDB price baseline

## Environment

Install the exact environment recorded in `uv.lock`:

```bash
uv sync
```

Run commands from the repository root. Python 3.12 is selected by
`.python-version`; dependencies and their exact versions are recorded in
`pyproject.toml` and `uv.lock`. The default seed is 42. Data preprocessing uses
2017--2020 for training and 2021 for validation, with statistics and category
maps fitted on training data only. The 2022 test set is reserved for final
evaluation.

## A1(b): Hyperparameter grid search

```bash
uv run python src/grid_search.py
```

Ray Tune runs all 25 combinations of hidden width `{16, 32, 64, 128, 256}`
and embedding dimension `{2, 4, 8, 12, 16}`. Each trial uses the same dimension
for all four categorical features and the same seed, learning rate, and batch
size from `src/config.py`. Training uses Adam, at most 20 epochs, early-stopping
patience 4, and minimum improvement `1e-4`. The selected model is the one with
the lowest validation RMSE reached during training; test results are not used
for selection.

Trials run sequentially on one CPU thread each by default. To run two trials
concurrently, if your machine has enough CPU and memory:

```bash
uv run python src/grid_search.py --max-concurrent-trials 2
```

Use `--cpus-per-trial N` to change the CPU allocation and PyTorch thread count
per trial. Local Ray workers reuse the Python interpreter and dependencies
already selected by `uv run`, without creating a separate virtual environment.
Ray stores detailed logs and each trial's final checkpoint beneath
`outputs/a1_b/trials/`. The checkpoint includes the best weights, full epoch
history, configuration, and scaling metadata. The main outputs are:

| File | Contents |
| --- | --- |
| `outputs/a1_b/search_results.csv` | All configurations, best validation RMSE, best epoch, number of epochs, and trial paths |
| `outputs/a1_b/validation_rmse_grid.csv` | Hidden-width-by-embedding-dimension table of best validation RMSE |
| `outputs/a1_b/selected/best_model.pt` | Selected model's weights at its best validation epoch |
| `outputs/a1_b/selected/history.csv` | Training and validation RMSE for every executed epoch, for A1(c) |
| `outputs/a1_b/selected/metadata.json` | Selected configuration, category-table sizes, scaling statistics, and best epoch/RMSE |

The metadata does not contain the category-to-ID maps; reproduce them with
`prepare_data` on the same CSV and training split when loading the weights.
The epoch training RMSE follows the supplied baseline's calculation over
training batches; validation RMSE uses the model at the end of each epoch.

## Original baseline

To run one experiment using the original configuration:

```bash
uv run python src/baseline.py
```

The input CSV and baseline output directory are constants at the top of
`src/baseline.py`. Data splits, features, model size, and training
hyperparameters are defined in `src/config.py`.

## Code map

| File | Responsibility |
| --- | --- |
| `src/config.py` | Experiment configuration |
| `src/data.py` | Data splits, preprocessing, and tensor conversion |
| `src/model.py` | Categorical embeddings and MLP |
| `src/baseline.py` | Shared training and RMSE evaluation, plus the single-run baseline entry point |
| `src/grid_search.py` | A1(b) Ray grid search and result export |
