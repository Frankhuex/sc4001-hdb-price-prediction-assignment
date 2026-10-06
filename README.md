# HDB price baseline

## Run it

Install the exact environment recorded in `uv.lock`:

```bash
uv sync
```

Run the baseline with:

```bash
uv run python src/baseline.py
```

Run the command from the repository root. The input CSV and output directory are constants at the top of `src/baseline.py`, while the data split, features, model size, and training hyperparameters are defined in `src/config.py`.

## Code map

```text
src/config.py       run configuration
src/data.py         split, fit preprocessing, tensor conversion
src/model.py        categorical embeddings and MLP
src/baseline.py     single-run training, RMSE evaluation, and model export
```
