# Everglades water-level forecasting benchmark: interactive website

An open web platform for the benchmark of task-specific deep learning models and a
time series foundation model on daily water levels in the Everglades. Visitors can
run eleven task-specific models and Chronos-Bolt with the same data, settings and
scoring as the paper, compare them, and see forecasts, without writing code or
needing a GPU. Every result shown on the site is computed by the site's own model
services.

## What you can do on the site

| Page | What it shows |
|---|---|
| **Evaluate** | Run one model at one horizon (7, 14, 21 or 28 days) on the test period, 2024-01-29 to 2024-08-26: observed vs. predicted water levels, MAE, RMSE, NSE, PBIAS and R² with Moriasi et al. (2015) ratings, per-station results, and a CSV download of the predictions |
| **Compare** | Run several models at once: a ranked table with training and single-forecast times, overlaid predictions per station, and accuracy across all four horizons |
| **Forecast** | Predict the 7-28 days after the last day in the data (2024-08-26), with the time one forecast takes |
| **Data** | All 37 daily series (water levels, structure flows, rainfall, evapotranspiration) with the training, validation and test periods marked |
| **About** | The evaluation protocol, every model's settings and the software versions in use |

## How it fits together

```
 Browser
    |
 web/  (Streamlit, lightweight env)          Evaluate, Compare, Forecast, Data, About
    |  HTTP
    +--> workers/nf_worker       :8001       NBEATS, NLinear, DLinear, KAN, PatchTST,
    |    (NeuralForecast env)                Informer, TimeMixer, TSMixer, TSMixerx,
    |                                        iTransformer, RMoK
    +--> workers/chronos_worker  :8002       Chronos-Bolt
         (AutoGluon/Chronos env)
                 |
               store/                        one folder per run: predictions, metrics,
                                             metadata, status
```

The two model families need incompatible library versions, so each runs as its own
service in its own environment; the website talks to both. Each service runs one job
at a time. A run's ID is a hash of the model, horizon, configuration version and the
data file's SHA-256, so each configuration is computed once and then served from the
store.

## Repository layout

| Path | What it is |
|---|---|
| `core/everglades_core/config.py` | The benchmark configuration: data path, 211/211 validation/test split, stations, horizons and every model's settings (`CONFIG_VERSION` changes whenever results would change) |
| `core/everglades_core/data.py` | Loads the data and makes the split; refuses to run if the data do not match the benchmark setup |
| `core/everglades_core/hydro_metrics.py` | MAE, RMSE, NSE, PBIAS, R² and Moriasi ratings |
| `core/everglades_core/scoring.py` | Per-station metrics on the five stations; overall = mean of the stations |
| `core/everglades_core/store.py`, `jobs.py`, `timing.py`, `service.py` | Results store, run execution, single-forecast timing and the shared web API |
| `workers/nf_worker/` | Service for the 11 task-specific models (NeuralForecast) |
| `workers/chronos_worker/` | Service for Chronos-Bolt (AutoGluon) |
| `web/` | The website |
| `scripts/` | Running models without the website, checking results against prediction files, exporting results |
| `tests/` | Automated tests |
| `data/final_concatenated_data.csv` | The benchmark dataset: 37 daily series, 2020-10-16 to 2024-08-26 |

## Evaluation protocol

- **Data and split:** 1,411 days; the last 211 days (2024-01-29 to 2024-08-26) are the
  test period and the 211 days before them the validation period.
- **Task-specific models:** trained with NeuralForecast on training + validation data
  (input 100 days, up to 1,000 steps, early stopping on the validation period), then
  evaluated with rolling cross-validation over the test period; each test date's
  prediction is the mean of the forecasts from every window that covers it.
- **Chronos-Bolt:** used zero-shot (`bolt_small`); for each day of the test period it
  forecasts from the preceding 100 + h days (107 to 128 days, h = horizon), following
  the evaluation code of the study.
- **Scoring:** metrics per station for the five target stations (NP205, P33, G620,
  NESRS1, NESRS2); the overall value is their mean.

Three times are recorded separately: **one forecast** (median of five calls after a
warm-up), **training** (task-specific models only, one-time) and **evaluation** (the
whole test-period run).

## Installation

Three Python 3.10 environments. Run each `pip install` from the folder holding the
requirements file; each lists the library versions the service was built with.

```bash
# 1. Task-specific model service
conda create -n eb-nf python=3.10 -y && conda activate eb-nf
cd workers/nf_worker && pip install -r requirements.txt && cd ../..

# 2. Chronos-Bolt service
conda create -n eb-chronos python=3.10 -y && conda activate eb-chronos
cd workers/chronos_worker && pip install -r requirements.txt && cd ../..

# 3. Website
conda create -n eb-web python=3.10 -y && conda activate eb-web
cd web && pip install -r requirements.txt && cd ..
```

## Running

Choose a folder for the results store, then start the three programs in separate
terminals:

```bash
export EVERGLADES_STORE=/path/to/everglades-store

# terminal 1: task-specific models
conda activate eb-nf
cd workers/nf_worker && uvicorn service:app --host 127.0.0.1 --port 8001

# terminal 2: Chronos-Bolt
conda activate eb-chronos
export USE_TF=0
cd workers/chronos_worker && uvicorn service:app --host 127.0.0.1 --port 8002

# terminal 3: website (run from web/ so the site's theme is used)
conda activate eb-web
cd web && streamlit run app.py --server.port 8501
```

Open http://localhost:8501. A GPU is recommended for running the models.

| Variable | Default | Used by |
|---|---|---|
| `EVERGLADES_STORE` | `./store` | services, scripts |
| `EVERGLADES_DATA` | `data/final_concatenated_data.csv` | everything |
| `NF_SERVICE_URL` | `http://localhost:8001` | website, scripts |
| `CHRONOS_SERVICE_URL` | `http://localhost:8002` | website, scripts |
| `ANONYMOUS` | `1` | website: hides paper and code links |

## Running models without the website

```bash
conda activate eb-nf      && python scripts/run_local.py --family nf --all
conda activate eb-chronos && export USE_TF=0 && python scripts/run_local.py --family chronos --all
conda activate eb-web     && python scripts/export_results.py      # all results in one CSV
```

Start with one quick model to check the setup:
`python scripts/run_local.py --family nf --model NLinear --horizon 7`.
These commands write to the same store the website reads, so they also fill it in
advance. `scripts/warm_cache.py` does the same through running services.

`scripts/parity_check.py` compares stored runs with a folder of prediction files
(`--paper1-root` with one `Everglades-Models-Results-{h}days` folder per horizon;
`--informer-dir` and `--chronos-dir` for those models) and reports, for every model
and horizon, whether all five metrics agree. Neural models trained on a GPU can differ
slightly across hardware and runs because of nondeterministic GPU operations.

## Tests

```bash
pip install pytest httpx fastapi      # in eb-web
python -m pytest tests
```

The tests check the data split and model settings, verify that the site's scoring
reproduces the published metrics exactly from the published Chronos-Bolt and Informer
predictions (`tests/fixtures`), and exercise the service API and timing.
`tests/stub_services.py` serves stand-in predictions for developing the website
without a GPU; it is not used for any reported result.
