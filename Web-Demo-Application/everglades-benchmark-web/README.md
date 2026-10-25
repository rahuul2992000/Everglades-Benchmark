# Everglades forecasting benchmark website

A website where visitors run eleven task-specific forecasting models and
Chronos-Bolt on Everglades water-level data, using the same data, settings,
and scoring as the benchmark paper. Every result shown on the site is computed
by the site's own model services.

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

The two model families need incompatible library versions, so each runs as
its own service in its own environment. The website talks to both.

`core/everglades_core` is shared by all three and has no ML dependencies:

| File | What it does |
|---|---|
| `config.py` | The paper configuration: data path, 211/211 split, stations, horizons, and every model's settings. Bump `CONFIG_VERSION` if you change anything. |
| `data.py` | Loads and melts the data and makes the split exactly as the notebooks do. Refuses to run if the data don't give 37 series and a test period of 2024-01-29 to 2024-08-26. |
| `hydro_metrics.py` | Your metrics module, unchanged. |
| `scoring.py` | `evaluate_by_station` on the five stations, overall = mean, Moriasi 2015 ratings (same as `NEWgof_all_models`). |
| `store.py`, `jobs.py` | Run IDs, on-disk results (predictions, metrics, metadata, and the trained model), and the run-validate-score-save sequence used by both services and the CLI. |
| `timing.py` | Single-forecast timing: one warm-up call, then the median of five. |
| `service.py` | The FastAPI app both services use. One job at a time per service. |

The runners are transcriptions of your notebooks:

- `workers/nf_worker/nf_runner.py`: build model, `nf.fit(train+val, val_size=211)`,
  `nf.cross_validation(Y_df, val_size=211, test_size=211, n_windows=None)`,
  `reset_index`, `groupby(['unique_id','ds']).mean()`.
  After cross-validation it saves the trained model (`nf.save`) and times a
  forecast from the latest data (`nf.predict(df=Y_df)`), as the notebooks do.
- `workers/chronos_worker/chronos_runner.py`: the loop from
  `FOllow-for-chronos-UPDATED-...-7-days copy 2.ipynb`, with `prediction_length`
  set to the horizon and a history window of 100 + horizon days (107, 114, 121, 128).

## Evaluate vs. Forecast

- **Evaluate** (and Compare) scores a model on the test period, 2024-01-29 to
  2024-08-26, against observed values.
- **Forecast** predicts the 7-28 days after 2024-08-26, the last day in the data.
  There are no observed values, so no metrics. Task-specific models use the model
  saved by their evaluation run, so each must be evaluated once before it can
  forecast; the page offers to do that. Chronos-Bolt is zero-shot.

Three times are recorded and shown separately: **one forecast** (median of five
after a warm-up, what a user waits for), **training** (one-time, task-specific
models only), and **evaluation** (reproducing the whole benchmark run).

A run's ID is a hash of model, horizon, `CONFIG_VERSION`, and the data file's
SHA-256, so each configuration is computed once and then served from the store.

## Setup

Three conda environments, all Python 3.10.

**Best option for the model environments: clone the environments that produced
the paper**, then add the service packages. This is how it was set up on diamond:

```bash
conda create -n eb-nf --clone py310 -y
conda create -n eb-chronos --clone /home/bear-b/users/rrang016/miniconda3/envs/chronos-bolt -y

conda activate eb-nf
pip uninstall -y torchvision torchaudio      # built for torch 2.6; break torch 2.5 imports
pip install "triton==3.1.0" fastapi uvicorn
pip install -e ~/everglades-benchmark-web/core

conda activate eb-chronos
pip install -U "typing_extensions>=4.14" fastapi uvicorn
pip install -e ~/everglades-benchmark-web/core
```

On diamond, AutoGluon 1.2 is installed in `~/.local/lib/python3.10/site-packages`,
which Python adds to every 3.10 environment. The model environments must see it,
and the website environment must not:

```bash
# model environments: always allow ~/.local, even when switching from eb-web
for e in eb-chronos eb-nf; do
  d=/home/bear-b/users/rrang016/miniconda3/envs/$e/etc/conda/activate.d
  mkdir -p "$d" && echo 'unset PYTHONNOUSERSITE' > "$d/allow_user_site.sh"
done
# website environment: ignore ~/.local
conda env config vars set PYTHONNOUSERSITE=1 -n eb-web
```

Set `export USE_TF=0` before running anything in `eb-chronos`, so the Hugging Face
libraries don't load TensorFlow.

If you can't clone, install from the requirements files instead. Run each
`pip install` from the folder that holds the requirements file (the `-e ../../core`
path is relative), and replace the pinned versions with the `pip freeze` output
from the environments that produced the paper.

```bash
# 1. NeuralForecast service
conda create -n eb-nf python=3.10 -y && conda activate eb-nf
cd workers/nf_worker && pip install -r requirements.txt && cd ../..

# 2. Chronos service
conda create -n eb-chronos python=3.10 -y && conda activate eb-chronos
cd workers/chronos_worker && pip install -r requirements.txt && cd ../..

# 3. Website
conda create -n eb-web python=3.10 -y && conda activate eb-web
cd web && pip install -r requirements.txt && cd ..
```

## Running

Set the store location once so every process shares it:

```bash
export EVERGLADES_STORE=/path/to/everglades-store
```

Then in three terminals:

```bash
# terminal 1
conda activate eb-nf
cd workers/nf_worker && CUDA_VISIBLE_DEVICES=0 uvicorn service:app --host 127.0.0.1 --port 8001

# terminal 2
conda activate eb-chronos
export USE_TF=0
cd workers/chronos_worker && CUDA_VISIBLE_DEVICES=0 uvicorn service:app --host 127.0.0.1 --port 8002

# terminal 3  (run from web/ so Streamlit picks up web/.streamlit/config.toml)
conda activate eb-web
cd web && streamlit run app.py --server.port 8501
```

Open http://localhost:8501.

Environment variables:

| Variable | Default | Used by |
|---|---|---|
| `EVERGLADES_STORE` | `./store` | services, scripts |
| `EVERGLADES_DATA` | `data/final_concatenated_data.csv` | everything |
| `NF_SERVICE_URL` | `http://localhost:8001` | website, warm_cache |
| `CHRONOS_SERVICE_URL` | `http://localhost:8002` | website, warm_cache |
| `ANONYMOUS` | `1` | website: hides paper and code links while under review |
| `PAPER_URL`, `CODE_URL` | empty | website About page, when `ANONYMOUS=0` |

## First run on the server: check parity

Before deploying, confirm the live runs reproduce the paper. Run everything
directly (no services needed), then compare with your published files:

```bash
conda activate eb-nf      && python scripts/run_local.py --family nf --all
conda activate eb-chronos && export USE_TF=0 && python scripts/run_local.py --family chronos --all

conda activate eb-web
python scripts/parity_check.py \
  --paper1-root /aul/homes/rrang016/ALL-MODEL-RESULTS-PAPER1/Paper1-ALL-MODELS-RESULTS \
  --informer-dir /aul/homes/rrang016/Documents
```

`parity_check.py` finds reference files with the same rules as
`NEWgof_all_models`, then reports for every model and horizon whether all five
metrics round to the same values as the published predictions, plus the
largest row-level difference. The full table goes to `parity_report.csv`.

Start with a single quick model to check the setup:

```bash
python scripts/run_local.py --family nf --model NLinear --horizon 7
```

Small differences for the neural models are possible across GPUs and library
versions; the report shows how large they are.

`run_local.py` writes to the same store the services read, so the runs above
also fill the cache. For a running deployment, `python scripts/warm_cache.py`
does the same through the services.

## Tests

```bash
pip install pytest httpx fastapi   # in eb-web
python -m pytest tests
```

- `test_scoring_parity.py` runs the published Chronos and Informer predictions
  through the site's scoring and checks every metric against
  `results_all_horizons.csv` at all four horizons.
- `test_data.py` checks the split and the model settings.
- `test_service.py` exercises the service API with a stub runner.
- `test_parity_script.py` checks the parity report end to end.
- `test_timing.py` checks the forecast timing helper.

`tests/stub_services.py` serves fake predictions on ports 8001 and 8002 so
the website can be developed without GPU libraries. It is for development
only; never deploy it.

## Deploying

- Run both services on the GPU machine, bound to 127.0.0.1, and put the
  website behind a reverse proxy (nginx or Caddy) with HTTPS. Only the website
  needs to be public.
- Each service runs one job at a time. The two services can run concurrently;
  Chronos-Bolt small needs little GPU memory.
- Warm the cache before a demo so every configuration opens instantly, and keep
  a copy of the store as an offline fallback for the conference.
