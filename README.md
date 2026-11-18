# PathSnap: Map-Matching & Route Reconstruction

PathSnap takes noisy GPS points and matches each one to the road segment (arc) the vehicle was on. It then rebuilds a connected route.

Pipeline:
**candidate generation → candidate classifier (XGBoost, Bayesian-tuned) → HMM/Viterbi decoding → route stitching**,
served by a FastAPI backend (Laptop 1) to a browser UI (Laptop 2).

## Repository layout

| Folder | Contents |
|---|---|
| `src/pathsnap/` | Library: data loading and projection (`data_io.py`), candidate generation and features (`candidates.py`), label alignment (`labels.py`), model zoo (`models.py`), HMM/Viterbi (`hmm.py`), metrics (`metrics.py`), inference pipeline (`pipeline.py`), config (`config.py`) |
| `notebooks/` | Numbered, reproducible Jupyter notebooks: `01_eda.ipynb` … `08_make_report.ipynb` |
| `api/` | FastAPI backend (`main.py`) and server-side scoring (`scoring.py`) |
| `ui/` | Evaluator UI (`index.html`, Leaflet) |
| `models/` | Exported final model (`final_model.joblib`) + `inference_config.json` |
| `results/` | CSV/JSON experiment results, Optuna trials, logs |
| `figures/` | All plots used in the report |
| `maps/` | Interactive Folium maps (HTML) |
| `report/` | Technical report (PDF) |
| `data/` | See `data/README.md` (raw data not committed) |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Put the raw data in `data/raw/` as described in `data/README.md`.

## Reproducing the experiments

Install Jupyter (`pip install notebook`), then run `jupyter notebook` **from the `notebooks/` folder** and run the
notebooks in this order (Kernel → Restart & Run All):

| Notebook | What it does | Time |
|---|---|---|
| `02_build_dataset.ipynb` | Candidates, features, labels (set `RUN_SWEEP = True` for the radius sweep) | ~1 min (+5) |
| `01_eda.ipynb` | EDA tables, plots, Folium maps | ~1 min |
| `03_train_compare.ipynb` | 8 models: 5-fold GroupKFold CV + student test | ~15 min |
| `04_bayes_opt.ipynb` | Optuna TPE search (XGBoost, 30 trials) + tuned vs untuned | ~15 min |
| `05_hmm_eval.ipynb` | Decoder tuning, final model export, classifier vs HMM | ~8 min |
| `06_maps.ipynb` | Folium maps of predictions | <1 min |
| `07_plots.ipynb` | Comparison figures | <1 min |
| `08_make_report.ipynb` | Builds `report/PathSnap_Technical_Report.pdf` from the result files | <1 min |

The reusable code (loading, candidates, features, labels, models, HMM, metrics, inference pipeline) lives in
`src/pathsnap/` as Python modules, because the notebooks and the FastAPI server both import it.
The random seed is fixed in `src/pathsnap/config.py` (`SEED = 42`).

## Evaluation-day deployment

### Laptop 1: model + API
1. Copy the evaluator trips into `data/raw/evaluator_test/<trip_id>/...`. Alternatively, point the API at any folder with the
   `PATHSNAP_TRIPS_DIR` environment variable.
2. Start the API from the repository root:
   ```bash
   # Windows (cmd):  set PATHSNAP_TRIPS_DIR=data\raw\evaluator_test
   # Linux/macOS:    export PATHSNAP_TRIPS_DIR=data/raw/evaluator_test
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
3. Find Laptop 1's LAN IP (`ipconfig` / `ip addr`) and allow port 8000 through the firewall.
   Every request is logged in this terminal (trip, points, accuracy, timing). Show these logs to the evaluator.

Rehearsal without evaluator data: `set PATHSNAP_TRIPS_DIR=data\raw\student_test`.

### Laptop 2: UI
```bash
cd ui
python -m http.server 5500
```
Open `http://localhost:5500/?api=http://<LAPTOP1_IP>:8000`, or type the API URL and press **Connect**.
The UI lists the trips. Click one, and the UI sends the trip id to `POST /predict`, then shows the predicted route and the
server-computed evaluation. The UI loads Leaflet and OSM tiles from the internet.

### API reference
```bash
curl http://<LAPTOP1_IP>:8000/health
curl http://<LAPTOP1_IP>:8000/trips
curl -X POST http://<LAPTOP1_IP>:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\": \"00000040\"}"
```
The response contains GPS points, the predicted route polyline, the predicted arc ids, timings and `evaluation`
(Top-1 accuracy, route continuity, route arc precision/recall/F1). Reference routes are never returned.

## Leakage safeguards
- `pipeline.py` reads only `.track/.nodes/.arcs` and never imports `labels.py`.
- `.route` files are read only for training labels (train), development scoring (student test) and
  `api/scoring.py` (after prediction).
- No notebook reads `data/raw/evaluator_test`. Model selection, the Optuna search and decoder tuning use only train and student-test data.
- Splits are trip-level, and CV uses GroupKFold by trip.

## AI assistance disclosure
The code in this repository was written with the help of an AI assistant but interpreted and understood by all. The results came by running the notebook 
and training the model
