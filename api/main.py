"""PathSnap FastAPI backend (Laptop 1).

Run:   uvicorn api.main:app --host 0.0.0.0 --port 8000      (from the repository root)
Env:   PATHSNAP_TRIPS_DIR  folder holding the evaluation trips (default data/raw/evaluator_test)

Endpoints
  GET  /health            liveness + loaded model info
  GET  /trips             list of available trip ids (ids only)
  POST /predict           {"trip_id": "00000012"} -> predicted route + server-side evaluation
"""
import os, sys, time, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import config as C
from pathsnap.pipeline import Predictor
from api.scoring import score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR", C.EVAL_DIR)).resolve()
app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = Predictor()
log.info("Loaded model %s | trips dir: %s", predictor.cfg["model_name"], TRIPS_DIR)


class PredictRequest(BaseModel):
    trip_id: str


def _trip_ids():
    if not TRIPS_DIR.exists():
        return []
    return sorted(p.name for p in TRIPS_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.cfg["model_name"], "decoder": predictor.cfg["decoder"],
            "trips_dir": str(TRIPS_DIR), "n_trips": len(_trip_ids())}


@app.get("/trips")
def trips(request: Request):
    ids = _trip_ids()
    log.info("GET /trips from %s -> %d trips", request.client.host if request.client else "?", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    ids = _trip_ids()
    if req.trip_id not in ids:
        raise HTTPException(404, f"unknown trip_id {req.trip_id}")
    client = request.client.host if request.client else "?"
    log.info("POST /predict trip=%s from %s : running pipeline", req.trip_id, client)
    t0 = time.time()
    trip_dir = TRIPS_DIR / req.trip_id
    pred = predictor.predict_trip_dir(trip_dir)          # uses .track/.nodes/.arcs only
    pred_s = time.time() - t0
    metrics = score(trip_dir, pred)                       # reference route read here, server-side only
    trip = pred["trip"]
    lonlat = trip.nodes
    route_coords = []
    for a in pred["route_arcs"]:
        f, t = trip.arcs[a]
        seg = [[float(lonlat[f, 1]), float(lonlat[f, 0])], [float(lonlat[t, 1]), float(lonlat[t, 0])]]
        if route_coords and route_coords[-1] == seg[0]:
            route_coords.append(seg[1])
        else:
            route_coords.extend(seg)
    stride = max(1, len(trip.track) // 3000)
    gps = [[float(la), float(lo)] for lo, la, _ in trip.track[::stride]]
    hmm = pred["hmm_arcs"]
    log.info("trip=%s points=%d candidates=%d predicted_arcs=%d top1_hmm=%s top1_cls=%s continuity=%s time=%.2fs",
             req.trip_id, len(trip.track), pred["n_candidates"], len(pred["route_arcs"]),
             f"{metrics.get('top1_accuracy_hmm', float('nan')):.4f}",
             f"{metrics.get('top1_accuracy_classifier_only', float('nan')):.4f}",
             f"{metrics.get('route_continuity_hmm', float('nan')):.4f}", time.time() - t0)
    return {
        "trip_id": req.trip_id,
        "model": predictor.cfg["model_name"],
        "n_gps_points": int(len(trip.track)),
        "n_points_matched": int(np.sum(~np.isnan(hmm))),
        "gps_points": gps,
        "predicted_route": route_coords,
        "predicted_route_arc_ids": [int(a) for a in pred["route_arcs"]],
        "timings_s": {**{k: round(v, 3) for k, v in pred["timings"].items()},
                      "prediction_total": round(pred_s, 3), "request_total": round(time.time() - t0, 3)},
        "evaluation": metrics,
    }
