"""End-to-end prediction pipeline used by the evaluation scripts AND the API.

It reads only .track/.nodes/.arcs. It never opens the .route file and never imports pathsnap.labels,
so reference labels cannot leak into predictions.
"""
import json
import time
import joblib
import numpy as np

from . import config as C
from .data_io import load_trip
from .candidates import generate_candidates, build_features, FEATURES
from .hmm import decode_trip, stitch_route, RoadGraph


class Predictor:
    def __init__(self, model_dir=C.MODELS):
        self.cfg = json.load(open(model_dir / "inference_config.json"))
        self.model = joblib.load(model_dir / self.cfg["model_file"])
        assert self.cfg["features"] == FEATURES, "feature list mismatch between model and code"

    def predict_trip_dir(self, trip_dir):
        trip = load_trip(trip_dir, with_route=False)
        return self.predict_trip(trip)

    def predict_trip(self, trip):
        cfg = self.cfg
        timings = {}
        t0 = time.time()
        cands = generate_candidates(trip, radius=cfg["cand_radius_m"], k=cfg["cand_max_k"])
        feats = build_features(trip, cands)
        timings["candidates_features_s"] = time.time() - t0
        t0 = time.time()
        feats["p"] = self.model.predict_proba(feats[FEATURES].to_numpy(dtype=np.float32))[:, 1] if len(feats) else []
        timings["classifier_s"] = time.time() - t0
        n = len(trip.gps_xy)
        cls = np.full(n, np.nan)
        if len(feats):
            top = feats.loc[feats.groupby("pt").p.idxmax()]
            cls[top.pt.to_numpy()] = top.arc.to_numpy()
        t0 = time.time()
        d = cfg["decoder"]
        hmm = decode_trip(trip, feats[["pt", "arc", "t_proj", "p"]], beta=d["beta"], w_e=d["w_e"],
                          no_path_logp=d["no_path_logp"], gap_s=d["gap_s"])
        timings["hmm_s"] = time.time() - t0
        t0 = time.time()
        route = stitch_route(trip, hmm)
        timings["stitch_s"] = time.time() - t0
        return dict(trip=trip, classifier_arcs=cls, hmm_arcs=hmm, route_arcs=route,
                    n_candidates=int(len(feats)), timings=timings)
