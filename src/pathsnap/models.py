"""Model zoo and dataset helpers shared by training, tuning and inference."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from . import config as C
from .candidates import FEATURES

# name -> (family, default params, max training rows (None = all))
MODEL_SPECS = {
    "LogisticRegression": ("Linear / probabilistic", dict(C=1.0, max_iter=500), None),
    "GaussianNB": ("Linear / probabilistic", dict(var_smoothing=1e-9), None),
    "KNN": ("Instance-based", dict(n_neighbors=25, weights="distance"), 80_000),
    "DecisionTree": ("Tree-based", dict(max_depth=10, min_samples_leaf=20), None),
    "RandomForest": ("Ensemble (bagging)", dict(n_estimators=100, max_depth=16, min_samples_leaf=5,
                                                 max_samples=0.3), None),
}


def make_model(name, params=None):
    p = dict(MODEL_SPECS[name][1]); p.update(params or {})
    s = C.SEED
    if name == "LogisticRegression":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=s, **p))
    if name == "GaussianNB":
        return GaussianNB(**p)
    if name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=2, **p))
    if name == "DecisionTree":
        return DecisionTreeClassifier(random_state=s, **p)
    if name == "RandomForest":
        return RandomForestClassifier(random_state=s, n_jobs=2, **p)
    raise KeyError(name)


def load_split(split, reliable_only=True):
    cands = pd.read_pickle(C.PROCESSED / f"{split}_cands.pkl")
    pts = pd.read_pickle(C.PROCESSED / f"{split}_points.pkl")
    if reliable_only:
        pts = pts[pts.ref_dist <= C.LABEL_RELIABLE_M]
        cands = cands.merge(pts[["trip", "pt"]], on=["trip", "pt"])
    return cands.reset_index(drop=True), pts.reset_index(drop=True)


def subsample_points(cands, pts, max_per_trip=C.MAX_POINTS_PER_TRIP, seed=C.SEED):
    rng = np.random.default_rng(seed)
    keep = []
    for tid, g in pts.groupby("trip"):
        idx = g.pt.to_numpy()
        if len(idx) > max_per_trip:
            idx = np.sort(rng.choice(idx, max_per_trip, replace=False))
        keep.append(pd.DataFrame({"trip": tid, "pt": idx}))
    keep = pd.concat(keep)
    return (cands.merge(keep, on=["trip", "pt"]).reset_index(drop=True),
            pts.merge(keep, on=["trip", "pt"]).reset_index(drop=True))


def X_of(df):
    return df[FEATURES].to_numpy(dtype=np.float32)
