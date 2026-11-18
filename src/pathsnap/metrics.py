"""Evaluation metrics for candidate classification and route reconstruction."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, accuracy_score


def candidate_metrics(y, p, thr=0.5):
    """Row-level (candidate) binary metrics. Macro F1 averages F1 of class 0 and class 1."""
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(
        accuracy=accuracy_score(y, yhat),
        precision_macro=precision_score(y, yhat, average="macro", zero_division=0),
        recall_macro=recall_score(y, yhat, average="macro", zero_division=0),
        f1_macro=f1_score(y, yhat, average="macro", zero_division=0),
        f1_pos=f1_score(y, yhat, zero_division=0),
        log_loss=log_loss(y, pc, labels=[0, 1]),
    )


def top1_predictions(cands: pd.DataFrame, score_col="p"):
    """For each (trip, pt) pick the candidate with highest score."""
    idx = cands.groupby(["trip", "pt"])[score_col].idxmax()
    return cands.loc[idx, ["trip", "pt", "arc", score_col]].rename(columns={"arc": "pred_arc"})


def top1_accuracy(points: pd.DataFrame, pred: pd.DataFrame):
    """Observation-level Top-1 accuracy over labelled points. Points with no candidates count as wrong."""
    m = points.merge(pred[["trip", "pt", "pred_arc"]], on=["trip", "pt"], how="left")
    return float((m.pred_arc == m.ref_arc).mean())


def route_continuity(pred_arcs, arcs):
    """Fraction of consecutive observation pairs whose predicted arcs are the same arc or
    directly connected in the directed graph (to_node of the first == from_node of the second).
    Pairs involving a missing prediction are ignored."""
    a = np.asarray(pred_arcs, dtype=float)
    ok = ~np.isnan(a[:-1]) & ~np.isnan(a[1:])
    a0 = a[:-1][ok].astype(int); a1 = a[1:][ok].astype(int)
    if len(a0) == 0:
        return np.nan
    same = a0 == a1
    conn = arcs[a0, 1] == arcs[a1, 0]
    return float((same | conn).mean())


def sequence_metrics(pred_arcs, ref_arcs, ref_route):
    """Route-level: precision/recall of the SET of distinct arcs vs reference route arcs."""
    p = set(int(x) for x in pred_arcs if not np.isnan(x))
    r = set(int(x) for x in ref_route)
    inter = len(p & r)
    prec = inter / max(len(p), 1); rec = inter / max(len(r), 1)
    return dict(route_arc_precision=prec, route_arc_recall=rec,
                route_arc_f1=0 if inter == 0 else 2 * prec * rec / (prec + rec))
