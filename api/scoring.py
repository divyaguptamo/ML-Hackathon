"""Server-side scoring. Runs AFTER prediction; reference data never reaches the model or the UI."""
import numpy as np

from pathsnap import config as C
from pathsnap.labels import align_points_to_route
from pathsnap.metrics import route_continuity, sequence_metrics


def score(trip_dir, pred):
    trip = pred["trip"]
    route_file = trip_dir / f"{trip_dir.name}.route"
    if not route_file.exists():
        return {"reference_available": False}
    trip.route = np.loadtxt(route_file, dtype=int, ndmin=1)
    ref_arc, _, ref_d = align_points_to_route(trip)
    rel = ref_d <= C.LABEL_RELIABLE_M
    cls, hmm = pred["classifier_arcs"], pred["hmm_arcs"]
    cls_route = [int(a) for i, a in enumerate(cls) if not np.isnan(a) and (i == 0 or a != cls[i - 1])]
    out = {
        "reference_available": True,
        "n_points": int(len(ref_arc)),
        "n_labelled_points": int(rel.sum()),
        "top1_accuracy_hmm": float((hmm == ref_arc)[rel].mean()),
        "top1_accuracy_classifier_only": float((cls == ref_arc)[rel].mean()),
        "route_continuity_hmm": route_continuity(hmm, trip.arcs),
        "route_continuity_classifier_only": route_continuity(cls, trip.arcs),
    }
    out.update({f"{k}_hmm": v for k, v in
                sequence_metrics(np.array(pred["route_arcs"], float), None, trip.route).items()})
    out.update({f"{k}_classifier_only": v for k, v in
                sequence_metrics(np.array(cls_route, float), None, trip.route).items()})
    trip.route = None
    return out
