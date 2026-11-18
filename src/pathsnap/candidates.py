"""Candidate generation and feature engineering.

For every GPS observation we find road arcs within CAND_RADIUS_M and keep the
K nearest. Each (observation, candidate arc) pair becomes one row; the learning
task is binary: "is this candidate the arc the vehicle was on?". At inference
the classifier probability is normalised over the candidates of an observation
and used as the HMM emission.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import config as C

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand",
    "heading_diff", "heading_cos", "heading_valid", "speed",
    "arc_len", "t_proj", "ctx_dist_mean", "ctx_dist_max", "ctx_heading_diff",
    "has_reverse_cand", "to_out_deg", "from_in_deg", "dt_prev", "dt_next",
]


def arc_geometry(trip):
    p0 = trip.node_xy[trip.arcs[:, 0]]
    p1 = trip.node_xy[trip.arcs[:, 1]]
    v = p1 - p0
    L = np.hypot(v[:, 0], v[:, 1])
    bearing = np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360  # 0 = north
    return p0, p1, v, L, bearing


def point_seg(px, py, p0, v, L):
    """Distance and clipped projection fraction of points onto segments (vectorised, row-wise)."""
    L2 = np.maximum(L * L, 1e-9)
    t = ((px - p0[:, 0]) * v[:, 0] + (py - p0[:, 1]) * v[:, 1]) / L2
    tc = np.clip(t, 0, 1)
    qx = p0[:, 0] + tc * v[:, 0]
    qy = p0[:, 1] + tc * v[:, 1]
    return np.hypot(px - qx, py - qy), tc


def gps_kinematics(trip, half=2, min_disp=3.0):
    """Heading (deg from north) and speed from a centred window of +-half points."""
    xy, t = trip.gps_xy, trip.track[:, 2]
    n = len(xy)
    i0 = np.clip(np.arange(n) - half, 0, n - 1)
    i1 = np.clip(np.arange(n) + half, 0, n - 1)
    d = xy[i1] - xy[i0]
    disp = np.hypot(d[:, 0], d[:, 1])
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    heading = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    valid = disp >= min_disp
    speed = disp / dt
    dt_prev = np.r_[0.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0.0]
    return heading, valid, speed, dt_prev, dt_next


def angdiff(a, b):
    d = np.abs((a - b + 180) % 360 - 180)
    return d


def generate_candidates(trip, radius=None, k=None):
    """Return DataFrame with one row per (point_idx, arc_id) candidate."""
    radius = radius or C.CAND_RADIUS_M
    k = k or C.CAND_MAX_K
    p0, p1, v, L, _ = arc_geometry(trip)
    # densify arcs for the spatial index
    nseg = np.maximum(np.ceil(L / C.SAMPLE_STEP_M).astype(int), 1)
    arc_rep = np.repeat(np.arange(len(L)), nseg + 1)
    frac = np.concatenate([np.linspace(0, 1, s + 1) for s in nseg])
    samples = p0[arc_rep] + frac[:, None] * v[arc_rep]
    tree = cKDTree(samples)
    hits = tree.query_ball_point(trip.gps_xy, r=radius + C.SAMPLE_STEP_M / 2 + 1)
    pi, ai = [], []
    for i, h in enumerate(hits):
        if h:
            a = np.unique(arc_rep[h])
            pi.append(np.full(len(a), i)); ai.append(a)
    if not pi:
        return pd.DataFrame(columns=["pt", "arc", "dist", "t_proj"])
    pi = np.concatenate(pi); ai = np.concatenate(ai)
    xy = trip.gps_xy[pi]
    dist, tp = point_seg(xy[:, 0], xy[:, 1], p0[ai], v[ai], L[ai])
    df = pd.DataFrame({"pt": pi, "arc": ai, "dist": dist, "t_proj": tp})
    df = df[df.dist <= radius]
    # tie-break identical distances (two-way arcs) deterministically, keep K nearest
    df = df.sort_values(["pt", "dist", "arc"], kind="stable")
    df["dist_rank"] = df.groupby("pt").cumcount()
    df = df[df.dist_rank < k].reset_index(drop=True)
    return df


def build_features(trip, cands, ctx=3):
    p0, p1, v, L, bearing = arc_geometry(trip)
    heading, hvalid, speed, dt_prev, dt_next = gps_kinematics(trip)
    df = cands.copy()
    pt = df.pt.to_numpy(); arc = df.arc.to_numpy()
    g = df.groupby("pt").dist
    df["n_cand"] = g.transform("size")
    dmin = g.transform("min")
    df["dist_minus_min"] = df.dist - dmin
    df["dist_ratio"] = (df.dist + 1) / (dmin + 1)
    hd = angdiff(heading[pt], bearing[arc])
    hv = hvalid[pt]
    df["heading_diff"] = np.where(hv, hd, 90.0)
    df["heading_cos"] = np.where(hv, np.cos(np.radians(hd)), 0.0)
    df["heading_valid"] = hv.astype(int)
    df["speed"] = speed[pt]
    df["arc_len"] = L[arc]
    # context: distances of neighbouring observations to the same arc
    n = len(trip.gps_xy)
    ds = []
    for off in range(-ctx, ctx + 1):
        if off == 0:
            continue
        j = np.clip(pt + off, 0, n - 1)
        d, _ = point_seg(trip.gps_xy[j, 0], trip.gps_xy[j, 1], p0[arc], v[arc], L[arc])
        ds.append(d)
    ds = np.stack(ds, 1)
    df["ctx_dist_mean"] = ds.mean(1)
    df["ctx_dist_max"] = ds.max(1)
    # heading over a wider window (more stable at low speed)
    h2, v2, *_ = gps_kinematics(trip, half=6)
    df["ctx_heading_diff"] = np.where(v2[pt], angdiff(h2[pt], bearing[arc]), 90.0)
    # does the opposite-direction arc also appear among this point's candidates?
    rev_key = dict()
    for a_id, (f, t) in enumerate(trip.arcs):
        rev_key[(f, t)] = a_id
    rev = np.array([rev_key.get((t, f), -1) for f, t in trip.arcs])
    pair = set(zip(pt.tolist(), arc.tolist()))
    df["has_reverse_cand"] = [int((p, r) in pair) for p, r in zip(pt.tolist(), rev[arc].tolist())]
    out_deg = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes))
    in_deg = np.bincount(trip.arcs[:, 1], minlength=len(trip.nodes))
    df["to_out_deg"] = out_deg[trip.arcs[arc, 1]]
    df["from_in_deg"] = in_deg[trip.arcs[arc, 0]]
    df["dt_prev"] = np.minimum(dt_prev[pt], 600)
    df["dt_next"] = np.minimum(dt_next[pt], 600)
    return df
