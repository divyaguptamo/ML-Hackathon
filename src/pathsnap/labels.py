"""Per-observation labels derived from the reference route.

The .route file lists the arcs traversed in order; it does NOT label GPS points.
We align points to route positions with a monotonic dynamic programme:
point i is assigned route position j(i) with j non-decreasing in i, minimising
the sum of (clipped) point-to-arc distances. Monotonicity stops a point from
being snapped to a later/earlier pass of the route that happens to be nearby
(e.g. on a two-way street or a loop).

This module is used ONLY for training labels and server-side scoring; it is
never imported by the prediction path.
"""
import numpy as np

from . import config as C
from .candidates import arc_geometry, point_seg


def align_points_to_route(trip):
    route = trip.route
    p0, p1, v, L, _ = arc_geometry(trip)
    xy = trip.gps_xy
    N, M = len(xy), len(route)
    D = np.empty((N, M), dtype=np.float32)
    for j, a in enumerate(route):
        d, _ = point_seg(xy[:, 0], xy[:, 1], np.repeat(p0[a][None], N, 0),
                         np.repeat(v[a][None], N, 0), np.repeat(L[a], N))
        D[:, j] = d
    cost = np.minimum(D, C.LABEL_CLIP_M)
    acc = np.empty_like(cost)
    acc[0] = cost[0]
    arg = np.empty((N, M), dtype=np.int32)
    for i in range(1, N):
        prev = acc[i - 1]
        # running min over k <= j, with argmin
        cm = np.minimum.accumulate(prev)
        idx = np.arange(M)
        is_new = prev <= cm
        am = np.maximum.accumulate(np.where(is_new, idx, 0))
        acc[i] = cost[i] + cm
        arg[i] = am
    j = np.empty(N, dtype=int)
    j[-1] = int(np.argmin(acc[-1]))
    for i in range(N - 1, 0, -1):
        j[i - 1] = arg[i, j[i]]
    lab_arc = route[j]
    lab_dist = D[np.arange(N), j]
    return lab_arc, j, lab_dist
