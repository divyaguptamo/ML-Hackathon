"""HMM + Viterbi route reconstruction.

Hidden state at observation i : one candidate arc of observation i (with its projection point).
Emission  log P(z_i | s)      : w_e * log(classifier probability of s, normalised over candidates of i).
Transition log P(s_j | s_i)   : -|d_route(s_i -> s_j) - d_gps(z_i, z_j)| / beta   (Newson & Krumm style)
    d_route = distance driven along the directed road graph from the projection on arc s_i to the
    projection on arc s_j (shortest path, Dijkstra with a cut-off).  If no path exists within the
    cut-off (disconnected / one-way violation) the transition gets a fixed log-penalty instead of -inf,
    so the decoder never dead-ends.
Gaps: if the time gap between consecutive observations exceeds GAP_BREAK_S the chain is split and each
    piece is decoded independently.  Observations without any candidate are skipped (no prediction).
"""
import heapq
import numpy as np
import networkx as nx

from . import config as C
from .candidates import arc_geometry


class RoadGraph:
    def __init__(self, trip):
        self.arcs = trip.arcs
        p0, p1, v, L, _ = arc_geometry(trip)
        self.L = L
        n = len(trip.nodes)
        order = np.argsort(trip.arcs[:, 0], kind="stable")
        self.out_arcs = order
        self.out_ptr = np.searchsorted(trip.arcs[order, 0], np.arange(n + 1))
        self._cache = {}

    def dists_from(self, src, cutoff):
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        dist = {src: 0.0}
        pq = [(0.0, src)]
        arcs, L, op, oa = self.arcs, self.L, self.out_ptr, self.out_arcs
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf) or d > cutoff:
                continue
            for k in range(op[u], op[u + 1]):
                a = oa[k]; w = arcs[a, 1]; nd = d + L[a]
                if nd < dist.get(w, np.inf) and nd <= cutoff:
                    dist[w] = nd; heapq.heappush(pq, (nd, w))
        self._cache[src] = (cutoff, dist)
        return dist


def viterbi_segment(steps, gps_xy, graph, beta, w_e, no_path_logp):
    """steps: list of (pt_idx, arcs[K], t_proj[K], logp[K]). Returns chosen arc per step."""
    arcs_all, L = graph.arcs, graph.L
    pt0, a0, t0, lp0 = steps[0]
    score = w_e * lp0
    back = []
    for (pi, ai, ti, _), (pj, aj, tj, lpj) in zip(steps[:-1], steps[1:]):
        dg = float(np.hypot(*(gps_xy[pj] - gps_xy[pi])))
        cutoff = min(3 * dg + 300.0, 5000.0)
        Ki, Kj = len(ai), len(aj)
        trans = np.full((Ki, Kj), no_path_logp, dtype=float)
        for x in range(Ki):
            a = ai[x]
            rest_a = (1 - ti[x]) * L[a]
            dd = graph.dists_from(int(arcs_all[a, 1]), cutoff)
            for y in range(Kj):
                b = aj[y]
                if a == b:
                    dr = abs(tj[y] - ti[x]) * L[a]
                else:
                    sp = dd.get(int(arcs_all[b, 0]))
                    if sp is None:
                        continue
                    dr = rest_a + sp + tj[y] * L[b]
                trans[x, y] = -abs(dr - dg) / beta
        tot = score[:, None] + trans
        bp = tot.argmax(0)
        score = tot[bp, np.arange(Kj)] + w_e * lpj
        back.append(bp)
    k = int(score.argmax())
    path = [k]
    for bp in reversed(back):
        k = int(bp[k]); path.append(k)
    path.reverse()
    return [steps[s][1][path[s]] for s in range(len(steps))]


def decode_trip(trip, cands, beta=10.0, w_e=1.0, no_path_logp=-30.0, gap_s=C.GAP_BREAK_S, graph=None):
    """cands: DataFrame for ONE trip with columns pt, arc, t_proj, p. Returns float array of arc per point (nan = none)."""
    n = len(trip.gps_xy)
    graph = graph or RoadGraph(trip)
    out = np.full(n, np.nan)
    if len(cands) == 0:
        return out
    c = cands.sort_values(["pt", "arc"])
    p = c.p.to_numpy()
    psum = c.groupby("pt").p.transform("sum").to_numpy()
    logp = np.log(np.clip(p / np.maximum(psum, 1e-12), 1e-6, 1.0))
    pts = c.pt.to_numpy(); arcs = c.arc.to_numpy(); tp = c.t_proj.to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(pts)) + 1, len(pts)]
    t = trip.track[:, 2]
    seg = []
    prev_pt = None
    for s, e in zip(starts[:-1], starts[1:]):
        i = int(pts[s])
        if prev_pt is not None and t[i] - t[prev_pt] > gap_s:
            _flush(seg, out, trip, graph, beta, w_e, no_path_logp); seg = []
        seg.append((i, arcs[s:e], tp[s:e], logp[s:e]))
        prev_pt = i
    _flush(seg, out, trip, graph, beta, w_e, no_path_logp)
    return out


def _flush(seg, out, trip, graph, beta, w_e, no_path_logp):
    if not seg:
        return
    chosen = viterbi_segment(seg, trip.gps_xy, graph, beta, w_e, no_path_logp)
    for (i, *_), a in zip(seg, chosen):
        out[i] = a


def stitch_route(trip, point_arcs, max_fill_m=3000.0):
    """Turn per-point arcs into an ordered, connected arc sequence by inserting shortest-path arcs
    between consecutive distinct predicted arcs that are not directly connected."""
    seq = []
    for a in point_arcs:
        if np.isnan(a):
            continue
        a = int(a)
        if not seq or seq[-1] != a:
            seq.append(a)
    if len(seq) < 2:
        return seq
    G = nx.DiGraph()
    p0, p1, v, L, _ = arc_geometry(trip)
    for k, (f, t) in enumerate(trip.arcs):
        if not G.has_edge(f, t) or G[f][t]["w"] > L[k]:
            G.add_edge(int(f), int(t), w=float(L[k]), arc=k)
    route = [seq[0]]
    for b in seq[1:]:
        a = route[-1]
        if trip.arcs[a, 1] != trip.arcs[b, 0]:
            try:
                d, path = nx.single_source_dijkstra(G, int(trip.arcs[a, 1]), int(trip.arcs[b, 0]),
                                                    cutoff=max_fill_m, weight="w")
                route.extend(G[u][w]["arc"] for u, w in zip(path[:-1], path[1:]))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        route.append(b)
    return route
