"""Reading raw trip files and projecting coordinates to metres."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

EARTH_R = 6371008.8


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N,3) lon, lat, t
    nodes: np.ndarray            # (V,2) lon, lat
    arcs: np.ndarray             # (E,2) from, to (int)
    route: np.ndarray | None     # (M,) arc ids, None if not available
    lon0: float = 0.0
    lat0: float = 0.0
    gps_xy: np.ndarray = field(default=None, repr=False)
    node_xy: np.ndarray = field(default=None, repr=False)

    def project(self, lon, lat):
        """Local equirectangular projection (metres) centred on the trip.
        Error is < 0.1% over the ~10-30 km extent of a single trip, which is far
        below the GPS noise; it lets us use fast Euclidean geometry in metres."""
        k = np.cos(np.radians(self.lat0))
        x = np.radians(np.asarray(lon) - self.lon0) * EARTH_R * k
        y = np.radians(np.asarray(lat) - self.lat0) * EARTH_R
        return np.stack([x, y], axis=-1)


def _load(path: Path, dtype=float, cols=None):
    a = np.loadtxt(path, dtype=dtype, ndmin=2)
    if cols is not None and a.shape[1] != cols:
        raise ValueError(f"{path}: expected {cols} columns, got {a.shape[1]}")
    return a


def load_trip(trip_dir, with_route=True) -> Trip:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _load(trip_dir / f"{tid}.track", float, 3)
    nodes = _load(trip_dir / f"{tid}.nodes", float, 2)
    arcs = _load(trip_dir / f"{tid}.arcs", int, 2)
    route = None
    rp = trip_dir / f"{tid}.route"
    if with_route and rp.exists():
        route = np.loadtxt(rp, dtype=int, ndmin=1)
    # basic cleaning: drop non-finite rows, sort by time, drop duplicate timestamps
    ok = np.isfinite(track).all(axis=1)
    track = track[ok]
    track = track[np.argsort(track[:, 2], kind="stable")]
    _, first = np.unique(track[:, 2], return_index=True)
    track = track[np.sort(first)]
    t = Trip(tid, track, nodes, arcs, route)
    t.lon0 = float(np.median(track[:, 0]))
    t.lat0 = float(np.median(track[:, 1]))
    t.gps_xy = t.project(track[:, 0], track[:, 1])
    t.node_xy = t.project(nodes[:, 0], nodes[:, 1])
    return t


def list_trips(split_dir):
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []
    return sorted(p for p in split_dir.iterdir() if p.is_dir())
