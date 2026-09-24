"""multi_interval_scan — the push into configurations with no closed form.

Each configuration is a dict
    {'label': str, 'intervals': [(start, length), ...], 'mass': m (0 = critical),
     'N': None (infinite chain) or ring size (massless only), 'dps': int or 'auto'}
and the scan returns, per configuration, a pure dict of arrays: the modular
Hamiltonian's precision diagnostics, locality score, per-interval BW profiles
('nn' and 'resummed'), inter-block coupling summaries (vacuum.modular.nonlocal_decay)
and — at criticality — the Casini-Huerta continuum predictions for the local
weight and for every pair's bilocal weight (vacuum.modular.casini_huerta), with
the interior relative deviations. Nothing is asserted here; the notebook in
papers/multi-interval-modular-hamiltonians/ turns these into the discovery
curves and the tests turn the anchors into assertions.

Precision policy ('auto'): start from dps = 0.8 n (1 + 0.8 m) + 30 (from
Eisler-Peschel 2017 Eq. (54) eps_max ~ 1.76 n at criticality, empirically ~1.8x
larger at m = 1), diagonalize, and rebuild C and h at more digits until the
spectrum is resolved with 15 digits to spare (info['sufficient']).
"""

from __future__ import annotations

import math

import numpy as np

from .casini_huerta import (
    ch_beta,
    ch_conjugate_points,
    ch_nonlocal_weight,
    intervals_from_parts,
)
from .fermionic import bw_profile, interval_modular, locality_score, nonlocal_decay
from .lattice import correlation_length, infinite_chain_C, massive_chain_C, ring_C

__all__ = ["multi_interval_scan", "modular_hamiltonian_auto", "sites_of_config"]

_LN10 = math.log(10.0)


def sites_of_config(intervals):
    """Site blocks [(start .. start+length-1), ...] of a configuration."""
    return [list(range(int(s), int(s) + int(L))) for s, L in intervals]


def _builder(mass, N):
    if mass > 0.0:
        if N is not None:
            raise ValueError("massive chains are supported on the infinite chain only (N=None)")
        return lambda sites, dps: massive_chain_C(sites, mass, dps=dps)
    if N is not None:
        return lambda sites, dps: ring_C(N, sites, dps=dps)
    return lambda sites, dps: infinite_chain_C(sites, dps=dps)


def modular_hamiltonian_auto(build, sites, mass=0.0, dps="auto", max_rounds=4):
    """h for the block `sites` with the precision raised until it is sufficient.

    `build(sites, dps)` must return the closed-form correlation matrix at `dps`
    digits. Returns (h, info, dps_used).
    """
    n = len(sites)
    if dps == "auto":
        dps = int(0.8 * n * (1.0 + 0.8 * float(mass))) + 30
    dps = int(dps)
    for _ in range(max_rounds):
        C = build(sites, dps)
        h, info = interval_modular(C, range(n), dps=dps, return_info=True)
        if info["sufficient"]:
            return h, info, dps
        dps = int(info["eps_max"] / _LN10) + 30
    raise RuntimeError(f"could not reach sufficient precision for {n} sites (last dps={dps})")


def multi_interval_scan(config_grid, include_h=False):
    """Run every configuration; return a list of report dicts (see module docstring)."""
    reports = []
    for cfg in config_grid:
        intervals = [(int(s), int(L)) for s, L in cfg["intervals"]]
        mass = float(cfg.get("mass", 0.0))
        N = cfg.get("N")
        parts = sites_of_config(intervals)
        sites = [s for P in parts for s in P]
        idx_parts, k = [], 0
        for P in parts:
            idx_parts.append(list(range(k, k + len(P))))
            k += len(P)
        h, info, dps_used = modular_hamiltonian_auto(_builder(mass, N), sites, mass, cfg.get("dps", "auto"))
        rep = {
            "label": cfg.get("label", ""),
            "intervals": intervals,
            "mass": mass,
            "xi": float(correlation_length(mass)) if mass > 0 else float("inf"),
            "N": N,
            "n_sites": len(sites),
            "dps": dps_used,
            "eps_max": info["eps_max"],
            "locality_nn": locality_score(h, bandwidth=1),
            "locality_5": locality_score(h, bandwidth=5),
            "nonlocal": nonlocal_decay(h, sites, idx_parts),
            "bw": [],
        }
        for P, I in zip(parts, idx_parts):
            blk = h[np.ix_(I, I)]
            x, b_nn = bw_profile(blk, mode="nn", offset=P[0])
            _, b_res = bw_profile(blk, mode="resummed", offset=P[0])
            rep["bw"].append({"x": x, "beta_nn": b_nn, "beta_resummed": b_res})
        if mass == 0.0:
            iv = intervals_from_parts(parts)
            for entry in rep["bw"]:
                entry["beta_ch"] = ch_beta(entry["x"], iv, N)
                entry["dev_resummed"] = entry["beta_resummed"] / entry["beta_ch"] - 1.0
                entry["dev_nn"] = entry["beta_nn"] / entry["beta_ch"] - 1.0
            for pair in rep["nonlocal"]["pairs"]:
                p, q = pair["p"], pair["q"]
                xp = np.asarray(parts[p], dtype=float) + 0.5
                xq = np.asarray(parts[q], dtype=float) + 0.5
                pair["ch_weight_pq"] = np.array([ch_nonlocal_weight(x, iv, q, N) for x in xp])
                pair["ch_weight_qp"] = np.array([ch_nonlocal_weight(x, iv, p, N) for x in xq])
                pair["ch_partner_pq"] = np.array([ch_conjugate_points(x, iv, N)[q] for x in xp])
                for key, xs, a, b in (("pq", xp, *iv[p]), ("qp", xq, *iv[q])):
                    inner = (xs > a + 0.2 * (b - a)) & (xs < b - 0.2 * (b - a))
                    dev = pair[f"row_weight_{key}"] / pair[f"ch_weight_{key}"] - 1.0
                    pair[f"dev_interior_{key}"] = float(np.max(np.abs(dev[inner]))) if np.any(inner) else float("nan")
        if include_h:
            rep["h"] = h
            rep["sites"] = sites
        reports.append(rep)
    return reports
