"""papers/interacting-vacua-first-numbers — re-runnable notebook.

Candidate claim (first numbers, PLAN.md Layer 7 leap L5)
--------------------------------------------------------
For the lattice lambda-phi^4 chain in its symmetric phase, DMRG ground
states with a truncated local oscillator basis give (i) how interactions
change the entanglement structure of the vacuum — the mutual-information
decay length, the exact small-block negativity and its sudden death, and a
Gaussian-covariance proxy for large-block negativity decay — and (ii) the
first interacting quantum-energy-inequality number: the smeared local
energy along a worldline, minimized over a stated family of locally
squeezed MPS by real-time TEBD (a variational UPPER bound on the infimum),
compared with the free theory's exact infimum at the same smearing, mass
and lattice (``vacuum.inequalities.qei``), and with the free infimum at the
Hartree-renormalized mass (the "mass renormalization only" line).

Run
---
    .venv/bin/python papers/interacting-vacua-first-numbers/notebook.py
    .venv/bin/python papers/interacting-vacua-first-numbers/notebook.py --quick  # smoke, ~1-2 min

``--quick`` (equivalently ``NOTEBOOK_SMOKE=1``) runs every section at smoke
size (L = 12, n_max = 5, two couplings) and writes to a fresh temporary
directory (``--out DIR`` or ``NOTEBOOK_DATA_DIR`` override it), so a smoke
run never overwrites the archive under ``data/``.

Top to bottom, headless, no hidden state; every array behind every number
is written to ``data/`` by the section that computed it (``_source`` field
in the JSON files). Compute budget <= 25 min on a laptop (the wall time is
printed and recorded in MANIFEST.md). Every reported number carries chi,
n_max, and the truncation error of the run that produced it, and is
re-evaluated at 2 chi.

Where the phase transition is
-----------------------------
With H = (1/2) sum [pi^2 + m^2 phi^2 + (dphi)^2] + (lambda/4!) sum phi^4 and
bare m^2 > 0 the chain is in the symmetric phase for every lambda >= 0: the
transition needs a negative bare mass squared, and in renormalized units it
sits at lambda/mu_R^2 ~ 65 (Milsted-Haegeman-Osborne PRD 88, 085030
(2013) quote ~66 in this lambda/4! normalization; Kadoh et al., JHEP 05
(2019) 184, 10.913(56) in the lambda/4 normalization, x6 = 65.5). The
sweep below reaches lambda/mu_H^2 <= ~6 (Hartree mass), i.e. a tenth of
the critical ratio, and the Hartree mass is recorded per point.

Structure
---------
S0  build provenance
S1  lambda = 0 anchor at paper resolution (L = 32; n_max sweep)
S2  the lambda sweep: energies vs perturbation theory, Hartree mass and
    distance from criticality, mutual-information decay length, exact
    small-block negativity (sudden death), Gaussian-proxy large-block
    negativity decay length, chi-doubling of everything, n_max checks
S3  the QEI number: variational smeared-energy minimum vs the free exact
    infimum, per lambda, with chi-doubling and n_max re-evaluation
S4  audits: passivity of every interacting ground state; energy-density
    closure; the anomaly-protocol verdict on the QEI comparison

Nothing here modifies any vacuum.* module; the notebook is a pure consumer
of vacuum.core / vacuum.inequalities / vacuum.interacting.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import platform
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="unit_cell_width")
logging.getLogger("tenpy").setLevel(logging.WARNING)

from vacuum.core import (  # noqa: E402
    entropy,
    ground_state_cov,
    log_negativity,
    mean_energy,
    mutual_information,
    reduce as reduce_cov,
)
from vacuum.inequalities.qei import gaussian_f, qei_bound_2d, qei_minimize, sampling_operator  # noqa: E402
import vacuum  # noqa: E402
from vacuum.interacting import (  # noqa: E402
    CRITICAL_RATIO_LAMBDA_OVER_MU2,
    apply_squeeze_family,
    block_entropies,
    covariance_from_mps,
    energy_density_profile,
    free_coupling_matrix,
    gaussian_proxy_mutual_information,
    gaussian_proxy_negativity,
    hartree_mass_squared,
    log_negativity_mps,
    mps_energy,
    mutual_information_mps,
    passivity_audit_mps,
    perturbative_energy,
    phi4_ground_mps,
    qei_variational_mps,
    relative_change,
    smeared_energy_mps,
)
import os  # noqa: E402
import argparse  # noqa: E402
import tempfile  # noqa: E402

_ap = argparse.ArgumentParser(description="interacting-vacua-first-numbers")
_ap.add_argument("--quick", action="store_true",
                 help="smoke configuration (same as NOTEBOOK_SMOKE=1): L = 12, n_max = 5, two couplings")
_ap.add_argument("--out", default=None,
                 help="output directory (default: data/; a fresh temporary directory for --quick)")
_args, _ = _ap.parse_known_args()

HERE = Path(__file__).resolve().parent
SMOKE = os.environ.get("NOTEBOOK_SMOKE", "") == "1" or bool(_args.quick)   # tiny parameters, scratch output (CI of the notebook itself)
if _args.out:
    DATA = Path(_args.out)
elif os.environ.get("NOTEBOOK_DATA_DIR"):
    DATA = Path(os.environ["NOTEBOOK_DATA_DIR"])
elif SMOKE:
    DATA = Path(tempfile.mkdtemp(prefix="interacting-vacua-quick-"))
else:
    DATA = HERE / "data"
DATA.mkdir(exist_ok=True, parents=True)

# ---- parameters (documented tuning) ---------------------------------------
MASS = 0.5                 # correlation length ~2 sites; local nbar ~ 0.054
L = 32                     # sweep chain
NMAX = 8                   # main cutoff (lambda = 0: dV ~ 1e-8)
NMAX_DENSE = 6             # cutoff for 2+2-site dense negativities (d^4 = 2401)
CHI = 32                   # bond dimension; everything re-run at 64
LAMS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
MI_SEPS = list(range(0, 9))
PROXY_BLOCK = 8
PROXY_SEPS = list(range(0, 15))
# QEI
QEI_L = 16
QEI_NMAX = 8
QEI_CHI = 16
TAU0 = 0.75                # sampling width (lattice units); t_max = 4 tau0 = 3
QEI_HW = 2                 # 5-site window, 9 parameters
QEI_DT = 0.2               # 4th-order Trotter; Simpson dev vs exact < 1e-7 (S3 check)
QEI_ORDER = 4
S_GRID = (0.5, 0.75, 1.0, 1.25, 1.5)
if SMOKE:
    L, NMAX, CHI, LAMS = 12, 5, 16, [0.0, 1.0]
    MI_SEPS, PROXY_SEPS, PROXY_BLOCK = list(range(0, 5)), list(range(0, 5)), 3
    QEI_L, QEI_NMAX, QEI_CHI, QEI_HW, S_GRID = 10, 5, 8, 1, (0.8, 1.0, 1.2)

T_START = time.time()
TIMES = {}


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _dump(name, obj):
    with open(DATA / name, "w") as fh:
        json.dump(obj, fh, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")


def _csv(name, rows, fields):
    with open(DATA / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _section(label):
    print(f"\n=== {label} ===", flush=True)


# ---------------------------------------------------------------------------
# S0  provenance
# ---------------------------------------------------------------------------
_section("S0 provenance")
try:
    build = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
except Exception as exc:  # pragma: no cover
    build = f"git unavailable: {exc}"
import tenpy  # noqa: E402

build_info = {
    "_source": "notebook.py::S0",
    "smoke_mode": SMOKE,
    "date_utc": _now(),
    "build_git_head": build,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "numpy": np.__version__,
    "tenpy": tenpy.__version__,
    "vacuum_version": vacuum.__version__,
    "parameters": {
        "MASS": MASS, "L": L, "NMAX": NMAX, "NMAX_DENSE": NMAX_DENSE, "CHI": CHI, "LAMS": LAMS,
        "QEI": {"L": QEI_L, "n_max": QEI_NMAX, "chi": QEI_CHI, "tau0": TAU0, "half_width": QEI_HW,
                "dt": QEI_DT, "order": QEI_ORDER, "s_grid": list(S_GRID)},
        "critical_ratio_lambda_over_mu2": CRITICAL_RATIO_LAMBDA_OVER_MU2,
    },
}
_dump("build_info.json", build_info)
print(json.dumps(build_info, indent=1))

# ---------------------------------------------------------------------------
# S1  lambda = 0 anchor at paper resolution
# ---------------------------------------------------------------------------
_section("S1 lambda = 0 anchor, L = 32")
t0 = time.time()
K0 = free_coupling_matrix(L, MASS)
V0 = ground_state_cov(K0)
E0_exact = mean_energy(V0, K0)
c = L // 2
S_exact = np.array([entropy(reduce_cov(V0, range(l))) for l in range(1, L)])
anchor_blocks = {
    "EN_1+1_d0": ([c - 1], [c]),
    "EN_2+2_d0": ([c - 2, c - 1], [c, c + 1]),
    "EN_2+2_d1": ([c - 2, c - 1], [c + 1, c + 2]),
    "MI_1+1_d3": ([c - 2], [c + 1]),
    "MI_1+1_d6": ([c - 3], [c + 3]),
}
s1_rows = []
for n_max in ((4, 5) if SMOKE else (4, 6, 8)):
    gs = phi4_ground_mps(L, MASS, 0.0, n_max=n_max, chi_max=CHI)
    Vm = covariance_from_mps(gs.psi)
    row = {"n_max": n_max, "chi": gs.info["chi"], "trunc_err": gs.info["max_trunc_err"],
           "dE": gs.E0 - E0_exact, "dV_max": float(np.max(np.abs(Vm - V0))),
           "dS_max": float(np.max(np.abs(block_entropies(gs.psi) - S_exact)))}
    for name, (A, B) in anchor_blocks.items():
        if len(A) + len(B) > 2 and n_max > NMAX_DENSE and name != "EN_2+2_d0":
            continue  # only the slowest-converging 4-site number is pushed to n_max = 8 (d^4 = 6561)
        if name.startswith("MI"):
            row["d" + name] = mutual_information_mps(gs.psi, A, B) - mutual_information(V0, A, B)
        else:
            row["d" + name] = log_negativity_mps(gs.psi, A, B) - log_negativity(V0, A, B)
    s1_rows.append(row)
    print(row)
_csv("s1_free_anchor.csv", s1_rows, ["n_max", "chi", "trunc_err", "dE", "dV_max", "dS_max"]
     + ["d" + k for k in anchor_blocks])
TIMES["S1"] = time.time() - t0

# ---------------------------------------------------------------------------
# S2  the lambda sweep
# ---------------------------------------------------------------------------
_section("S2 lambda sweep")
t0 = time.time()


def _mi_profile(psi, seps):
    out = []
    for d in seps:
        a = c - 1 - d // 2
        out.append(mutual_information_mps(psi, [a], [a + 1 + d]))
    return np.array(out)


def _proxy_profiles(V, seps, block):
    en, mi = [], []
    for d in seps:
        a0 = (L - 2 * block - d) // 2
        A = list(range(a0, a0 + block))
        B = list(range(a0 + block + d, a0 + 2 * block + d))
        en.append(gaussian_proxy_negativity(V, A, B))
        mi.append(gaussian_proxy_mutual_information(V, A, B))
    return np.array(en), np.array(mi)


def _decay_fit(d, y, mask):
    """ln y = a - d/xi on the masked points; returns (xi, a, max residual)."""
    d = np.asarray(d, float)[mask]
    ly = np.log(np.asarray(y, float)[mask])
    if d.size < 3:
        return float("nan"), float("nan"), float("nan"), int(d.size)
    slope, a = np.polyfit(d, ly, 1)
    resid = ly - (slope * d + a)
    return -1.0 / slope, a, float(np.max(np.abs(resid))), int(d.size)


def _lattice_xi(mu2):
    """Correlation length of a free lattice field of mass^2 mu2: cosh(1/xi) = 1 + mu2/2."""
    return 1.0 / math.acosh(1.0 + 0.5 * mu2)


sweep_rows = []
mi_rows = []
proxy_rows = []
fits = {}
chi_doubling = {}
cov_store = {}
for lam in LAMS:
    tl = time.time()
    gs = phi4_ground_mps(L, MASS, lam, n_max=NMAX, chi_max=CHI)
    gs2 = phi4_ground_mps(L, MASS, lam, n_max=NMAX, chi_max=2 * CHI, psi0=gs.psi)
    gsd = phi4_ground_mps(L, MASS, lam, n_max=NMAX_DENSE, chi_max=CHI)
    gsd2 = phi4_ground_mps(L, MASS, lam, n_max=NMAX_DENSE, chi_max=2 * CHI, psi0=gsd.psi)
    mu2 = hartree_mass_squared(L, MASS, lam)
    pt = perturbative_energy(L, MASS, lam) if lam <= 1.0 else None
    numbers = {}
    numbers["E0"] = (gs.E0, gs2.E0)
    S_a, S_b = block_entropies(gs.psi), block_entropies(gs2.psi)
    numbers["S_half"] = (S_a[c - 1], S_b[c - 1])
    numbers["EN_1+1_d0"] = (log_negativity_mps(gs.psi, [c - 1], [c]), log_negativity_mps(gs2.psi, [c - 1], [c]))
    for d in (0, 1, 2):
        A = [c - 2, c - 1]
        B = [c + d, c + d + 1]
        numbers[f"EN_2+2_d{d}"] = (log_negativity_mps(gsd.psi, A, B), log_negativity_mps(gsd2.psi, A, B))
    numbers["MI_2+2_d1"] = (mutual_information_mps(gsd.psi, [c - 2, c - 1], [c + 1, c + 2]),
                            mutual_information_mps(gsd2.psi, [c - 2, c - 1], [c + 1, c + 2]))
    mi_a, mi_b = _mi_profile(gs.psi, MI_SEPS), _mi_profile(gs2.psi, MI_SEPS)
    V_a, V_b = covariance_from_mps(gs.psi), covariance_from_mps(gs2.psi)
    cov_store[f"V_lam{lam:g}"] = V_a
    pen_a, pmi_a = _proxy_profiles(V_a, PROXY_SEPS, PROXY_BLOCK)
    pen_b, pmi_b = _proxy_profiles(V_b, PROXY_SEPS, PROXY_BLOCK)
    for d, a, b in zip(MI_SEPS, mi_a, mi_b):
        numbers[f"MI_1+1_d{d}"] = (a, b)
    for d, a, b in zip(PROXY_SEPS, pen_a, pen_b):
        if a > 1e-10:
            numbers[f"proxyEN_8+8_d{d}"] = (a, b)
    # chi-doubling gate: relative change of every reported number above an
    # absolute resolution floor of 1e-8 (numbers below it — e.g. a proxy
    # negativity at the edge of sudden death, or MI at the largest separation
    # — are listed separately: their relative change is roundoff, not physics)
    # Two floors. General numbers: 1e-8. Gaussian-proxy negativities: 1e-5 —
    # a proxy E_N below 1e-5 sits within ~1e-7 of sudden death, where the
    # float64 partial-transpose route amplifies the 1e-9 covariance
    # differences between chi = 32 and 64 into O(10%) relative changes
    # (measured: 0.12 at lambda = 0, d = 13, E_N = 7.5e-7, absolute change
    # 9e-8). Absolute changes are recorded for every number regardless.
    RES_FLOOR, PROXY_FLOOR = 1e-8, 1e-5
    def _floor(k):
        return PROXY_FLOOR if k.startswith("proxyEN") else RES_FLOOR
    rel = {k: relative_change(a, b) for k, (a, b) in numbers.items() if max(abs(a), abs(b)) >= _floor(k)}
    absd = {k: abs(a - b) for k, (a, b) in numbers.items()}
    below = {k: (a, b) for k, (a, b) in numbers.items() if max(abs(a), abs(b)) < _floor(k)}
    chi_doubling[f"lam={lam:g}"] = {"chi": (gs.info["chi"], gs2.info["chi"]),
                                    "trunc_err": (gs.info["max_trunc_err"], gs2.info["max_trunc_err"]),
                                    "relative_changes": rel, "worst": max(rel.values()),
                                    "absolute_changes": absd, "worst_absolute": max(absd.values()),
                                    "below_resolution_floor": below,
                                    "resolution_floor": RES_FLOOR, "proxy_floor": PROXY_FLOOR}
    # fits: MI decay length (1+1 blocks, d = 2..8 where MI > 1e-9)
    mask_mi = (np.array(MI_SEPS) >= 2) & (mi_a > 1e-9)
    xi_mi, a_mi, r_mi, n_mi = _decay_fit(MI_SEPS, mi_a, mask_mi)
    xi_mi2, _, _, _ = _decay_fit(MI_SEPS, mi_b, mask_mi)
    # proxy negativity: live points d >= 1 before death (threshold 1e-10, float64 route)
    live = pen_a > 1e-10
    death = int(PROXY_SEPS[int(np.argmin(live))]) if not live.all() else -1
    mask_en = live & (np.array(PROXY_SEPS) >= 1)
    xi_en, a_en, r_en, n_en = _decay_fit(PROXY_SEPS, pen_a, mask_en)
    xi_en2, _, _, _ = _decay_fit(PROXY_SEPS, pen_b, mask_en)
    xi_H = _lattice_xi(mu2)
    fits[f"lam={lam:g}"] = {
        "xi_MI": xi_mi, "xi_MI_2chi": xi_mi2, "MI_fit_max_resid": r_mi, "MI_fit_points": n_mi,
        "xi_MI_over_half_xi_H": xi_mi / (0.5 * xi_H),
        "xi_proxyEN": xi_en, "xi_proxyEN_2chi": xi_en2, "proxyEN_fit_max_resid": r_en,
        "proxyEN_fit_points": n_en, "proxyEN_death_separation": death,
        "xi_hartree_lattice": xi_H, "xi_free_lattice": _lattice_xi(MASS**2),
    }
    row = {
        "lam": lam, "L": L, "n_max": NMAX, "chi": gs.info["chi"], "chi2": gs2.info["chi"],
        "trunc_err": gs.info["max_trunc_err"], "trunc_err_2chi": gs2.info["max_trunc_err"],
        "sweeps": gs.info["sweeps"], "E0": gs.E0, "E0_2chi": gs2.E0,
        "E_PT2": pt["total"] if pt else "", "E0_minus_PT2": (gs.E0 - pt["total"]) if pt else "",
        "E0_minus_PT1": (gs.E0 - pt["E0"] - lam * pt["E1"]) if pt else "",
        "mu_H2": mu2, "lam_over_mu_H2": lam / mu2 if lam > 0 else 0.0,
        "lam_over_mu_H2_over_critical": (lam / mu2) / CRITICAL_RATIO_LAMBDA_OVER_MU2 if lam > 0 else 0.0,
        "phi2_center": V_a[c, c], "pi2_center": V_a[L + c, L + c],
        "S_half": S_a[c - 1],
        "EN_1+1_d0": numbers["EN_1+1_d0"][0],
        "EN_2+2_d0": numbers["EN_2+2_d0"][0], "EN_2+2_d1": numbers["EN_2+2_d1"][0],
        "EN_2+2_d2": numbers["EN_2+2_d2"][0], "EN_2+2_nmax": NMAX_DENSE,
        "MI_2+2_d1": numbers["MI_2+2_d1"][0],
        "xi_MI": xi_mi, "xi_proxyEN": xi_en, "proxyEN_death": death,
        "xi_hartree_lattice": xi_H,
        "chi_doubling_worst": max(rel.values()), "time_s": time.time() - tl,
    }
    sweep_rows.append(row)
    for d, a, b in zip(MI_SEPS, mi_a, mi_b):
        mi_rows.append({"lam": lam, "d": d, "MI_1+1": a, "MI_1+1_2chi": b,
                        "MI_1+1_gaussian_proxy": gaussian_proxy_mutual_information(V_a, [c - 1 - d // 2], [c - 1 - d // 2 + 1 + d])})
    for d, a, b, m1, m2 in zip(PROXY_SEPS, pen_a, pen_b, pmi_a, pmi_b):
        proxy_rows.append({"lam": lam, "d": d, "proxyEN_8+8": a, "proxyEN_8+8_2chi": b,
                           "proxyMI_8+8": m1, "proxyMI_8+8_2chi": m2})
    print(f"lam={lam:g}: E0={gs.E0:.8f} (chi {gs.info['chi']}->{gs2.info['chi']}: {rel['E0']:.1e}) "
          f"mu_H^2={mu2:.4f} lam/mu_H^2={row['lam_over_mu_H2']:.3f} "
          f"EN(1+1,d0)={row['EN_1+1_d0']:.4e} EN(2+2,d0/d1/d2)={row['EN_2+2_d0']:.4e}/{row['EN_2+2_d1']:.4e}/{row['EN_2+2_d2']:.1e} "
          f"xi_MI={xi_mi:.3f} (xi_H/2={0.5 * xi_H:.3f}) xi_proxyEN={xi_en:.3f} death={death} "
          f"worst chi-doubling={max(rel.values()):.1e} [{time.time() - tl:.0f}s]", flush=True)

_csv("s2_sweep.csv", sweep_rows, list(sweep_rows[0].keys()))
_csv("s2_mi_profiles.csv", mi_rows, list(mi_rows[0].keys()))
_csv("s2_proxy_negativity.csv", proxy_rows, list(proxy_rows[0].keys()))
_dump("s2_fits.json", {"_source": "notebook.py::S2", "fits": fits,
                       "MI_fit_window": "d >= 2, MI > 1e-9", "proxy_fit_window": "1 <= d < death, EN > 1e-10"})
_dump("s2_chi_doubling.json", {"_source": "notebook.py::S2", **chi_doubling})
np.savez(DATA / "s2_covariances.npz", **cov_store, V_free=V0)

# ---- S2c: what the Gaussian proxy is, and the like-for-like free references ----
# The interacting covariance is the second-moment matrix of a NON-Gaussian pure
# state; the Gaussian state sharing it is MIXED (global symplectic eigenvalues
# above 1/2, Gaussian entropy S_G(V) > 0), and a mixed Gaussian state loses
# long-range negativity fast. So the proxy's early sudden death at lambda > 0
# is a property of the proxy, not of the vacuum. Reported per lambda, together
# with the free chain at the Hartree mass evaluated with the SAME fits and
# blocks (the "mass renormalization only" reference for xi_MI and the 8+8
# death separation).
_section("S2c Gaussian-proxy mixedness and free Hartree references")
from vacuum.core import symplectic_eigenvalues  # noqa: E402
from vacuum.interacting import hartree_coupling_matrix  # noqa: E402

s2c_rows = []
for lam in LAMS:
    V_a = cov_store[f"V_lam{lam:g}"]
    nu = symplectic_eigenvalues(V_a)
    K_H = hartree_coupling_matrix(L, MASS, lam)
    V_H = ground_state_cov(K_H)
    mi_H = np.array([mutual_information(V_H, [c - 1 - d // 2], [c - 1 - d // 2 + 1 + d]) for d in MI_SEPS])
    mi_a = np.array([r["MI_1+1"] for r in mi_rows if r["lam"] == lam])
    mask = (np.array(MI_SEPS) >= 2) & (mi_a > 1e-9)
    xi_mi_H, _, _, _ = _decay_fit(MI_SEPS, mi_H, mask)
    pen_H, _ = _proxy_profiles(V_H, PROXY_SEPS, PROXY_BLOCK)
    live_H = pen_H > 1e-10
    death_H = int(PROXY_SEPS[int(np.argmin(live_H))]) if not live_H.all() else -1
    mask_H = live_H & (np.array(PROXY_SEPS) >= 1)
    xi_en_H, _, _, _ = _decay_fit(PROXY_SEPS, pen_H, mask_H)
    row = {"lam": lam, "nu_min": float(nu.min()), "nu_max": float(nu.max()),
           "S_gaussian_of_V": float(entropy(V_a)),
           "xi_MI_free_at_hartree_mass_same_fit": xi_mi_H,
           "xi_MI_interacting": fits[f"lam={lam:g}"]["xi_MI"],
           "xi_MI_ratio_to_free_hartree": fits[f"lam={lam:g}"]["xi_MI"] / xi_mi_H,
           "proxyEN_death_free_at_hartree_mass": death_H, "xi_proxyEN_free_at_hartree_mass": xi_en_H,
           "proxyEN_death_interacting_proxy": fits[f"lam={lam:g}"]["proxyEN_death_separation"]}
    s2c_rows.append(row)
    print(row, flush=True)
_csv("s2c_proxy_mixedness.csv", s2c_rows, list(s2c_rows[0].keys()))

# n_max checks at the largest lambda: E0 and the 2+2 negativity vs cutoff
_section("S2b n_max checks at lambda = max")
lam_max = LAMS[-1]
nmax_rows = []
for n_max in ((4, 5) if SMOKE else (5, 6, 7, 8)):
    g = phi4_ground_mps(L, MASS, lam_max, n_max=n_max, chi_max=CHI)
    r = {"lam": lam_max, "n_max": n_max, "chi": g.info["chi"], "trunc_err": g.info["max_trunc_err"],
         "E0": g.E0, "EN_1+1_d0": log_negativity_mps(g.psi, [c - 1], [c]),
         "MI_1+1_d3": mutual_information_mps(g.psi, [c - 2], [c + 1]),
         "N_center": float(np.real(g.psi.expectation_value("Num")[c]))}
    if n_max <= 7:
        r["EN_2+2_d1"] = log_negativity_mps(g.psi, [c - 2, c - 1], [c + 1, c + 2])
    nmax_rows.append(r)
    print(r, flush=True)
_csv("s2b_nmax_checks.csv", nmax_rows, ["lam", "n_max", "chi", "trunc_err", "E0", "EN_1+1_d0", "MI_1+1_d3", "N_center", "EN_2+2_d1"])
TIMES["S2"] = time.time() - t0

# ---------------------------------------------------------------------------
# S3  the QEI number
# ---------------------------------------------------------------------------
_section("S3 QEI: variational smeared-energy minimum vs the free exact infimum")
t0 = time.time()
cq = QEI_L // 2
f = gaussian_f(TAU0, n_sigmas=4.0)
bound = qei_bound_2d(f)
qei_rows = []
qei_curves = {}
qei_gs = {}
for lam in LAMS:
    tl = time.time()
    gs = phi4_ground_mps(QEI_L, MASS, lam, n_max=QEI_NMAX, chi_max=QEI_CHI)
    qei_gs[lam] = gs
    res_H = qei_variational_mps(gs, cq, f, half_width=QEI_HW, s_grid=S_GRID, dt=QEI_DT, order=QEI_ORDER,
                                chi_max=QEI_CHI, shape_mass="hartree")
    if lam >= 1.0:  # the bare-mass shape differs from the Hartree shape only at strong coupling
        res_B = qei_variational_mps(gs, cq, f, half_width=QEI_HW, s_grid=S_GRID, dt=QEI_DT, order=QEI_ORDER,
                                    chi_max=QEI_CHI, shape_mass="bare")
    else:
        res_B = res_H
    best = res_H if res_H.E_var <= res_B.E_var else res_B
    # chi-doubling of the reported state (reference re-evolved at 2 chi as well)
    psi_best = apply_squeeze_family(gs.psi, best.meta["family"], best.s_star * best.theta_star)
    out2 = smeared_energy_mps(psi_best, gs.model, cq, f, MASS, lam, dt=QEI_DT, order=QEI_ORDER,
                              chi_max=2 * QEI_CHI, psi_ref=gs.psi, reference="evolved")
    # dt halving of the reported state (at the ends and the middle of the sweep)
    if lam in (LAMS[0], 1.0, LAMS[-1]):
        out_dt = smeared_energy_mps(psi_best, gs.model, cq, f, MASS, lam, dt=QEI_DT / 2, order=QEI_ORDER,
                                    chi_max=QEI_CHI, psi_ref=gs.psi, reference="evolved")
    else:
        out_dt = {"E_f": float("nan")}
    row = {
        "lam": lam, "L": QEI_L, "n_max": QEI_NMAX, "chi": best.chi, "dt": QEI_DT, "order": QEI_ORDER,
        "tau0": TAU0, "half_width": QEI_HW, "shape": best.meta["shape_mass"], "s_star": best.s_star,
        "E_var": best.E_var, "E_var_2chi": out2["E_f"], "E_var_half_dt": out_dt["E_f"],
        "rel_change_2chi": relative_change(best.E_var, out2["E_f"]),
        "rel_change_half_dt": (relative_change(best.E_var, out_dt["E_f"]) if np.isfinite(out_dt["E_f"]) else float("nan")),
        "trunc_err": best.trunc_err, "trunc_err_2chi": out2["trunc_err"],
        "E_var_hartree_shape": res_H.E_var, "E_var_bare_shape": res_B.E_var,
        "E_free_exact": best.E_free_exact, "E_free_family": best.E_free_family,
        "E_hartree_exact": best.E_hartree_exact, "mu_H2": best.mu_H2, "bound_2d_flanagan": bound,
        "ratio_var_over_free_exact": best.E_var / best.E_free_exact,
        "ratio_var_over_hartree_exact": best.E_var / best.E_hartree_exact,
        "ratio_var_over_free_family": best.E_var / best.E_free_family,
        "ratio_hartree_over_free_exact": best.E_hartree_exact / best.E_free_exact,
        "ratio_var_over_bound2d": best.E_var / bound,
        "margin_vs_free_exact": best.E_var - best.E_free_exact,
        "dmrg_trunc_err": gs.info["max_trunc_err"], "time_s": time.time() - tl,
    }
    qei_rows.append(row)
    qei_curves[f"lam={lam:g}"] = {
        "hartree_shape": {"s": res_H.s_grid, "E": res_H.E_grid, "theta_star": res_H.theta_star},
        "bare_shape": {"s": res_B.s_grid, "E": res_B.E_grid, "theta_star": res_B.theta_star},
        "h_trace_best_2chi": out2["h"], "h_ref_2chi": out2["h_ref"], "times": out2["times"],
    }
    print(f"lam={lam:g}: E_var={best.E_var:.6e} ({best.meta['shape_mass']} shape, s*={best.s_star:.3f}; "
          f"2chi {row['rel_change_2chi']:.1e}, dt/2 {row['rel_change_half_dt']:.1e}) "
          f"free exact={best.E_free_exact:.6e} free family={best.E_free_family:.6e} "
          f"hartree exact={best.E_hartree_exact:.6e} | var/free_exact={row['ratio_var_over_free_exact']:.4f} "
          f"var/hartree_exact={row['ratio_var_over_hartree_exact']:.4f} [{time.time() - tl:.0f}s]", flush=True)
_csv("s3_qei.csv", qei_rows, list(qei_rows[0].keys()))
_dump("s3_qei_curves.json", {"_source": "notebook.py::S3", **qei_curves})

# n_max re-evaluation of the reported state at one interacting point
_section("S3b QEI n_max check")
lam_chk = 1.0
best_row = next(r for r in qei_rows if r["lam"] == lam_chk)
curve = qei_curves[f"lam={lam_chk:g}"][best_row["shape"] + "_shape"]
theta_best = np.asarray(curve["theta_star"]) * best_row["s_star"]
from vacuum.interacting import squeeze_family  # noqa: E402

fam_chk = squeeze_family(cq, QEI_HW, QEI_L)
nmax_qei = []
for n_max in ((4, 5) if SMOKE else (6, 8, 10)):
    g = phi4_ground_mps(QEI_L, MASS, lam_chk, n_max=n_max, chi_max=QEI_CHI)
    psi_s = apply_squeeze_family(g.psi, fam_chk, theta_best)
    out = smeared_energy_mps(psi_s, g.model, cq, f, MASS, lam_chk, dt=QEI_DT, order=QEI_ORDER,
                             chi_max=QEI_CHI, psi_ref=g.psi, reference="evolved")
    nmax_qei.append({"lam": lam_chk, "n_max": n_max, "E_f": out["E_f"], "trunc_err": out["trunc_err"],
                     "chi": out["chi"], "E0_dmrg": g.E0})
    print(nmax_qei[-1], flush=True)
_csv("s3b_qei_nmax.csv", nmax_qei, ["lam", "n_max", "E_f", "trunc_err", "chi", "E0_dmrg"])
TIMES["S3"] = time.time() - t0

# ---------------------------------------------------------------------------
# S4  audits
# ---------------------------------------------------------------------------
_section("S4 audits")
t0 = time.time()
audit_rows = []
for lam in LAMS:
    gs = qei_gs[lam]
    worst = float("inf")
    for strength in (0.05, 0.3, 1.0):
        aud = passivity_audit_mps(gs.psi, gs.model, n_samples=6, max_sites=2, strength=strength,
                                  seed=int(1000 * lam + 100 * strength), tol=1e-8)
        worst = min(worst, aud["worst"])
        passed = aud["passed"]
        if not passed:
            print(f"!! passivity audit FAILED at lam={lam}, strength={strength}: worst {aud['worst']}")
    prof = energy_density_profile(gs.psi, MASS, lam)
    closure = abs(prof.sum() - mps_energy(gs.psi, gs.model))
    audit_rows.append({"lam": lam, "passivity_worst_dE": worst, "passivity_passed": worst >= -1e-8,
                       "energy_density_closure": closure, "chi": gs.info["chi"],
                       "trunc_err": gs.info["max_trunc_err"]})
    print(audit_rows[-1], flush=True)
_csv("s4_audits.csv", audit_rows, list(audit_rows[0].keys()))

# anomaly-protocol verdict on the QEI comparison
verdict = []
for r in qei_rows:
    if r["E_var"] < r["E_free_exact"] - 1e-6:
        verdict.append({"lam": r["lam"], "event": "E_var below the free exact infimum",
                        "E_var": r["E_var"], "E_free_exact": r["E_free_exact"],
                        "status": "ANOMALY CANDIDATE — suspect numerics first (PLAN.md protocol)"})
TIMES["S4"] = time.time() - t0

summary = {
    "_source": "notebook.py::summary",
    "build_git_head": build,
    "wall_time_s": time.time() - T_START,
    "section_times_s": TIMES,
    "sweep": {f"lam={r['lam']:g}": {k: r[k] for k in ("E0", "mu_H2", "lam_over_mu_H2", "EN_1+1_d0", "EN_2+2_d0",
                                                       "EN_2+2_d1", "EN_2+2_d2", "xi_MI", "xi_proxyEN",
                                                       "proxyEN_death", "chi_doubling_worst")}
              for r in sweep_rows},
    "qei": {f"lam={r['lam']:g}": {k: r[k] for k in ("E_var", "E_free_exact", "E_free_family", "E_hartree_exact",
                                                     "ratio_var_over_free_exact", "ratio_var_over_hartree_exact",
                                                     "ratio_var_over_free_family", "rel_change_2chi",
                                                     "rel_change_half_dt", "trunc_err", "s_star", "shape")}
            for r in qei_rows},
    "qei_bound_2d_flanagan": bound,
    "audits": audit_rows,
    "anomaly_events": verdict,
    "nmax_checks": {"sweep": nmax_rows, "qei": nmax_qei},
}
_dump("summary.json", summary)
print(f"\nTotal wall time: {summary['wall_time_s']:.1f} s; anomaly events: {len(verdict)}")
