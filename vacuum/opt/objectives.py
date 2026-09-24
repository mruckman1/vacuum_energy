"""Objectives for the differentiable vacuum (Layer 6): waveforms, layouts, schedules.

``make_objective(kind, **cfg)`` builds an :class:`Objective` — a flat
parameter vector theta, a JAX quantity-with-auxiliaries to MAXIMIZE, and a
plain-numpy ROUND TRIP that re-runs the same parameters through the
validated stack with its standing audits.  Kinds:

- ``'negativity'``  — harvested log-negativity of two (or more) oscillator
  detectors on a lattice field (``vacuum.opt.gjax_detectors`` twin of the
  M2.2 stack), parametrized over the switching waveform lam(t) (cos^2
  envelope times a Fourier series, or hat-function samples), a gap
  schedule Omega_d(t), per-detector delays, smearing centres/widths and
  the coupling amplitude; noise channels and field temperature fixed by
  the configuration (the noise floor).  Round trip:
  ``vacuum.detectors.initial_state`` (mandatory passivity audit) ->
  dt-halving-gated ``protocol_evolve`` with a closing ``EnergyLedger`` ->
  ``harvested_log_negativity`` (mp-margin rule) -> ``precision_audit`` on
  the detector block -> ``causality_audit`` on the field chain where the
  lattice is large enough for it not to be vacuous -> the M2.5
  communication split |M_comm|/|M| from the perturbative engine on the
  same kernel, switching shape and smearing (static gap, compact
  support, vacuum field, no channels — the rows where a perturbative
  state exists).
- ``'qet_energy'`` — Hotta's minimal QET model in JAX (dense 4x4 complex,
  arXiv:1101.3954 Sec. 3, Eqs. (5)-(8), (14)): Bob's extracted energy
  E_B(theta) as a function of his rotation angle.  Round trip:
  ``vacuum.qet.hotta.run_protocol`` at theta*, the closed forms
  ``theta_optimal``/``e_b_optimal``, the standing ``hotta_sweep`` audit
  (E_A >= E_B) and the no-signaling check on Bob's reduced state.
- ``'qei_ratio'`` — the 2d sharp-constant diagnostic E_min(f)/bound(f) of
  ``vacuum.inequalities.qei`` over a smooth sampling-function family
  (theta = the family parameter), assembled in JAX on the FIXED time grid
  and mode support the numpy ``optimize_sampling`` uses: the pulled-back
  sampling operator O_f = sum_a w_a f(t_a)^2 S(t_a)^T h S(t_a) on the
  support, E_min = (1/2) sum_k sigma_k(O_f) - (1/2) Tr(O_f V_vac) from the
  PSD-safe symplectic spectrum (``gjax.symplectic_eigenvalues``), and the
  analytic bound -C int f'^2 (Flanagan PRD 56, 4922 (1997), Eq. (8) with
  rho = f^2).  Round trip: numpy ``sampling_operator`` / ``qei_minimize``
  / ``qei_ratio`` on the same grid and support, the exhibited extremal
  state's physicality (``assert_physical``) and the QEI audit on it.
- ``'pkmm_negativity'`` — numpy-only: PKMM's closed-form harvested
  negativity estimator N^(2)(alpha, beta, gamma, delta) (PRD 92, 064042
  (2015), Eqs. (29)-(31), (68); ``vacuum.detectors.udw``), the recorded
  hand-tuned landscape of tests/test_udw.py, for the re-discovery anchor.
  Its quantity is evaluated with numpy (scipy quadrature inside), so
  ``optimize`` uses finite differences or Nelder-Mead on it.

Waveform bases (identical formulas evaluated with numpy on the round trip
and with jnp under tracing, ``_basis``):

    'cos2'    : B_0(t) = chi(t) = cos^2(pi u / 2T) on |u| <= T, u = t - t0
                (the M2.1 compact C^1 switching; the published baseline);
    'fourier' : B_0 = chi, B_k = chi cos(k pi u / T), B_{n_h + k} = chi sin(k pi u / T),
                k = 1..n_h  (compact support, C^1 at the edges);
    'pwl'     : hat functions on n interior knots of [t0 - T, t0 + T]
                (waveform SAMPLES; zero at the window edges).

lam_d(t) = lam_max sum_k a_k B_k(t - delay_d);  Omega_d(t) = Omega_d (1 + sum_k g_k B_k(t)).
The support [t0 - T, t0 + T] (shifted by the delay) is fixed by the
configuration, so the causal status of a (window, separation) pair — the
signaling bound's projection — is not something the optimizer can move.

The round-trip rule: ``Objective.roundtrip(theta)`` is the ONLY source of
reportable numbers.  It returns a dict with the observable, its precision
flag, the ledger work, every audit result, the communication fraction
where defined, and ``audits_passed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError("vacuum.opt.objectives needs JAX (install the '.[diff]' extra)") from exc

from vacuum.audits.ledger import DEFAULT_CLOSURE_ATOL
from vacuum.opt import gjax as G
from vacuum.opt import gjax_detectors as GD

jax.config.update("jax_enable_x64", True)

__all__ = [
    "Objective",
    "make_objective",
    "basis_functions",
    "ShapeSwitching",
    "NEGATIVITY_BLOCKS",
]


# --------------------------------------------------------------------------
# Objective container
# --------------------------------------------------------------------------


@dataclass
class Objective:
    """A parametrized objective: JAX quantity to maximize + numpy round trip.

    ``quantity_aux(theta) -> (q, aux)`` (traced), ``roundtrip(theta) -> dict``
    (numpy stack + audits), ``unpack(theta) -> dict`` of named parameter
    blocks, ``theta0`` a generic start, ``bounds`` a tuple of (lo, hi) per
    entry or None, ``names`` one label per theta entry.  ``f(theta)`` is
    the loss -q the optimizers minimize.
    """

    kind: str
    names: Tuple[str, ...]
    theta0: np.ndarray
    bounds: Optional[Tuple[Tuple[Optional[float], Optional[float]], ...]]
    quantity_aux: Callable[[Any], Tuple[Any, Dict[str, Any]]]
    roundtrip: Callable[[Any], Dict[str, Any]]
    unpack: Callable[[Any], Dict[str, Any]]
    cfg: Dict[str, Any] = field(default_factory=dict)
    differentiable: bool = True

    def f(self, theta):
        return -self.quantity_aux(theta)[0]

    def quantity(self, theta):
        return self.quantity_aux(theta)[0]

    def __repr__(self):
        return f"Objective(kind={self.kind!r}, n_params={len(self.names)}, names={self.names})"


def make_objective(kind, **cfg):
    """Build an :class:`Objective` of the given kind (module docstring)."""
    if kind == "negativity":
        return _negativity_objective(**cfg)
    if kind == "qet_energy":
        return _qet_objective(**cfg)
    if kind == "qei_ratio":
        return _qei_objective(**cfg)
    if kind == "pkmm_negativity":
        return _pkmm_objective(**cfg)
    if kind == "gain_per_loss":
        raise NotImplementedError(
            "'gain_per_loss' (Layer 5 Floquet amplification per loss) is reserved; "
            "its simulator is owned by vacuum.floquet and is not mirrored here"
        )
    raise ValueError(
        f"unknown objective kind {kind!r}: choose from 'negativity', 'qet_energy', "
        f"'qei_ratio', 'pkmm_negativity'"
    )


# --------------------------------------------------------------------------
# Waveform bases (numpy and jnp from one formula)
# --------------------------------------------------------------------------


def _envelope(t, t0, T, xp):
    u = t - t0
    return xp.where(xp.abs(u) <= T, xp.cos(0.5 * xp.pi * u / T) ** 2, 0.0)


def basis_functions(t, kind, n_coeff, t0, T, xp=np):
    """(n_coeff,) basis values at scalar time t (module docstring)."""
    t0 = float(t0)
    T = float(T)
    if kind == "cos2":
        if n_coeff != 1:
            raise ValueError("'cos2' has exactly one coefficient")
        return xp.stack([_envelope(t, t0, T, xp)])
    if kind == "fourier":
        if n_coeff < 1 or (n_coeff - 1) % 2 != 0:
            raise ValueError("'fourier' needs n_coeff = 1 + 2 n_harmonics")
        n_h = (n_coeff - 1) // 2
        chi = _envelope(t, t0, T, xp)
        u = t - t0
        ks = xp.arange(1, n_h + 1, dtype=float) if n_h > 0 else xp.zeros(0)
        cos = chi * xp.cos(ks * xp.pi * u / T)
        sin = chi * xp.sin(ks * xp.pi * u / T)
        return xp.concatenate([xp.stack([chi]), cos, sin])
    if kind == "pwl":
        if n_coeff < 1:
            raise ValueError("'pwl' needs >= 1 interior knot")
        knots = xp.linspace(t0 - T, t0 + T, n_coeff + 2)[1:-1]
        dk = 2.0 * T / (n_coeff + 1)
        return xp.maximum(0.0, 1.0 - xp.abs(t - knots) / dk)
    raise ValueError(f"basis kind must be 'cos2', 'fourier' or 'pwl', got {kind!r}")


class ShapeSwitching:
    """Duck-typed Switching (callable, .support, .width, .t0, .derivative) of a basis expansion.

    Used to hand an optimized waveform SHAPE s(t) = sum_k a_k B_k(t) to the
    perturbative engine (``UDWDetector`` requires a Switching-like object)
    for the communication split.  ``derivative`` is a central difference
    (the engine does not use it; the QEI module's f-handles are separate).
    """

    __slots__ = ("kind", "coeffs", "t0", "width", "support", "n_coeff")

    def __init__(self, kind, coeffs, t0, T):
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "coeffs", np.array(coeffs, dtype=float))
        object.__setattr__(self, "t0", float(t0))
        object.__setattr__(self, "width", float(T))
        object.__setattr__(self, "support", (float(t0) - float(T), float(t0) + float(T)))
        object.__setattr__(self, "n_coeff", int(np.size(coeffs)))

    def __setattr__(self, name, value):
        raise AttributeError("ShapeSwitching objects are immutable")

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        if t.ndim == 0:
            return float(basis_functions(float(t), self.kind, self.n_coeff, self.t0, self.width, np) @ self.coeffs)
        return np.array([self(float(x)) for x in t.ravel()]).reshape(t.shape)

    def derivative(self, t, h=1e-6):
        return (self(np.asarray(t) + h) - self(np.asarray(t) - h)) / (2.0 * h)

    def __repr__(self):
        return f"ShapeSwitching(kind={self.kind!r}, n_coeff={self.n_coeff}, t0={self.t0}, T={self.width})"


# --------------------------------------------------------------------------
# 'negativity'
# --------------------------------------------------------------------------

NEGATIVITY_BLOCKS = ("lam", "lam_max", "gap", "gap_mod", "delay", "center", "width")


def _negativity_objective(
    K=None,
    *,
    N=None,
    mass=None,
    bc="dirichlet",
    detectors,
    gaps,
    window,
    basis="cos2",
    n_coeff=1,
    lam_coeffs=None,
    lam_max=0.3,
    gap_basis=None,
    n_gap_coeff=0,
    gap_coeffs=None,
    delays=None,
    free=("lam",),
    T=0.0,
    noise=None,
    ts=None,
    splits=64,
    quantity="surrogate",
    conv_tol=1e-9,
    max_halvings=14,
    ledger_tol=DEFAULT_CLOSURE_ATOL,
    floor_tol=1e-10,
    dps=50,
    perturbative_quad=None,
    lam_bounds=None,
    modes_A=None,
    modes_B=None,
    signaling_proxy=None,
    proxy_panel_div=2,
):
    conv_tol_cfg, max_halvings_cfg = float(conv_tol), int(max_halvings)
    """See :func:`make_objective` ('negativity').

    detectors : sequence of dicts, one per detector:
        {'site': int}  or  {'profile': 'gaussian'|'cos2', 'center': c, 'width': w, 'bc': 'open'|'periodic'}
    gaps : (n_det,) static gaps Omega_d
    window : (t0, T) — the envelope centre and half-width (compact support [t0-T, t0+T] per detector, shifted by its delay)
    noise : None or {'kappa', 'nbar', 'gamma_phi', 'axis'} — detector channel rates (the noise floor)
    quantity : 'surrogate' (climbs through sudden death), 'E_N', or 'E_N_per_work'
    free : which parameter blocks form theta (NEGATIVITY_BLOCKS)
    signaling_proxy : None (auto: two detectors, static gaps) or bool — carry the
        differentiable second-order communication fraction |M_comm|/|M|
        (``gjax_detectors.communication_split_fixed``, TMM21 split on a fixed
        triangular grid, panel = the M2.1 hazard-note width / proxy_panel_div)
        in ``aux['comm_fraction']`` so the signaling bound can be penalized
        inside the loop; the numpy split on the round trip is what is reported.
    """
    from vacuum.core import harmonic_chain_K

    if K is None:
        if N is None or mass is None:
            raise ValueError("pass K, or (N, mass, bc) to build a harmonic chain")
        K = harmonic_chain_K(int(N), float(mass), bc=bc)
    K = np.asarray(K, dtype=float)
    Nf = K.shape[0]
    Kj = jnp.asarray(K)
    detectors = [dict(d) for d in detectors]
    n_det = len(detectors)
    gaps = np.asarray(gaps, dtype=float).reshape(n_det)
    t0, Tw = float(window[0]), float(window[1])
    delays = np.zeros(n_det) if delays is None else np.asarray(delays, dtype=float).reshape(n_det)
    free = tuple(free)
    for b in free:
        if b not in NEGATIVITY_BLOCKS:
            raise ValueError(f"unknown free block {b!r}; choose from {NEGATIVITY_BLOCKS}")
    if lam_coeffs is None:
        lam_coeffs = np.zeros(int(n_coeff))
        lam_coeffs[0] = 1.0
    lam_coeffs = np.asarray(lam_coeffs, dtype=float).reshape(int(n_coeff))
    gap_basis = basis if gap_basis is None else gap_basis
    n_gap = int(n_gap_coeff)
    gap_coeffs = np.zeros(n_gap) if gap_coeffs is None else np.asarray(gap_coeffs, dtype=float).reshape(n_gap)
    if "gap_mod" in free and n_gap == 0:
        raise ValueError("free block 'gap_mod' needs n_gap_coeff > 0")
    smeared = [("profile" in d) for d in detectors]
    centers = np.array([float(d.get("center", d.get("site", 0))) for d in detectors])
    widths = np.array([float(d.get("width", 1.0)) for d in detectors])
    if ("center" in free or "width" in free) and not all(smeared):
        raise ValueError("free 'center'/'width' blocks need every detector smeared ({'profile': ...})")
    noise = None if noise is None else dict(noise)
    axis = (noise or {}).get("axis", "isotropic")
    ts_grid = np.array([min(t0 - Tw + delays.min(), t0 - Tw), max(t0 + Tw + delays.max(), t0 + Tw)]) if ts is None else np.asarray(ts, dtype=float)
    det_modes = tuple(range(Nf, Nf + n_det))
    quad = dict(rtol=1e-8, atol=1e-16, max_doublings=12)
    if perturbative_quad:
        quad.update(perturbative_quad)
    if "delay" in free:  # free delays move the supports: give the grids room
        ts_grid = np.array([ts_grid[0] - Tw, ts_grid[-1] + Tw]) if ts is None else ts_grid
    use_proxy = (n_det == 2 and n_gap == 0) if signaling_proxy is None else bool(signaling_proxy)
    if use_proxy and (n_det != 2 or n_gap != 0):
        raise ValueError("the signaling proxy needs exactly two detectors and static gaps")
    proxy = None
    if use_proxy:
        w2, U_K = np.linalg.eigh(K)
        omega_K = np.sqrt(w2)
        panel = GD.initial_panel(gaps, (Tw, Tw)) / float(proxy_panel_div)
        grid = GD.pair_term_grid(ts_grid[0], ts_grid[-1], panel, 16, 4)
        proxy = dict(omega=jnp.asarray(omega_K), U=jnp.asarray(U_K), grid=grid,
                     t=jnp.asarray(grid[0]), tp=jnp.asarray(grid[2]), panel=panel)

    # ---- theta layout -------------------------------------------------
    blocks: Dict[str, np.ndarray] = {
        "lam": lam_coeffs, "lam_max": np.array([float(lam_max)]), "gap": gaps,
        "gap_mod": gap_coeffs, "delay": delays, "center": centers, "width": widths,
    }
    layout: List[Tuple[str, int, int]] = []
    names: List[str] = []
    off = 0
    for b in free:
        n = int(blocks[b].size)
        layout.append((b, off, off + n))
        names += [f"{b}[{i}]" for i in range(n)] if n > 1 else [b]
        off += n
    theta0 = np.concatenate([blocks[b] for b in free]) if free else np.zeros(0)
    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    for b in free:
        n = int(blocks[b].size)
        if b == "gap":
            bounds += [(1e-3, None)] * n
        elif b == "width":
            bounds += [(1e-2, None)] * n
        elif b == "lam" and lam_bounds is not None:
            bounds += [tuple(lam_bounds)] * n
        else:
            bounds += [(None, None)] * n

    def unpack(theta, xp=np):
        theta = xp.asarray(theta)
        out = {b: (xp.asarray(v) if xp is not np else np.asarray(v, dtype=float)) for b, v in blocks.items()}
        for b, a, z in layout:
            out[b] = theta[a:z]
        return out

    # ---- traced quantity ------------------------------------------------
    def _lam_of_t_factory(p, xp):
        def lam_of_t(t):
            vals = []
            for d in range(n_det):
                B = basis_functions(t - p["delay"][d], basis, int(n_coeff), t0, Tw, xp)
                vals.append(p["lam_max"][0] * xp.sum(p["lam"] * B))
            return xp.stack(vals)
        return lam_of_t

    def _gaps_of_t_factory(p, xp):
        def gaps_of_t(t):
            if n_gap == 0:
                return p["gap"]
            B = basis_functions(t, gap_basis, n_gap, t0, Tw, xp)
            return p["gap"] * (1.0 + xp.sum(p["gap_mod"] * B))
        return gaps_of_t

    def _profiles_jax(p):
        cols = []
        for d, det in enumerate(detectors):
            if smeared[d]:
                fn = GD.gaussian_profile if det["profile"] == "gaussian" else GD.cos2_profile
                cols.append(fn(Nf, p["center"][d], p["width"][d], bc=det.get("bc", "open")))
            else:
                cols.append(GD.point_profile(Nf, int(det["site"])))
        return jnp.stack(cols, axis=1)

    def quantity_aux(theta):
        p = unpack(theta, jnp)
        F = _profiles_jax(p)
        channels = ()
        if noise is not None:
            channels = (GD.rate_channel(Nf + n_det, det_modes, kappa=noise.get("kappa", 0.0),
                                        nbar=noise.get("nbar", 0.0), gamma_phi=noise.get("gamma_phi", 0.0),
                                        axis=axis, gaps=p["gap"]),)
        out = GD.run_harvesting(Kj, F, _gaps_of_t_factory(p, jnp), _lam_of_t_factory(p, jnp), ts_grid,
                                splits=int(splits), T=T, channels=channels)
        aux = {"E_N": out.E_N, "surrogate": out.surrogate, "work": out.work,
               "dissipated": out.dissipated, "n_d": out.n_d}
        if proxy is not None:
            shape = [lambda tt, d=d: jnp.sum(p["lam"] * basis_functions(tt - p["delay"][d], basis, int(n_coeff), t0, Tw, jnp))
                     for d in range(n_det)]
            chi_t = [jax.vmap(shape[d])(proxy["t"]) for d in range(2)]
            chi_tp = [jax.vmap(shape[d])(proxy["tp"]) for d in range(2)]
            amps = GD.smeared_amplitudes(proxy["omega"], proxy["U"], F[:, 0], F[:, 1])
            lam_udw = p["lam_max"][0] / jnp.sqrt(2.0 * p["gap"])
            M, M_vac, M_comm, frac = GD.communication_split_fixed(
                proxy["omega"], amps, chi_t[0], chi_t[1], chi_tp[0], chi_tp[1], p["gap"][0], p["gap"][1],
                proxy["grid"], lam_udw[0], lam_udw[1])
            aux.update({"comm_fraction": frac, "M_abs_proxy": jnp.abs(M), "M_vac_abs_proxy": jnp.abs(M_vac),
                        "M_comm_abs_proxy": jnp.abs(M_comm)})
        if quantity == "surrogate":
            q = out.surrogate
        elif quantity == "E_N":
            q = out.E_N
        elif quantity == "E_N_per_work":
            q = out.E_N / out.work
        else:
            raise ValueError(f"quantity must be 'surrogate', 'E_N' or 'E_N_per_work', got {quantity!r}")
        return q, aux

    # ---- numpy round trip ----------------------------------------------
    def roundtrip(theta, *, communication=True, causality=True, evolve=True, conv_tol=None, max_halvings=None):
        """numpy stack + audits at theta; ``evolve=False`` returns only the
        perturbative communication columns (no gated evolution).  ``conv_tol``
        / ``max_halvings`` override the configured gate (recorded in the row)."""
        _conv_tol = conv_tol_cfg if conv_tol is None else float(conv_tol)
        _max_halvings = max_halvings_cfg if max_halvings is None else int(max_halvings)
        from vacuum.audits import EnergyLedger, causality_audit, precision_audit
        from vacuum.core import partial_transpose, symplectic_eigenvalues
        from vacuum.detectors import (
            UDWDetector, attach_detectors, detector_block, detector_occupations,
            harvested_log_negativity, imperfection_channel, initial_state, pair_state_with_split,
            protocol_evolve, smearing, wightman_lattice,
        )

        p = unpack(theta, np)
        profiles = []
        for d, det in enumerate(detectors):
            if smeared[d]:
                profiles.append(smearing(det["profile"], float(p["center"][d]), float(p["width"][d]), Nf,
                                         bc=det.get("bc", "open")))
            else:
                profiles.append(int(det["site"]))
        lam_of_t = _lam_of_t_factory(p, np)
        gaps_of_t = _gaps_of_t_factory(p, np)
        gaps0 = np.asarray(gaps_of_t(ts_grid[0]), dtype=float)
        if not evolve:
            row = {"comm_fraction": float("nan"), "E_N_pert": float("nan"), "M_vac_abs": float("nan"),
                   "M_comm_abs": float("nan"), "theta": np.asarray(theta, dtype=float), "params": p}
            _communication_columns(row, p, profiles, gaps0)
            return row
        init = initial_state(K, profiles, tuple(float(g) for g in gaps0), float(T), strict=False)

        def K_of_t(t):
            return attach_detectors(K, profiles, gaps_of_t(t), lam_of_t(t))

        channels = ()
        if noise is not None:
            channels = (imperfection_channel(det_modes, kappa=float(noise.get("kappa", 0.0)),
                                             nbar=float(noise.get("nbar", 0.0)),
                                             gamma_phi=float(noise.get("gamma_phi", 0.0)),
                                             axis=axis, gaps=tuple(float(g) for g in gaps0)),)
        observables = {"E_N": lambda V: harvested_log_negativity(detector_block(V, Nf), floor_tol=floor_tol, dps=dps).E_N}
        for d in range(n_det):
            observables[f"n_d{d}"] = lambda V, _d=d: float(detector_occupations(V, Nf, gaps_of_t(ts_grid[-1]))[_d])
        ledger = EnergyLedger(K=init.K_tot, tol=float(ledger_tol))
        ev = protocol_evolve(init.V, K_of_t, ts_grid, channels=channels, observables=observables,
                             conv_tol=_conv_tol, max_halvings=_max_halvings, ledger=ledger,
                             require_convergence=False)
        ledger_res = ledger.close(strict=False)
        V_AB = detector_block(ev.V, Nf)
        neg = harvested_log_negativity(V_AB, floor_tol=floor_tol, dps=dps)
        prec = precision_audit(V_AB, strict=False)
        n_AB = V_AB.shape[0] // 2
        A = (0,) if modes_A is None else tuple(modes_A)
        Bm = tuple(range(1, n_AB)) if modes_B is None else tuple(modes_B)
        nu_pt = symplectic_eigenvalues(partial_transpose(V_AB, Bm))
        surrogate = -math.log(2.0 * float(np.min(nu_pt)))
        caus = None
        if causality:
            try:
                caus = causality_audit(K, float(ts_grid[-1] - ts_grid[0]), bc=("periodic" if bc == "periodic" else "open"), strict=False)
            except ValueError as exc:  # lattice too small for a non-vacuous cone check
                caus = {"not_applicable": str(exc)}
        row: Dict[str, Any] = {
            "E_N": float(neg.E_N), "flagged": bool(neg.flagged), "regime": neg.regime,
            "min_nu_pt": float(neg.min_nu_pt), "surrogate": surrogate,
            "work": float(ev.work), "dissipated": float(ev.dissipated),
            "n_d": np.asarray([observables[f"n_d{d}"](ev.V) for d in range(n_det)]),
            "converged": bool(ev.converged), "halvings": int(ev.halvings), "dt_converged": float(ev.dt_converged),
            "movement": ev.movement, "conv_tol": _conv_tol, "max_halvings": _max_halvings,
            "V_AB": V_AB, "params": p, "theta": np.asarray(theta, dtype=float),
            "noise": {"kappa": float((noise or {}).get("kappa", 0.0)), "nbar": float((noise or {}).get("nbar", 0.0)),
                      "gamma_phi": float((noise or {}).get("gamma_phi", 0.0)), "T": float(T)},
            "audits": {"passivity": init.passivity, "ledger": ledger_res, "precision": prec, "causality": caus},
            "comm_fraction": float("nan"), "E_N_pert": float("nan"), "M_vac_abs": float("nan"), "M_comm_abs": float("nan"),
        }
        passed = [ledger_res.passed, prec.passed, bool(ev.converged)]
        if init.passivity is not None:
            passed.append(init.passivity.passed)
        if caus is not None and not isinstance(caus, dict):
            passed.append(caus.passed)
        row["audits_passed"] = bool(all(passed))
        # communication split: static gap, no channels, vacuum field, compact support
        if communication and n_gap == 0 and noise is None and float(T) == 0.0 and n_det == 2:
            _communication_columns(row, p, profiles, gaps0)
        return row

    def _communication_columns(row, p, profiles, gaps0):
        """M2.5 split of the second-order pair state on the same kernel/shape/smearing (T = 0)."""
        from vacuum.detectors import UDWDetector, pair_state_with_split, wightman_lattice

        if n_gap != 0 or n_det != 2 or not all(float(g) > 0 for g in gaps0):
            return
        try:
            kernel = wightman_lattice(K)
            dets = []
            for d in range(n_det):
                chi = ShapeSwitching(basis, p["lam"], t0 + float(p["delay"][d]), Tw)
                Fd = np.asarray(profiles[d]) if smeared[d] else np.eye(Nf)[int(detectors[d]["site"])]
                dets.append(UDWDetector(chi, Fd))
            lam_udw = tuple(float(p["lam_max"][0]) / math.sqrt(2.0 * float(g)) for g in gaps0)
            st = pair_state_with_split(kernel, dets[0], dets[1], lam_udw, tuple(float(g) for g in gaps0), **quad)
            row["comm_fraction"] = float(st.comm.comm_fraction)
            row["E_N_pert"] = float(st.log_negativity)
            row["M_vac_abs"] = float(abs(st.comm.M_vac))
            row["M_comm_abs"] = float(abs(st.comm.M_comm))
            row["perturbative_ok"] = bool(st.meta.get("perturbative_ok", False))
        except RuntimeError as exc:
            row["perturbative_error"] = str(exc)

    cfg = dict(kind="negativity", N=Nf, bc=bc, detectors=detectors, gaps=gaps, window=(t0, Tw), basis=basis,
               n_coeff=int(n_coeff), gap_basis=gap_basis, n_gap_coeff=n_gap, free=free, T=float(T), noise=noise,
               ts=ts_grid, splits=int(splits), quantity=quantity, conv_tol=conv_tol, det_modes=det_modes,
               signaling_proxy=use_proxy, proxy_panel=None if proxy is None else proxy["panel"])
    return Objective(kind="negativity", names=tuple(names), theta0=theta0, bounds=tuple(bounds) if bounds else None,
                     quantity_aux=quantity_aux, roundtrip=roundtrip, unpack=lambda th: unpack(th, np), cfg=cfg)


# --------------------------------------------------------------------------
# 'qet_energy' — Hotta's minimal model in JAX
# --------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _qet_objective(*, h, k, theta0=0.1):
    """Bob's extracted energy E_B(theta) of arXiv:1101.3954 Sec. 3 (dense 4x4, jnp)."""
    from vacuum.qet import hotta as H

    h = float(h)
    k = float(k)
    if not (h > 0 and k > 0):
        raise ValueError("Hotta's minimal model needs h > 0, k > 0")
    s = math.hypot(h, k)
    I4 = np.eye(4, dtype=complex)
    Hm = (h * np.kron(_SZ, _I2) + h * h / s * I4 + h * np.kron(_I2, _SZ) + h * h / s * I4
          + 2 * k * np.kron(_SX, _SX) + 2 * k * k / s * I4)  # Eqs. (5)-(7)
    g = np.zeros(4, dtype=complex)
    g[0] = math.sqrt(1 - h / s) / math.sqrt(2)
    g[3] = -math.sqrt(1 + h / s) / math.sqrt(2)
    rho_g = np.outer(g, g.conj())
    P = {mu: np.kron((_I2 + mu * _SX) / 2, _I2) for mu in (+1, -1)}
    branches = {mu: P[mu] @ rho_g @ P[mu] for mu in (+1, -1)}
    E_meas = float(np.real(np.trace(Hm @ (branches[1] + branches[-1]))))
    Hj = jnp.asarray(Hm)
    br = {mu: jnp.asarray(branches[mu]) for mu in (+1, -1)}
    SYj = jnp.asarray(np.kron(_I2, _SY))
    I4j = jnp.asarray(I4)

    def quantity_aux(theta):
        th = jnp.asarray(theta).reshape(-1)[0]
        rho_f = jnp.zeros((4, 4), dtype=jnp.complex128)
        for mu in (+1, -1):
            U = jnp.cos(th) * I4j - 1j * mu * jnp.sin(th) * SYj  # U_B(mu, theta), item III
            rho_f = rho_f + U @ br[mu] @ U.conj().T
        E_final = jnp.real(jnp.trace(Hj @ rho_f))
        E_B = E_meas - E_final
        return E_B, {"E_B": E_B, "E_A": jnp.asarray(E_meas), "E_final": E_final}

    def roundtrip(theta):
        th = float(np.asarray(theta).reshape(-1)[0])
        res = H.run_protocol(h, k, th)
        audit = H.hotta_sweep([h], [k], strict=False)
        ns = H.trace_distance(H.bob_reduced_state(h, k, False), H.bob_reduced_state(h, k, True))
        return {"E_B": res["E_B"], "E_A": res["E_A"], "E_B_closed": H.e_b_closed(h, k, th),
                "theta": th, "theta_optimal": H.theta_optimal(h, k), "E_B_optimal": H.e_b_optimal(h, k),
                "audits": {"hotta_sweep": audit}, "no_signaling_trace_distance": ns,
                "audits_passed": bool(audit.passed and ns < 1e-12 and res["E_A"] >= res["E_B"] - 1e-12)}

    return Objective(kind="qet_energy", names=("theta",), theta0=np.array([float(theta0)]),
                     bounds=((0.0, math.pi / 2),), quantity_aux=quantity_aux, roundtrip=roundtrip,
                     unpack=lambda th: {"theta": float(np.asarray(th).reshape(-1)[0])}, cfg={"h": h, "k": k})


# --------------------------------------------------------------------------
# 'qei_ratio' — sampling-function family ascent in JAX
# --------------------------------------------------------------------------


def _gl_grid(t_lo, t_hi, n_panels, nodes_per_panel):
    from numpy.polynomial.legendre import leggauss

    x, w = leggauss(int(nodes_per_panel))
    edges = np.linspace(float(t_lo), float(t_hi), int(n_panels) + 1)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    ts = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    ws = (half[:, None] * w[None, :]).ravel()
    return ts, ws


def _qei_objective(*, N, site, family="gaussian", bounds, m=0.0, bc="dirichlet", pad=8,
                   nodes_per_panel=10, panel_width=None, constant="sharp", theta0=None, n_sigmas=4.5):
    """See :func:`make_objective` ('qei_ratio')."""
    from vacuum.core import chain_propagator, ground_state_cov, harmonic_chain_K, reduce as reduce_cov
    from vacuum.inequalities import qei as Q

    N = int(N)
    site = int(site)
    lo, hi = float(bounds[0]), float(bounds[1])
    if family == "gaussian":
        t_hi = n_sigmas * hi
        make_f = lambda th: Q.gaussian_f(float(th), n_sigmas=n_sigmas)
    elif family == "cos2":
        t_hi = hi
        make_f = lambda th: Q.cos2_f(float(th))
    else:
        raise ValueError("family must be 'gaussian' or 'cos2'")
    t_lo = -t_hi
    K = harmonic_chain_K(N, float(m), bc)
    w2, U = np.linalg.eigh(K)
    w = np.sqrt(w2)
    if panel_width is None:
        panel_width = min(1.6, 2.4 / float(np.max(w)))
    n_panels = max(1, int(math.ceil((t_hi - t_lo) / panel_width)))
    ts, wq = _gl_grid(t_lo, t_hi, n_panels, nodes_per_panel)
    radius = int(math.ceil(t_hi)) + int(pad)
    if bc == "periodic":
        d = np.abs(np.arange(N) - site)
        d = np.minimum(d, N - d)
        support = np.flatnonzero(d <= radius)
    else:
        support = np.arange(max(0, site - radius), min(N - 1, site + radius) + 1)
    n_s = support.size
    # rank factor of the local density h = L L^T (energy = (1/2) R^T h R)
    hmat = Q.chain_energy_density_form(N, site, m=float(m), bc=bc)
    lam_h, Vh = np.linalg.eigh(hmat)
    keep = lam_h > 1e-13 * lam_h.max()
    L = Vh[:, keep] * np.sqrt(lam_h[keep])  # (2N, r)
    a_hat = U.T @ L[:N, :]   # (N, r)
    b_hat = U.T @ L[N:, :]
    U_S = U[support, :]
    # Y[a, r, :] = (S(t_a)^T l_r) restricted to the support quadratures
    c = np.cos(np.outer(w, ts))
    s = np.sin(np.outer(w, ts))
    sw = s / w[:, None]
    ws_ = s * w[:, None]
    Ys = []
    for r in range(L.shape[1]):
        top = U_S @ (c * a_hat[:, r][:, None] - ws_ * b_hat[:, r][:, None])   # (n_s, nt)
        bot = U_S @ (sw * a_hat[:, r][:, None] + c * b_hat[:, r][:, None])
        Ys.append(np.concatenate([top, bot], axis=0).T)  # (nt, 2 n_s)
    Y = np.concatenate(Ys, axis=0)  # (r*nt, 2 n_s)
    wq_r = np.tile(wq, L.shape[1])
    ts_r = np.tile(ts, L.shape[1])
    V_vac_sub = reduce_cov(ground_state_cov(K), support)
    Yj, wqj, tsj, Vvj = jnp.asarray(Y), jnp.asarray(wq_r), jnp.asarray(ts_r), jnp.asarray(V_vac_sub)
    C = Q.SHARP_CONSTANT_2D if constant == "sharp" else Q.FEWSTER_EVESON_CONSTANT_2D

    def f_of(th, t):
        if family == "gaussian":
            return jnp.exp(-0.5 * (t / th) ** 2)
        return jnp.where(jnp.abs(t) < th, jnp.cos(0.5 * jnp.pi * t / th) ** 2, 0.0)

    def fp2_of(th):
        return jnp.sqrt(jnp.pi) / (2.0 * th) if family == "gaussian" else jnp.pi ** 2 / (4.0 * th)

    def quantity_aux(theta):
        th = jnp.asarray(theta).reshape(-1)[0]
        f = f_of(th, tsj)
        scale = wqj * f * f
        O = (Yj * scale[:, None]).T @ Yj
        O = 0.5 * (O + O.T)
        sigma = G.symplectic_eigenvalues(O)
        e_min = 0.5 * jnp.sum(sigma) - 0.5 * jnp.sum(O * Vvj)
        bound = -C * fp2_of(th)
        ratio = e_min / bound
        return ratio, {"e_min": e_min, "bound": bound, "ratio": ratio}

    def roundtrip(theta, *, weight_floor=0.0, eps_reg=1e-8):
        """numpy ratio on the objective's grid/support + the exhibited state's audits.

        ``eps_reg`` regularizes only the EXHIBITED extremal state (e_min is
        taken from the raw spectrum, independent of it); the numpy module's
        default 1e-8 keeps that state float64-physical, which its 1e-10
        optimizer setting does not (margin ~ -1e-7, measured).
        """
        th = float(np.asarray(theta).reshape(-1)[0])
        fh = make_f(th)
        op = Q.sampling_operator(N, site, fh, m=float(m), bc=bc, support=support, panel_width=panel_width,
                                 nodes_per_panel=nodes_per_panel, t_lo=t_lo, t_hi=t_hi, weight_floor=weight_floor)
        res = Q.qei_minimize(op, eps_reg=eps_reg)
        bound = Q.qei_bound_2d(fh, constant=constant)
        ratio = res.e_min / bound
        # Physicality tolerance of the EXHIBITED state: 1e-6, the tolerance the
        # QEI module's own anchor uses for it (tests/test_qei.py,
        # test_extremal_state_exhibited: margin_star > -1e-6 and the embedded
        # margin > -1e-6).  e_min itself needs no exhibited state.
        phys_tol = 1e-6
        try:
            audit = Q.qei_audit(Q.embed_extremal(res, op), op, fh, constant=constant, strict=False,
                                tol=phys_tol, mp_mode_limit=0)
            unphysical = None
        except Q.QEIUnphysicalError as exc:
            audit = {"passed": False, "error": str(exc)}
            unphysical = str(exc)
        return {"ratio": ratio, "e_min": res.e_min, "bound": bound, "theta": th, "gap": res.gap,
                "margin_star": res.margin_star, "audits": {"qei": audit}, "unphysical": unphysical,
                "audits_passed": bool(audit["passed"] and res.margin_star >= -phys_tol), "support_size": int(n_s),
                "eps_reg": float(eps_reg), "weight_floor": float(weight_floor), "physicality_tol": phys_tol}

    th0 = 0.5 * (lo + hi) if theta0 is None else float(theta0)
    return Objective(kind="qei_ratio", names=("theta",), theta0=np.array([th0]), bounds=((lo, hi),),
                     quantity_aux=quantity_aux, roundtrip=roundtrip,
                     unpack=lambda th: {"theta": float(np.asarray(th).reshape(-1)[0])},
                     cfg=dict(N=N, site=site, family=family, bounds=(lo, hi), m=m, bc=bc, t_lo=t_lo, t_hi=t_hi,
                              support=support, n_nodes=int(ts.size), panel_width=panel_width))


# --------------------------------------------------------------------------
# 'pkmm_negativity' — the recorded PKMM landscape (numpy only)
# --------------------------------------------------------------------------


def _pkmm_objective(*, free=("alpha",), alpha=4.0, beta=6.0, gamma=3.0, delta=0.02, m_kwargs=None):
    """N^(2)(alpha = Omega T, beta = d/T, gamma = Delta/T, delta = sigma/T), PKMM Eq. (68)."""
    from vacuum.detectors import pkmm_negativity_estimator

    fixed = {"alpha": float(alpha), "beta": float(beta), "gamma": float(gamma), "delta": float(delta)}
    free = tuple(free)
    for f_ in free:
        if f_ not in fixed:
            raise ValueError(f"unknown PKMM parameter {f_!r}")
    mk = dict(m_kwargs or {})

    def unpack(theta):
        p = dict(fixed)
        for i, f_ in enumerate(free):
            p[f_] = float(np.asarray(theta).reshape(-1)[i])
        return p

    def quantity_aux(theta):
        p = unpack(theta)
        val = pkmm_negativity_estimator(p["alpha"], p["beta"], p["gamma"], p["delta"], **mk)
        return val, {"N2": val, **p}

    def roundtrip(theta):
        p = unpack(theta)
        val = pkmm_negativity_estimator(p["alpha"], p["beta"], p["gamma"], p["delta"], **mk)
        return {"N2": val, "E_N": math.log1p(2.0 * max(0.0, val)), **p, "audits_passed": True}

    lo_hi = {"alpha": (0.05, 12.0), "beta": (0.05, 25.0), "gamma": (0.0, 12.0), "delta": (0.01, 3.0)}
    return Objective(kind="pkmm_negativity", names=free, theta0=np.array([fixed[f_] for f_ in free]),
                     bounds=tuple(lo_hi[f_] for f_ in free), quantity_aux=quantity_aux, roundtrip=roundtrip,
                     unpack=unpack, cfg=dict(fixed=fixed, free=free), differentiable=False)
