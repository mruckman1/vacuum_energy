"""Loader for ``experiments/`` parameter files (Milestone M-I, item I.3).

This module is the ONLY place where published experimental numbers are
converted to the program's hbar = 1 lattice conventions (docs/API.md).
Parameter files hold values *verbatim* as published, in the paper's own
units (see ``experiments/README.md``, provenance rules); everything the
simulation layers consume in dimensionless form flows through
:func:`convert_to_lattice` here.

Provenance enforcement
----------------------
:func:`load_experiment` refuses to load (``ValueError``, not a warning)
any file that is missing one of the mandatory provenance fields:

- top level: ``source``, ``location``, ``retrieved``, ``transcribed_by``,
  ``quantities``;
- per quantity: ``units`` (validated against the unit table below) and
  ``value``;
- declared variants (``_variant-`` filename suffix, or a ``variant_of``
  or ``changes`` key): BOTH ``variant_of`` and a non-empty ``changes``
  dict;
- synthetic files (``"synthetic": true``): a non-empty ``comment``
  stating why the values are not transcribed from a publication.

Unit conventions (hbar = 1, k_B = 1)
------------------------------------
Natural base units used internally:

- energy-like quantities (frequencies, temperatures, energies) are
  angular frequencies in rad/s: ``E/hbar``.  A frequency quoted in Hz is
  multiplied by 2*pi; a temperature ``T`` becomes ``k_B*T/hbar``.
- times are kept in seconds, lengths in meters, velocities in m/s.

Lattice (dimensionless) units are then fixed by a caller-chosen reference
angular frequency ``omega_ref`` (rad/s) and, when lengths appear, a
propagation speed ``v`` (m/s, e.g. the speed of light in the waveguide):

- energy-like:  x_lattice = x_natural / omega_ref
- time:         t_lattice = t_seconds * omega_ref
- length:       L_lattice = (L_meters / v) * omega_ref   (light-crossing units)
- velocity:     u_lattice = u / v                        (waveguide speed -> 1)
- dimensionless quantities pass through (percent -> fraction).

Physical constants are the SI-2019 *exact defined* values (BIPM SI
Brochure, 9th edition, 2019): h = 6.62607015e-34 J s, k_B = 1.380649e-23
J/K, e = 1.602176634e-19 C.  No measured constants enter.

Everything here is pure-functional over plain data: no hidden state, no
mutation of inputs (Layer 6 JAX-mirror pre-positioning).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "H_PLANCK_SI",
    "HBAR_SI",
    "K_B_SI",
    "EV_SI",
    "MANDATORY_FIELDS",
    "Quantity",
    "Experiment",
    "experiments_root",
    "list_experiments",
    "load_experiment",
    "unit_kind",
    "to_natural",
    "quantity_to_lattice",
    "convert_to_lattice",
    "bose_occupation",
]

# --- SI 2019 exact defined constants (BIPM SI Brochure 9th ed., 2019) ----
H_PLANCK_SI = 6.62607015e-34  # J s  (exact by definition of the SI second/kg)
HBAR_SI = H_PLANCK_SI / (2.0 * math.pi)  # J s
K_B_SI = 1.380649e-23  # J / K  (exact)
EV_SI = 1.602176634e-19  # J     (exact, elementary charge x 1 V)

MANDATORY_FIELDS = ("source", "location", "retrieved", "transcribed_by", "quantities")

_TWO_PI = 2.0 * math.pi

# unit string -> (kind, factor to the natural base unit of that kind)
#   energy        -> rad/s        time -> s        length -> m
#   velocity      -> m/s          dimensionless -> 1
#   other / label -> no hbar=1 conversion defined (carried, never converted)
_UNIT_TABLE: Mapping[str, tuple[str, float]] = MappingProxyType(
    {
        # energy-like (hbar = 1: energy == angular frequency)
        "rad/s": ("energy", 1.0),
        "Hz": ("energy", _TWO_PI),
        "kHz": ("energy", _TWO_PI * 1e3),
        "MHz": ("energy", _TWO_PI * 1e6),
        "GHz": ("energy", _TWO_PI * 1e9),
        "THz": ("energy", _TWO_PI * 1e12),
        # temperatures (k_B = 1: T -> k_B T / hbar)
        "K": ("energy", K_B_SI / HBAR_SI),
        "mK": ("energy", K_B_SI / HBAR_SI * 1e-3),
        "uK": ("energy", K_B_SI / HBAR_SI * 1e-6),
        # energies proper
        "J": ("energy", 1.0 / HBAR_SI),
        "eV": ("energy", EV_SI / HBAR_SI),
        "ueV": ("energy", EV_SI / HBAR_SI * 1e-6),
        # times
        "s": ("time", 1.0),
        "ms": ("time", 1e-3),
        "us": ("time", 1e-6),
        "ns": ("time", 1e-9),
        "ps": ("time", 1e-12),
        # lengths
        "m": ("length", 1.0),
        "cm": ("length", 1e-2),
        "mm": ("length", 1e-3),
        "um": ("length", 1e-6),
        "nm": ("length", 1e-9),
        # velocities
        "m/s": ("velocity", 1.0),
        # dimensionless
        "dimensionless": ("dimensionless", 1.0),
        "1": ("dimensionless", 1.0),
        "percent": ("dimensionless", 1e-2),
        "%": ("dimensionless", 1e-2),
        # carried through verbatim, no hbar=1 conversion defined
        "ohm": ("other", 1.0),
        "A": ("other", 1.0),
        "uA": ("other", 1.0),
        "nH": ("other", 1.0),
        "counts": ("other", 1.0),
        # non-numeric metadata that still must declare what it is
        "label": ("label", 1.0),
    }
)


# ------------------------------------------------------------------ data --


@dataclass(frozen=True)
class Quantity:
    """One published quantity, verbatim, with its provenance."""

    value: Any  # number, list of numbers, or (units='label') a string
    units: str
    error: Any = None  # published uncertainty, same units as value
    location: str | None = None  # sub-location (equation, figure, table cell)
    note: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Experiment:
    """A validated experiments/ parameter file."""

    name: str
    path: str
    source: str
    location: str
    retrieved: str
    transcribed_by: str
    quantities: Mapping[str, Quantity]
    synthetic: bool = False
    comment: str | None = None
    variant_of: str | None = None
    changes: Mapping[str, Any] | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def lattice(self, omega_ref: float, v: float | None = None, strict: bool = False):
        """Dimensionless (hbar = 1 lattice) values; see convert_to_lattice."""
        return convert_to_lattice(self.quantities, omega_ref, v=v, strict=strict)


# ------------------------------------------------------------ file layout --


def experiments_root() -> Path:
    """Default root: the repo's experiments/ directory (editable install)."""
    return Path(__file__).resolve().parent.parent / "experiments"


def list_experiments(root=None) -> tuple[str, ...]:
    """Sorted relative names (no .json suffix) of every parameter file."""
    rootp = Path(root) if root is not None else experiments_root()
    names = [
        str(p.relative_to(rootp).with_suffix("")) for p in rootp.glob("**/*.json")
    ]
    return tuple(sorted(names))


# -------------------------------------------------------------- validation --


def _refuse(path, msg: str):
    raise ValueError(f"experiments file {path}: {msg}")


def _require_nonempty_str(data: Mapping[str, Any], key: str, path) -> str:
    val = data.get(key)
    if not isinstance(val, str) or not val.strip():
        _refuse(path, f"missing mandatory provenance field '{key}'")
    return val


def _validate_quantity(name: str, spec: Any, path) -> Quantity:
    if not isinstance(spec, dict):
        _refuse(path, f"quantity '{name}' must be an object with 'value' and 'units'")
    if "value" not in spec:
        _refuse(path, f"quantity '{name}' is missing mandatory field 'value'")
    units = spec.get("units")
    if not isinstance(units, str) or not units.strip():
        _refuse(path, f"quantity '{name}' is missing mandatory field 'units'")
    if units not in _UNIT_TABLE:
        _refuse(
            path,
            f"quantity '{name}' has unknown units '{units}' "
            f"(not in the validated unit table of vacuum.experiments_io)",
        )
    known = {"value", "units", "error", "location", "note"}
    extra = {k: v for k, v in spec.items() if k not in known}
    return Quantity(
        value=spec["value"],
        units=units,
        error=spec.get("error"),
        location=spec.get("location"),
        note=spec.get("note"),
        extra=MappingProxyType(extra),
    )


def load_experiment(name, root=None) -> Experiment:
    """Load and validate one parameter file.

    Parameters
    ----------
    name : str or Path
        Either a name relative to the experiments root, with or without
        the ``.json`` suffix (``"dce_microwave/wilson_2011_table1"``), or
        an absolute path to a ``.json`` file (used by tests for fixtures).
    root : path-like, optional
        Alternate experiments root (defaults to the repo's experiments/).

    Returns
    -------
    Experiment
        Frozen record; ``quantities`` maps name -> :class:`Quantity` with
        values verbatim as published (NO unit conversion here).

    Raises
    ------
    ValueError
        On any missing mandatory provenance field (load-time refusal).
    FileNotFoundError
        If the file does not exist.
    """
    rootp = Path(root) if root is not None else experiments_root()
    p = Path(name)
    if p.suffix != ".json":
        p = p.with_suffix(".json")
    if not p.is_absolute():
        p = rootp / p
    with open(p, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"experiments file {p}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        _refuse(p, "top level must be a JSON object")

    source = _require_nonempty_str(data, "source", p)
    location = _require_nonempty_str(data, "location", p)
    retrieved = _require_nonempty_str(data, "retrieved", p)
    transcribed_by = _require_nonempty_str(data, "transcribed_by", p)

    raw_q = data.get("quantities")
    if not isinstance(raw_q, dict) or not raw_q:
        _refuse(p, "missing mandatory provenance field 'quantities' (non-empty object)")
    quantities = {k: _validate_quantity(k, v, p) for k, v in raw_q.items()}

    # variant rule: filename suffix or either variant key triggers the full check
    stem = p.stem
    is_variant = "_variant-" in stem or "variant_of" in data or "changes" in data
    variant_of = data.get("variant_of")
    changes = data.get("changes")
    if is_variant:
        if not isinstance(variant_of, str) or not variant_of.strip():
            _refuse(p, "variant file must declare 'variant_of' (parent file name)")
        if not isinstance(changes, dict) or not changes:
            _refuse(
                p,
                "variant file must declare a non-empty 'changes' dict "
                "(every altered value, with a one-line reason)",
            )

    synthetic = bool(data.get("synthetic", False))
    comment = data.get("comment")
    if synthetic and (not isinstance(comment, str) or not comment.strip()):
        _refuse(
            p,
            "synthetic file must carry a non-empty 'comment' stating why its "
            "values are not transcribed from a publication",
        )

    known_top = {
        "source",
        "location",
        "retrieved",
        "transcribed_by",
        "quantities",
        "synthetic",
        "comment",
        "variant_of",
        "changes",
    }
    extras = {k: v for k, v in data.items() if k not in known_top}

    try:
        rel_name = str(p.relative_to(rootp).with_suffix(""))
    except ValueError:  # outside the root (absolute fixture path)
        rel_name = p.stem
    return Experiment(
        name=rel_name,
        path=str(p),
        source=source,
        location=location,
        retrieved=retrieved,
        transcribed_by=transcribed_by,
        quantities=MappingProxyType(quantities),
        synthetic=synthetic,
        comment=comment,
        variant_of=variant_of,
        changes=MappingProxyType(changes) if isinstance(changes, dict) else None,
        extras=MappingProxyType(extras),
    )


# -------------------------------------------------------- unit conversion --


def unit_kind(units: str) -> str:
    """Kind of a unit string: energy | time | length | velocity | dimensionless | other | label."""
    try:
        return _UNIT_TABLE[units][0]
    except KeyError:
        raise ValueError(f"unknown units '{units}'") from None


def to_natural(value, units: str):
    """Convert to the natural base unit of the unit's kind.

    Returns ``(converted, kind)`` where the base units are rad/s (energy,
    hbar = k_B = 1), s (time), m (length), m/s (velocity), or 1
    (dimensionless).  Lists convert elementwise (returned as tuples).
    'other'/'label' kinds are returned unchanged (no conversion defined).
    """
    kind, factor = _UNIT_TABLE.get(units, (None, None))
    if kind is None:
        raise ValueError(f"unknown units '{units}'")
    if kind in ("other", "label"):
        return value, kind
    if isinstance(value, (list, tuple)):
        return tuple(float(x) * factor for x in value), kind
    return float(value) * factor, kind


def quantity_to_lattice(value, units: str, omega_ref: float, v: float | None = None):
    """Convert one published value to dimensionless hbar = 1 lattice units.

    omega_ref : reference angular frequency (rad/s) defining the lattice
        energy unit; times scale as t * omega_ref, energies as E / omega_ref.
    v : propagation speed (m/s), required for lengths and velocities
        (length unit = v / omega_ref, the light-crossing convention).

    Raises ValueError for units with no hbar = 1 conversion ('other'/'label').
    """
    if not (omega_ref > 0.0):
        raise ValueError(f"omega_ref must be positive, got {omega_ref}")
    nat, kind = to_natural(value, units)
    if kind == "energy":
        scale = 1.0 / omega_ref
    elif kind == "time":
        scale = omega_ref
    elif kind == "dimensionless":
        scale = 1.0
    elif kind in ("length", "velocity"):
        if v is None or not (v > 0.0):
            raise ValueError(
                f"converting units '{units}' requires a propagation speed v (m/s)"
            )
        scale = omega_ref / v if kind == "length" else 1.0 / v
    else:  # 'other' / 'label'
        raise ValueError(f"no hbar=1 conversion defined for units '{units}'")
    if isinstance(nat, tuple):
        return tuple(x * scale for x in nat)
    return nat * scale


def convert_to_lattice(
    quantities: Mapping[str, Quantity],
    omega_ref: float,
    v: float | None = None,
    strict: bool = False,
) -> Mapping[str, Any]:
    """Convert a validated quantities mapping to lattice units.

    Returns a read-only mapping name -> dimensionless value (tuples for
    ranges).  Quantities whose units have no hbar = 1 conversion (kind
    'other'/'label', or lengths/velocities when ``v`` is not given) are
    skipped, unless ``strict=True``, in which case they raise ValueError.
    Published errors convert with the same linear factor and are returned
    under ``"<name>_error"`` when present and numeric.
    """
    out: dict[str, Any] = {}
    for name, q in quantities.items():
        try:
            out[name] = quantity_to_lattice(q.value, q.units, omega_ref, v=v)
        except (ValueError, TypeError):
            if strict:
                raise ValueError(
                    f"quantity '{name}' (units '{q.units}') has no hbar=1 "
                    f"lattice conversion"
                ) from None
            continue
        if q.error is not None:
            try:
                out[name + "_error"] = quantity_to_lattice(
                    q.error, q.units, omega_ref, v=v
                )
            except (ValueError, TypeError):
                pass
    return MappingProxyType(out)


def bose_occupation(omega_natural: float, temperature_natural: float) -> float:
    """Thermal occupation n = 1/(exp(omega/T) - 1), both in natural units.

    ``omega_natural`` and ``temperature_natural`` are energy-like values in
    the same natural units (e.g. both rad/s from :func:`to_natural`, or
    both already in lattice units) — only their ratio enters.  T = 0
    returns 0.
    """
    if omega_natural <= 0.0:
        raise ValueError(f"omega must be positive, got {omega_natural}")
    if temperature_natural < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature_natural}")
    if temperature_natural == 0.0:
        return 0.0
    x = omega_natural / temperature_natural
    if x > 700.0:  # exp overflow guard; occupation is exactly 0.0 at float64
        return 0.0
    return 1.0 / math.expm1(x)
