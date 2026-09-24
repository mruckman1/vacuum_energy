"""Integration-layer contract: every public name is reachable from its package.

Pinned here so the surface cannot drift when packages are built in parallel
(the Layer 4-7 build left ``vacuum.qet.resource`` and the 4d mode-space
minimizer of ``vacuum.inequalities.qei`` unreachable from their packages, and
``vacuum.composite`` / ``vacuum.interacting`` / ``vacuum.hardware`` absent
from ``vacuum.__all__``):

1. every subpackage on disk is an attribute of ``vacuum`` and in ``vacuum.__all__``;
2. every name in every package ``__all__`` resolves (the lazy JAX / TeNPy /
   qiskit names raise their guided ImportError only when the extra is absent);
3. package ``__all__`` is a superset of every submodule's own ``__all__`` —
   the one documented exception being the two JAX *mirror* modules
   ``vacuum.opt.gjax`` / ``vacuum.opt.gjax_detectors``, whose names duplicate
   ``vacuum.core`` / ``vacuum.detectors`` on purpose and are reached as
   namespaces (``vacuum.opt.gjax.entropy``);
4. no duplicates in any ``__all__``;
5. with tenpy, cvxpy, jax, qiskit, qiskit_aer and qiskit_ibm_runtime blocked at
   the import system, ``import vacuum`` and every subpackage still import
   and every non-lazy
   name resolves (the extras policy of README.md), in a subprocess so the
   block cannot leak into this session.  The block is a meta-path finder that
   raises ModuleNotFoundError — what a missing package does — rather than
   ``sys.modules[name] = None``, which scipy's array-API helper trips over
   (``getattr(sys.modules['jax'], 'Array')``) inside every ``eigsh`` call.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
import textwrap

import pytest

import vacuum

PACKAGES = ("audits", "composite", "core", "detectors", "fermions", "floquet", "geometry",
            "hardware", "inequalities", "interacting", "modular", "opt", "qet")
MODULES = ("experiments_io", "protocol")
#: JAX mirrors: reached as namespaces, never flattened (see vacuum/opt/__init__.py).
MIRROR_MODULES = {"vacuum.opt.gjax", "vacuum.opt.gjax_detectors"}
EXTRAS = ("tenpy", "cvxpy", "jax", "jaxlib", "qiskit", "qiskit_aer", "qiskit_ibm_runtime")


def _has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):  # an import block counts as absent
        return False


def test_every_subpackage_is_reachable_from_vacuum():
    on_disk = sorted(m.name for m in pkgutil.iter_modules(vacuum.__path__))
    assert set(on_disk) == set(PACKAGES) | set(MODULES), on_disk
    for name in on_disk:
        assert name in vacuum.__all__, f"vacuum.__all__ is missing {name!r}"
        assert getattr(vacuum, name).__name__ == f"vacuum.{name}"
    assert "__version__" in vacuum.__all__


@pytest.mark.parametrize("pkg", PACKAGES + MODULES)
def test_package_all_resolves_and_has_no_duplicates(pkg):
    p = importlib.import_module(f"vacuum.{pkg}")
    names = list(p.__all__)
    assert names, f"vacuum.{pkg}.__all__ is empty"
    dups = sorted({n for n in names if names.count(n) > 1})
    assert not dups, f"vacuum.{pkg}.__all__ duplicates {dups}"
    unresolved = []
    for n in names:
        try:
            getattr(p, n)
        except ImportError as exc:  # a guided extra ImportError is the contract without the extra
            if "extra" in str(exc) and not (_has("jax") and _has("tenpy") and _has("qiskit")):
                continue
            unresolved.append((n, str(exc).splitlines()[0]))
        except AttributeError as exc:
            unresolved.append((n, str(exc)))
    assert not unresolved, f"vacuum.{pkg}: names in __all__ that do not resolve: {unresolved}"


@pytest.mark.parametrize("pkg", PACKAGES)
def test_submodule_all_is_subset_of_package_all(pkg):
    p = importlib.import_module(f"vacuum.{pkg}")
    exported = set(p.__all__)
    missing = {}
    for sm in pkgutil.iter_modules(p.__path__):
        full = f"vacuum.{pkg}.{sm.name}"
        if full in MIRROR_MODULES:
            assert sm.name in exported, f"{full} must be reachable as a namespace from vacuum.{pkg}"
            continue
        try:
            mod = importlib.import_module(full)
        except ImportError as exc:
            if "extra" in str(exc):
                continue  # JAX-top-level module without the extra: its names are lazy in the package
            raise
        sub_all = getattr(mod, "__all__", None)
        assert sub_all is not None, f"{full} has no __all__"
        gap = [n for n in sub_all if n not in exported]
        # vacuum.hardware.protocols.qet_* are served under the encoded_field_ prefix from vacuum.qet,
        # but from their own package they are exported verbatim, so no aliasing is needed here.
        if gap:
            missing[full] = gap
    assert not missing, f"public names not reachable from vacuum.{pkg}: {missing}"


def test_lazy_names_are_listed_in_dir():
    import vacuum.opt
    import vacuum.qet
    import vacuum.hardware
    for n in ("make_objective", "optimize", "inverse_design", "gjax", "gjax_detectors"):
        assert n in dir(vacuum.opt)
    for n in ("tfim_ground_mps", "ikeda_ibm_aer", "encoded_field_qet_dense"):
        assert n in dir(vacuum.qet)
    for n in ("vqe_ground_state", "qet_dense", "survival_sweep"):
        assert n in dir(vacuum.hardware)


def test_import_with_every_extra_blocked():
    """`import vacuum` and every subpackage work with all extras absent (README extras policy)."""
    script = textwrap.dedent(f"""
        import importlib, pkgutil, sys
        from importlib.abc import MetaPathFinder
        class Blocker(MetaPathFinder):  # a missing package: ModuleNotFoundError, sys.modules untouched
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {EXTRAS!r}:
                    raise ModuleNotFoundError("blocked: " + fullname, name=fullname)
        sys.meta_path.insert(0, Blocker())
        import vacuum
        guarded = {{"vacuum.opt.gjax", "vacuum.opt.gjax_detectors", "vacuum.opt.objectives",
                   "vacuum.opt.optimize", "vacuum.opt.inverse_design",
                   "vacuum.opt.multistart"}}
        bad = []
        for m in pkgutil.walk_packages(vacuum.__path__, "vacuum."):
            try:
                importlib.import_module(m.name)
            except ImportError as e:
                if not (m.name in guarded and "extra" in str(e)):
                    bad.append((m.name, repr(e)))
            except Exception as e:
                bad.append((m.name, repr(e)))
        for name in vacuum.__all__:
            getattr(vacuum, name)
        for pkg in {PACKAGES!r}:
            p = importlib.import_module("vacuum." + pkg)
            assert p.__all__
            for n in p.__all__:
                try:
                    getattr(p, n)
                except ImportError as e:
                    if not ("extra" in str(e)):
                        bad.append((pkg + "." + n, repr(e)))
                except Exception as e:
                    bad.append((pkg + "." + n, repr(e)))
        import vacuum.opt
        assert vacuum.opt.HAS_JAX is False
        print("BAD", bad)
        sys.exit(1 if bad else 0)
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-3000:]}"
