"""Tests for vacuum.audits — the Layer-0 standing audits.

Covers the four hard walls as executable checks (PLAN.md, Layer 0):

1. Passivity (Pusz-Woronowicz 1978): random local symplectics never lower
   <H> on the chain vacuum — and the audit has teeth: on a locally squeezed
   (active) state it exhibits an energy-lowering local operation (the inverse
   squeeze) and reports the violation.
2. Causality (Lieb-Robinson 1972): the phi-phi commutator from the symplectic
   propagator vanishes outside the lattice light cone (super-exponential
   tails absorbed by a buffer), is order-1 inside the cone, and the
   propagator is symplectic to 1e-10.
3. Energy ledger: vacuum -> local squeeze (work in) -> loss channel
   (dissipation out) closes; and the closure *criterion* scales with the
   energies and the entry count actually booked, so a clean protocol cannot
   fire on its own float64 roundoff while a relative violation still fires at
   every size and a low-energy one is still held to the absolute floor.
4. Precision: a state engineered with min(nu) - 1/2 ~ 1e-13 (weak two-mode
   squeezing, reduced to one mode) trips the float64 guard, escalates to
   mpmath, and the escalated entropy matches the analytic small-r expansion
   S ~ r^2 (1 - 2 ln r).
"""

import numpy as np
import pytest

from vacuum.core import (
    evolve,
    ground_state_cov,
    harmonic_chain_K,
    loss,
    mean_energy,
    reduce,
    symplectic_eigenvalues,
    thermal_state_cov,
)
from vacuum.audits import (
    AuditViolation,
    MAX_OPS_PER_ENTRY,
    causality_audit,
    embed_local_symplectic,
    energy_ledger,
    passivity_audit,
    precision_audit,
)
from vacuum.protocol import ChannelStep, GeneratorStep, run_protocol


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chain16():
    """N=16, m=1 harmonic chain: coupling matrix and ground-state covariance."""
    K = harmonic_chain_K(16, m=1.0, bc="periodic")
    V = ground_state_cov(K)
    return K, V


def _local_squeeze(N, mode, r):
    """Single-mode squeeze S = diag(e^{-r}, e^{+r}) on (x_mode, p_mode), embedded."""
    S_sub = np.diag([np.exp(-r), np.exp(r)])
    return embed_local_symplectic(S_sub, [mode], N)


# ---------------------------------------------------------------------------
# 1. passivity
# ---------------------------------------------------------------------------


class TestPassivity:
    def test_vacuum_is_passive_under_random_local_symplectics(self, chain16):
        K, V = chain16
        res = passivity_audit(K, V, n_samples=200, rng=20260830)
        assert res.passed, f"passivity FAILED on the vacuum: {res}"
        deltas = res.details["delta_E"]
        assert len(deltas) == 200
        assert np.all(deltas >= -1e-12), (
            f"energy-lowering sample on the vacuum: min dE = {deltas.min():.3e}"
        )
        # sanity: the sampler is not trivial — some operations cost real energy
        assert np.max(deltas) > 1e-2

    def test_teeth_active_state_violation_is_exhibited(self, chain16):
        """On a locally squeezed state the inverse squeeze lowers <H> and the
        audit must report it — an audit that can never fire is untrustworthy."""
        K, V = chain16
        N = K.shape[0]
        r, mode = 0.5, 5
        S_sq = _local_squeeze(N, mode, r)
        V_active = evolve(V, S_sq)
        E_active = mean_energy(V_active, K)
        E_vac = mean_energy(V, K)
        assert E_active > E_vac + 1e-3  # squeezing injected real energy

        S_inv = _local_squeeze(N, mode, -r)  # the energy-lowering operation

        def inverse_squeeze_sampler(rng, n_modes):
            return S_inv

        res = passivity_audit(
            K,
            V_active,
            sampler=inverse_squeeze_sampler,
            n_samples=1,
            rng=0,
            strict=False,
        )
        assert not res.passed, "audit failed to fire on an active state"
        # the exhibited violation is exactly the squeezing energy, recovered
        assert res.worst_violation == pytest.approx(E_active - E_vac, rel=1e-10)
        assert np.allclose(res.details["worst_S"], S_inv)
        # strict mode raises loudly on the same input
        with pytest.raises(AuditViolation):
            passivity_audit(
                K, V_active, sampler=inverse_squeeze_sampler, n_samples=1, rng=0
            )
        # ...and the very same audit machinery passes on the true ground state,
        # so it distinguishes ground from active rather than always firing:
        res_vac = passivity_audit(K, V, n_samples=200, rng=20260830)
        assert res_vac.passed


# ---------------------------------------------------------------------------
# 2. causality
# ---------------------------------------------------------------------------


class TestCausality:
    def test_light_cone_and_symplectic_condition(self):
        K = harmonic_chain_K(200, m=0.01, bc="periodic")
        res = causality_audit(
            K,
            t=40.0,
            v_max=1.0,
            buffer=15.0,
            tol_commutator=1e-8,
            tol_symplectic=1e-10,
        )
        assert res.passed, f"causality FAILED: {res}"
        d = res.details
        assert d["max_abs_commutator_outside"] < 1e-8, (
            f"spacelike phi-phi commutator = {d['max_abs_commutator_outside']:.3e}"
        )
        # not vacuous: order-1 commutator somewhere inside the cone
        assert d["max_abs_commutator_inside"] > 0.1, (
            f"inside-cone commutator only {d['max_abs_commutator_inside']:.3e}"
        )
        assert d["symplectic_defect"] < 1e-10, (
            f"S Omega S^T - Omega defect = {d['symplectic_defect']:.3e}"
        )
        assert d["n_outside_pairs"] > 0

    def test_expm_cross_check(self):
        """Generic expm propagator agrees with the spectral one and passes."""
        K = harmonic_chain_K(200, m=0.01, bc="periodic")
        res = causality_audit(K, t=40.0, method="expm")
        assert res.passed
        assert res.details["max_abs_commutator_outside"] < 1e-8
        assert res.details["symplectic_defect"] < 1e-10

    def test_vacuous_configuration_raises(self):
        """A cone larger than the lattice cannot certify anything."""
        K = harmonic_chain_K(32, m=0.01, bc="periodic")
        with pytest.raises(ValueError, match="vacuous"):
            causality_audit(K, t=40.0, buffer=15.0)


# ---------------------------------------------------------------------------
# 3. energy ledger
# ---------------------------------------------------------------------------


class TestEnergyLedger:
    def test_squeeze_then_loss_closes(self, chain16):
        K, V0 = chain16
        N = K.shape[0]
        led = energy_ledger(K, tol=1e-9)

        E0 = led.record("vacuum", V0)

        # local squeeze: external drive puts work in (r large enough that the
        # subsequent loss channel is net-dissipative: a weakly squeezed chain
        # mode is *colder* in x than the environment vacuum I/2)
        S_sq = _local_squeeze(N, mode=4, r=1.0)
        V1 = evolve(V0, S_sq)
        E1 = mean_energy(V1, K)
        led.work("squeeze drive", E1 - E0)
        led.record("squeezed", V1)

        # loss channel: energy dissipated to the environment
        V2 = loss(V1, [4], eta=0.7)
        E2 = mean_energy(V2, K)
        assert E2 < E1  # the squeezed mode sheds energy into the environment
        led.dissipate("loss eta=0.7", E1 - E2)
        led.record("after loss", V2)

        res = led.close()
        assert res.passed, f"ledger did not close: {res}"
        assert abs(res.details["defect"]) < 1e-9, (
            f"closure defect = {res.details['defect']:.3e}"
        )
        table = res.details["table"]
        assert [row["kind"] for row in table] == [
            "snapshot", "work", "snapshot", "dissipation", "snapshot",
        ]
        assert res.details["delta_E"] == pytest.approx(E2 - E0, abs=1e-12)

    def test_single_mode_analytic_closure(self):
        """Closure against *independent* closed-form values (real power).

        The squeeze/loss test above computes its work and dissipation entries
        from the same mean_energy snapshots close() uses, so it checks the
        ledger's bookkeeping, not the physics. Here every entry comes from
        pencil-and-paper formulas for one oscillator H = (p^2 + w^2 x^2)/2:

          E_vac = w/2,   E_squeezed = (w/2) cosh 2r,
          E_after_loss = eta (w/2) cosh 2r + (1 - eta)(1 + w^2)/4,

        so closure now cross-checks ground_state_cov, evolve, loss and
        mean_energy against analytics — a wrong factor anywhere breaks it.
        """
        w, r, eta = 1.7, 0.8, 0.6
        K = np.array([[w**2]])
        V0 = ground_state_cov(K)
        E_vac = w / 2.0
        E_sq = (w / 2.0) * np.cosh(2.0 * r)
        E_loss = eta * E_sq + (1.0 - eta) * (1.0 + w**2) / 4.0

        led = energy_ledger(K, tol=1e-9)
        E0 = led.record("vacuum", V0)
        assert abs(E0 - E_vac) < 1e-12

        S_sq = np.diag([np.exp(-r), np.exp(r)])
        V1 = evolve(V0, S_sq)
        led.work("squeeze drive (analytic)", E_sq - E_vac)
        led.record("squeezed", V1)

        V2 = loss(V1, [0], eta=eta)
        led.dissipate("loss (analytic)", E_sq - E_loss)
        led.record("after loss", V2)

        res = led.close()
        assert res.passed, f"analytic ledger did not close: {res}"
        assert abs(res.details["defect"]) < 1e-9

        # power check: a 1% error in the claimed work must NOT close
        led_bad = energy_ledger(K, tol=1e-9)
        led_bad.record("vacuum", V0)
        led_bad.work("wrong drive", 1.01 * (E_sq - E_vac))
        led_bad.record("squeezed", V1)
        with pytest.raises(AuditViolation):
            led_bad.close()

    def test_cooked_books_raise(self, chain16):
        K, V0 = chain16
        led = energy_ledger(K)
        led.record("vacuum", V0)
        led.work("phantom drive", 1.0)  # claims work that never showed up in <H>
        led.record("still vacuum", V0)
        with pytest.raises(AuditViolation):
            led.close()
        res = led.close(strict=False)
        assert not res.passed
        assert res.worst_violation == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 3b. the closure criterion itself (scaled, not absolute)
# ---------------------------------------------------------------------------


def _driven_lossy_protocol(n_segments, temperature, n_modes=16, substeps=4):
    """A clean protocol whose books telescope exactly: only roundoff is left.

    `n_segments` alternating (loss channel, time-dependent drive) steps on an
    `n_modes` chain, run through the protocol runner so the drive's substep
    count reaches the ledger as the entry's ``n_ops``.  Raising
    `temperature` raises every booked energy without changing the physics of
    the closure identity; raising `n_segments` books more entries.
    """
    K = harmonic_chain_K(n_modes, m=1.0, bc="periodic")
    V0 = ground_state_cov(K) if temperature == 0.0 else thermal_state_cov(K, temperature)
    window, site = 4.0, n_modes // 2

    def K_of_t(t):
        K_t = K.copy()
        K_t[site, site] += 3.0 * np.sin(np.pi * t / window) ** 2
        return K_t

    ts = np.linspace(0.0, window, int(n_segments) + 1)
    steps = []
    for i in range(int(n_segments)):
        duration = float(ts[i + 1] - ts[i])
        steps.append(
            ChannelStep(
                channel=loss, args=([site],),
                kwargs={"eta": float(np.exp(-0.01 * duration))}, label=f"loss{i}",
            )
        )
        steps.append(
            GeneratorStep(H=K_of_t, duration=duration, n_substeps=substeps,
                          label=f"drive{i}")
        )
    led = energy_ledger(K)
    res = run_protocol(V0, K, steps, ledger=led, report_modes=[],
                       run_causality=False, strict=False)
    return led, res.audits["ledger"]


def _telescoping_ledger(omega, n_entries, r=0.7):
    """Single-mode squeeze booked as `n_entries` work rows that sum exactly.

    A synthetic ledger with a controllable entry count and a controllable
    energy scale (H = (p^2 + omega^2 x^2)/2, so <H> ~ omega/2): the exact
    handle needed to ask whether a *relative* violation still fires.
    """
    K = np.array([[omega**2]])
    V0 = ground_state_cov(K)
    V1 = evolve(V0, np.diag([np.exp(-r), np.exp(r)]))
    led = energy_ledger(K)
    E0 = led.record("in", V0)
    E1 = mean_energy(V1, K)
    for k in range(int(n_entries)):
        led.work(f"drive{k}", (E1 - E0) / float(n_entries))
    led.record("out", V1)
    return led, abs(E1)


class TestLedgerClosureCriterion:
    """The closure criterion scales with what was booked (docs/STATUS.md, L0).

    The identity `close()` checks telescopes exactly in exact arithmetic, so
    its only error is float64 roundoff — which grows with the energies being
    differenced and with the number of accumulations booked.  An *absolute*
    tolerance therefore fired on arithmetic in energetic, finely stepped
    protocols (the circuit-QED cutoff-bracket event) while under-checking
    small ones.  These tests pin the corrected criterion from both sides.
    """

    def test_entry_count_scaling_is_absorbed(self):
        """Clean protocols whose roundoff exceeds the old 1e-9 now pass.

        Same physics at five entry counts: the defect grows with the count
        (and with <H>), crosses the old absolute tolerance while nothing
        about the protocol is wrong, and the scaled criterion keeps it
        passing with two orders of margin.
        """
        sizes = (8, 64, 256, 512, 1024)
        defects, results = [], []
        for n in sizes:
            _, res = _driven_lossy_protocol(n, temperature=6000.0)
            defects.append(abs(res.details["defect"]))
            results.append(res)
        for n, d, res in zip(sizes, defects, results):
            assert res.passed, (
                f"clean protocol failed closure at {n} segments: defect "
                f"{d:.3e} vs tol {res.details['tol']:.3e}"
            )
        # the roundoff really does grow with the number of booked entries
        assert max(defects[2:]) > 10.0 * defects[0], (
            f"expected the defect to grow with the entry count, got {defects}"
        )
        # ...and it crosses the absolute 1e-9 that used to be the criterion
        assert sum(d > 1e-9 for d in defects) >= 2, (
            f"calibration lost: no clean run reaches the old tolerance, {defects}"
        )
        # the criterion that passes them is the scaled term, not the floor
        budget = results[-1].details["tolerance"]
        assert budget["relative"] > budget["atol"]
        assert budget["n_ops"] > 5000  # the drive's substeps reached the ledger
        assert budget["n_modes"] == 16

    def test_declared_n_ops_is_validated_and_auditable(self):
        """`n_ops` is caller-declared, so it is validated and recorded.

        It enters the tolerance linearly: an inflated declaration would
        loosen the caller's own audit. The ledger cannot verify a count
        against the integrator that produced it, so it does the three things
        it can — refuse a malformed or absurd count, keep every declaration
        on its row, and report the totals in the closure result.
        """
        K = harmonic_chain_K(4, m=1.0, bc="periodic")
        led = energy_ledger(K)
        for bad in (0, -3, 2.5, float("inf"), MAX_OPS_PER_ENTRY + 1):
            with pytest.raises(ValueError):
                led.work("bad declaration", 1.0, n_ops=bad)
            with pytest.raises(ValueError):
                led.dissipate("bad declaration", 1.0, n_ops=bad)
        for wrong_type in ("many", "12", b"12", None, [4], True):
            # a numeric string must be refused, not parsed: a str that gets
            # through here is a silent type coercion waiting to happen
            with pytest.raises(TypeError):
                led.work("bad declaration", 1.0, n_ops=wrong_type)
            with pytest.raises(TypeError):
                led.dissipate("bad declaration", 1.0, n_ops=wrong_type)
        assert led.table == [], "a refused declaration must not book a row"

        V0 = ground_state_cov(K)
        led.record("in", V0)
        led.work("drive", 0.0, n_ops=64)  # an integral float is accepted too
        led.dissipate("channel", 0.0, n_ops=8.0)
        led.record("out", V0)
        res = led.close(strict=False)
        assert [r["n_ops"] for r in res.details["table"]] == [1, 64, 8, 1]
        budget = res.details["tolerance"]
        assert budget["n_ops"] == 74 and budget["n_entries"] == 4
        assert budget["n_ops_excess"] == 70 and budget["max_n_ops"] == 64

    def test_a_one_percent_mis_booking_fires_at_every_size(self):
        """The teeth: 1 % of the booked work is a violation at any size."""
        for temperature in (0.0, 6000.0):
            for n in (8, 256, 1024):
                led, res = _driven_lossy_protocol(n, temperature=temperature)
                assert res.passed
                work = sum(r["value"] for r in led.table if r["kind"] == "work")
                led.work("mis-booked 1%", 0.01 * abs(work))
                bad = led.close(strict=False)
                assert not bad.passed, (
                    f"1% mis-booking passed at T={temperature}, {n} segments "
                    f"(defect {bad.details['defect']:.3e} vs tol "
                    f"{bad.details['tol']:.3e})"
                )
                with pytest.raises(AuditViolation):
                    led.close()

    def test_relative_violation_fires_at_every_energy_scale_and_entry_count(self):
        """A 1e-6 *relative* mis-booking fires whether <H> is 0.5 or 1000,
        and whether the ledger carries 4 entries or 4000 — the property the
        absolute criterion did not have."""
        for omega in (1.0, 2000.0):
            for n_entries in (4, 4000):
                led, scale = _telescoping_ledger(omega, n_entries)
                clean = led.close(strict=False)
                assert clean.passed, (
                    f"clean telescoping ledger failed at omega={omega}, "
                    f"{n_entries} entries: {clean.details['defect']:.3e}"
                )
                led.work("relative violation", 1e-6 * scale)
                with pytest.raises(AuditViolation):
                    led.close()

    def test_criterion_is_scale_covariant(self):
        """Rescaling every booked energy rescales the tolerance with it."""
        _, scale_lo = _telescoping_ledger(1.0, 16)
        led_lo, _ = _telescoping_ledger(1.0, 16)
        led_hi, scale_hi = _telescoping_ledger(1000.0, 16)
        t_lo = led_lo.closure_tolerance(tol=0.0)
        t_hi = led_hi.closure_tolerance(tol=0.0)
        assert t_hi["tol"] / t_lo["tol"] == pytest.approx(
            t_hi["scale"] / t_lo["scale"], rel=1e-12
        )

    def test_absolute_floor_still_checks_a_low_energy_protocol(self):
        """A protocol at <H> ~ 1e-6 is still held to 1e-12, not to <H>.

        This is where the new criterion is *stricter* than the 1e-9 it
        replaced: the injected defect is 1e-5 of the system's whole energy
        and 100x below the old tolerance.
        """
        led, scale = _telescoping_ledger(1e-6, 4)
        assert scale < 1e-5  # a genuinely tiny-energy protocol
        clean = led.close(strict=False)
        assert clean.passed
        assert clean.details["tol"] == pytest.approx(1e-12, rel=1e-12), (
            "the floor, not the relative term, must bind at this energy"
        )
        led.work("small but real", 1e-11)  # 1e-11 << the old 1e-9 tolerance
        bad = led.close(strict=False)
        assert not bad.passed
        assert abs(bad.details["defect"]) < 1e-9  # the old criterion missed it
        with pytest.raises(AuditViolation):
            led.close()


# ---------------------------------------------------------------------------
# 4. precision
# ---------------------------------------------------------------------------


class TestPrecision:
    def test_near_floor_flags_and_mpmath_matches_analytic(self):
        """Very weak two-mode squeezing: the reduced mode has
        nu - 1/2 = sinh^2 r ~ 1e-13, deep in the float64 danger zone; the
        escalated entropy must match the analytic small-r expansion
        S = r^2 (1 - 2 ln r) + O(r^4)  (nats), not merely self-agree."""
        r = 3.2e-7  # sinh^2 r ~ 1.02e-13
        c, s = np.cosh(r), np.sinh(r)
        A = np.array([[c, s], [s, c]])
        B = np.array([[c, -s], [-s, c]])
        S_tms = np.block([[A, np.zeros((2, 2))], [np.zeros((2, 2)), B]])
        V = evolve(np.eye(4) * 0.5, S_tms)  # two-mode squeezed vacuum
        V_A = reduce(V, [0])  # thermal-like mode, nu = cosh(2r)/2

        # the engineered gap really is ~1e-13
        gap64 = float(np.min(symplectic_eigenvalues(V_A))) - 0.5
        assert 1e-14 < gap64 < 1e-12

        res = precision_audit(V_A, tol=1e-12)
        assert res.details["flagged"], "float64 guard failed to trip near the floor"
        assert res.details["regime"] == "mpmath", (
            f"expected mpmath escalation, got {res.details['regime']}"
        )
        assert res.passed  # near the floor, but physical

        S_analytic = r**2 * (1.0 - 2.0 * np.log(r))
        S_audit = res.details["entropy"]
        rel = abs(S_audit - S_analytic) / S_analytic
        assert rel < 0.10, (
            f"escalated entropy {S_audit:.6e} vs analytic {S_analytic:.6e} "
            f"(rel err {rel:.2%})"
        )

    def test_comfortable_state_stays_float64(self, chain16):
        """A block of the gapped chain vacuum sits above the guard tolerance
        (its smallest gap is ~3e-7 — small, but resolvable in float64): no
        escalation."""
        K, V = chain16
        V_A = reduce(V, [0, 1, 2, 3])
        res = precision_audit(V_A, tol=1e-10)
        assert res.passed
        assert not res.details["flagged"]
        assert res.details["regime"] == "float64"
        assert res.details["gap"] > 1e-8

    def test_unphysical_state_fails_loudly(self):
        """Below the floor is a violation, not a flag."""
        V_bad = np.eye(4) * 0.4  # nu = 0.4 < 1/2: violates V + i Omega/2 >= 0
        with pytest.raises(AuditViolation):
            precision_audit(V_bad, tol=1e-10)
        res = precision_audit(V_bad, tol=1e-10, strict=False)
        assert not res.passed
        assert res.worst_violation == pytest.approx(0.1, abs=1e-9)
