# experiments/ — parameter files mirroring published setups

This directory holds *data, not code*: machine-readable parameter sets that mirror
published experimental configurations, so that simulation layers (`vacuum.detectors`,
`vacuum.qet`, `vacuum.floquet`) can load the exact regime a real device operates in.
First tenants: circuit-QED entanglement harvesting (the Waterloo flux-qubit program,
tunable ultrastrong couplers, the 2026 variable-gap detector model) and the Ikeda IBM
quantum-energy-teleportation protocol.

## Naming scheme

One subdirectory per experimental platform, one file per published parameter set:

```
experiments/
  circuit_qed_harvesting/
    <firstauthor>_<year>_<source>.json      # e.g. teixido-bonfill_2026_table1.json
  ibm_qet/
    ikeda_2023_fig2.json                    # device, qubit pair, gate angles, shots
  dce_microwave/
    wilson_2011_table1.json                 # SQUID modulation depth, pump frequency, line temp
```

- `<source>` names the table, figure, or section the numbers were transcribed from
  (`table1`, `fig3b`, `sec4`). One file = one source location; never merge tables.
- Lowercase, hyphenated author names; the year is the publication year, not the arXiv
  year, unless only a preprint exists (then `<firstauthor>_<year>p_...`).
- Variants explored in simulation but not published get the suffix `_variant-<tag>` and
  MUST declare what changed (see provenance rules).

## Provenance rules (non-negotiable)

1. **Every file cites its source.** Mandatory top-level fields:
   `"source"` (full citation: authors, journal/arXiv id, year), `"location"` (table/
   figure/equation), `"retrieved"` (date of transcription), `"transcribed_by"`.
2. **Numbers are verbatim.** Values are copied as published, in the paper's own units,
   recorded in a `"units"` field per quantity. Unit conversion to program conventions
   (ħ = 1, lattice units) happens in loader code, never in the file.
3. **No silent tuning.** A file either mirrors a publication exactly or is a declared
   variant: variants carry a `"variant_of"` field naming the parent file and a
   `"changes"` dict listing every altered value with a one-line reason.
4. **Uncertainties travel with values** when the paper reports them
   (`"value"`/`"error"` pairs), because noise-threshold maps are only as honest as the
   error bars they inherit.
5. **Loaders validate.** `vacuum.detectors.load_experiment` (and siblings) refuse files
   missing the mandatory fields — provenance failures are load-time errors.

## Provenance tags (per quantity)

Every quantity in `circuit_qed_harvesting/` carries a `"provenance"` field, one of:

- `transcribed` — read verbatim from the source at the stated location;
- `plot_digitized` — an order-of-magnitude read off a figure (not verbatim; the file
  also lists these names under `"plot_digitized"`);
- `selection` — a grid whose every entry is a published number (each located in the
  note); only the choice of subset is ours;
- `derived` — arithmetic on transcribed numbers (formula in the note, inputs named in
  `"derived_from"` as `<name>` or `<file>:<name>`); never an estimate;
- `synthetic` — at least one entry is a simulation choice.

Each file also lists `"synthetic_quantities"` and `"derived_quantities"` at top level;
`tests/test_experiments_io.py` asserts that these lists equal the tagged sets, that the
file-level `"synthetic"` flag is true whenever the synthetic set is non-empty (a variant
with no synthetic number may still carry the flag if its comment declares that only the
*selection* of published operating points is a simulation choice), and that the union of synthetic quantities across the platform equals the documented list
(currently three sweep-extension grids in `janzen_2023_fig6_variant-noise-sweep.json`).
A number that was not read from a real source stays `synthetic` — no exceptions.

Files carrying numbers that are *published but not measured* (design values, circuit
simulations, simulated tomography) say so per quantity in a `"kind"` field
(`measured` / `design` / `simulated` / `assumed` / `stated` / `bound` / `calculated`).

### circuit_qed_harvesting/ — what is where

| file | what it is |
|---|---|
| `teixido-bonfill_2026_table1.json` | the harvesting proposal (PRA 113, 043732): Table I scenarios, Sec. VI.A fixed parameters, switching durations from Figs. 17-23, plus the derived `negativity_resolution_nats` |
| `teixido-bonfill_2026_table1_variant-switching.json` | three named operating points (spacelike / marginal / causal) assembled from the proposal's published t_d, d and T values; no synthetic quantity (file flag kept for the selection only) |
| `teixido-bonfill_2025_thesis-outlook.json` | the authors' stated expected operating temperature (PhD thesis, 2025) |
| `janzen_2023_fig6.json` | the measured tunable-coupler device (PRResearch 5, 033155) |
| `janzen_2023_fig6_variant-noise-sweep.json` | the Layer 2 sweep grids; three grids remain synthetic (temperature, relaxation, dephasing extensions) |
| `ren_2022_sec4-3.json` | the Waterloo group's MSc-thesis design and simulated tomography precision (source of the negativity floor) |
| `orgiazzi_2016_fig3.json` | measured T1 / Ramsey / echo dephasing of Lupascu-group flux qubits (different device; a measured floor) |
| `dicarlo_2009_fig3.json` | standard transmon two-qubit tomography uncertainties (measured cross-check) |

A parameter file that cannot answer "which table of which paper is this?" does not
belong here. Simulation results built on these files inherit their citations
automatically, which is what makes a preregistered noise-threshold map (the Layer 2
deliverable) checkable by the experimental groups it is written for.
