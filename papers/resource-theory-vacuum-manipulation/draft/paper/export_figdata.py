"""Export the manuscript's figure tables from the candidate's archived data.

Reads ../../data/*.npz (produced by ../../notebook.py; generator sha256 in each
file's __build__) and writes plain whitespace-separated tables to ./data/ for
pgfplots. Nothing is recomputed except the closed-form curve of Fig. 1, which
is Hotta's Eq. (11) divided by his Eq. (8) and is printed beside the archived
grid so the two can be compared.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)

# Fig. 1 -- minimal model: W_arrow / E_A against k/h (archived 33x33 grid) and Hotta's closed form.
z = np.load(DATA / "s2_hotta_grid.npz", allow_pickle=True)
W, EA = z["W_arrow"], z["E_A"]
h, k = z["h"], z["k"]
if h.ndim == 1 and k.ndim == 1:          # grid axes -> full mesh (rows follow h, columns k)
    hh, kk = np.meshgrid(h, k, indexing="ij")
    if hh.shape != W.shape:
        hh, kk = hh.T, kk.T
else:
    hh, kk = h, k
r = (kk / hh).ravel(); rate = (W / EA).ravel()
order = np.argsort(r)
np.savetxt(OUT / "fig1_grid.dat", np.column_stack([r[order], rate[order]]), header="k_over_h W_arrow_over_E_A", comments="")
rr = np.logspace(np.log10(r.min()), np.log10(r.max()), 400)
closed = (1 + 2 * rr**2) * (np.sqrt(1 + rr**2 / (1 + 2 * rr**2) ** 2) - 1)   # Hotta Eq. (11) / Eq. (8)
np.savetxt(OUT / "fig1_closed.dat", np.column_stack([rr, closed]), header="k_over_h rate_closed_form", comments="")
print(f"fig1: {r.size} grid points, max |grid - closed form| = "
      f"{np.max(np.abs(rate[order] - np.interp(r[order], rr, closed))):.2e} (interpolated), "
      f"max rate = {rate.max():.6f}")

# Fig. 2 -- TFIM chain, Alice at site 0: the one-site rotation E_B, the one-site CPTP potential W_site,
# the region-unitary entropic bound, and the negativity, against Bob's distance, for g = 1 and g = 0.5.
c = np.load(DATA / "s3_chain_rows.npz", allow_pickle=True)
for L in (10,):
    for g in (1.0, 0.5):
        m = (c["L"] == L) & (np.isclose(c["g"], g)) & (c["A"] == 0)
        idx = np.argsort(c["d"][m])
        cols = [c["d"][m][idx], c["E_B"][m][idx], c["W_site"][m][idx], c["bound_exact"][m][idx], c["negativity_AB"][m][idx]]
        np.savetxt(OUT / f"fig2_L{L}_g{g:g}.dat", np.column_stack(cols), header="d E_B W_site bound_region negativity", comments="")
        print(f"fig2 L={L} g={g}: {m.sum()} rows; W_full = {np.unique(np.round(c['W_full'][m], 6))}; "
              f"max W_site/E_B (E_B>1e-6) = {np.max((c['W_site'][m]/c['E_B'][m])[c['E_B'][m] > 1e-6]):.3f}")
