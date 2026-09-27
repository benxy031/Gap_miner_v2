#!/usr/bin/env python3
"""HL natural model vs our measured hunt data: shape + absolute-rate calibration.

Two independent checks of the HL 4-parameter model (see scripts/hl_model.py and
/memories/repo/hl-4param-model.md) against our own gathers:

SHAPE (emission by gap length)
  For a walk corpus (data/gap_hunt_records_f1.txt, f2.txt) compare the per-length
  counts (all gaps >= m0*L) with the model weight
      w(g) = S_g / L^2 * exp(-(c1/L + c2/L^2 + c3/L^3 + c4/L^4)),
  normalized over the table range.  The observed/model ratio curve IS the measured
  cover response: for a fixed cover it should rise as exp((g/L)*(1/sigma_nat -
  1/sigma_cover)) inside the design window; the fitted slope is printed next to
  that prediction.  A CSV of the curve is written per corpus.

RATES (absolute, per window)
  For documented runs (constants below carry provenance) compare
      natural rate/window = W * sum_{g>=m0*L} w(g)
  with the observed gaps/window.  The ratio is the empirical price/gain of the
  cover at that (L, m0, cover) -- data for planning new hunts.

L convention: hunts anchor at 2^(255+shift) (gap_hunt default), so
L = (255+shift)*ln2 for walk datasets; corpora derive L from the data itself
(gap/merit of the first record).  winnability_map.py uses (256+shift) instead
(0.05-0.1% larger L) -- irrelevant at this precision.

Usage:
  python3 scripts/hl_calibrate.py                # both blocks
  python3 scripts/hl_calibrate.py --shape-only
  python3 scripts/hl_calibrate.py --rates-only
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LN2 = math.log(2.0)

# --- measured runs (provenance in comments; rates are gaps/window) -----------
RUNS = [
    dict(name="509 p74 strong m30 lex arm", shift=509, m0=8, W=2 * 15908,
         rate=70723 / 1043040, sigma=1.260, design=30,
         src="docs/CRT_DESIGN_MERIT_EXPERIMENT.md (600 s @1738.4 win/s)"),
    dict(name="509 p74 m30 random-residue arm", shift=509, m0=8, W=2 * 15908,
         rate=0.0256, sigma=0.978, design=30,
         src="docs/CRT_DESIGN_MERIT_EXPERIMENT.md (random r_p control)"),
    dict(name="720 p98 lex m40 arm", shift=720, m0=8, W=2 * 27061,
         rate=39250 / 442368, sigma=1.257, design=40,
         src="docs/CRT_DESIGN_MERIT_EXPERIMENT.md m40 section (600 s)"),
    dict(name="1784 p207 strong m30 walk", shift=1784, m0=21, W=84842,
         rate=288 / 20073472, sigma=None, design=30,
         src="commit aebf185 (18.7 h; 288 gaps >= m21)"),
]


def _s_g(g: int, primes: list[int]) -> float:
    """S_g = 2*C2 * prod_{p|g, p>2} (p-1)/(p-2)."""
    h = 1.0
    for p in primes:
        if p > g:
            break
        if g % p == 0:
            h *= (p - 1) / (p - 2)
    return 2.0 * 0.6601618158468696 * h


def _primes_upto(n: int) -> list[int]:
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = b"\x00" * len(sieve[i * i :: i])
    return [i for i in range(3, n + 1, 2) if sieve[i]]


def _rho(rec: dict, g: int, L: float, primes: list[int]) -> float:
    c1, c2, c3 = rec["c1"], rec["c2"], rec["c3"]
    c4 = rec.get("c4") or 0.0
    s = rec.get("S_g") or _s_g(g, primes)
    return s / L**2 * math.exp(-(c1 / L + c2 / L**2 + c3 / L**3 + c4 / L**4))


def natural_rate_per_win(L: float, m0: float, W: int, tab: dict) -> tuple[float, str]:
    """Model rate/window = W * sum_{g>=m0*L} rho_g(L).

    Dense coverage (>50 anchors in range): exact table sum + continuum tail.
    Sparse coverage: log-linear continuum fit over the in-range anchors (coarse;
    the head below the first anchor is extrapolated).
    """
    g_min = m0 * L
    primes = _primes_upto(int(max(tab)) + 1)
    anchors = [(g, _rho(tab[g], g, L, primes)) for g in sorted(tab)
               if g >= g_min and tab[g].get("c1") is not None]
    if len(anchors) < 50:
        if len(anchors) < 2:
            return float("nan"), f"only {len(anchors)} anchors >= {g_min:.0f}; not computed"
        (g1, r1), (g2, r2) = anchors[0], anchors[-1]
        b = math.log(r2 / r1) / (g2 - g1)
        a = math.log(r1) - b * g1
        total = 0.5 * math.exp(a + b * g_min) / (-b)  # even-g lattice integral
        note = (f"continuum fit over {len(anchors)} anchors ({g1}..{g2}); "
                f"head below {g1} extrapolated over {g1 - g_min:.0f} adders")
        if g1 - g_min > 3 * L:
            note += " -- COARSE, add anchors near the floor"
        return W * total, note
    total = sum(r for _, r in anchors)
    rtop = anchors[-1][1]
    tail = rtop * L / 2.0  # integral of rho_top * exp(-(g-gtop)/L) over even-g lattice
    note = f"table sum ({len(anchors)} gaps >= {g_min:.0f})"
    if tail > 1e-6 * total:
        note += f" + continuum tail {tail / total * 100:.2f}%"
    return W * (total + tail), note


# ------------------------------------------------------------------ shape ---

def load_corpus(path: Path) -> tuple[float, dict, int, int, int]:
    counts: dict[int, int] = {}
    L = None
    n = 0
    gmin = 10**9
    gmax = 0
    with path.open() as fh:
        for line in fh:
            tok = line.split()
            if len(tok) < 2:
                continue
            try:
                g = int(tok[0])
                merit = float(tok[1])
            except ValueError:
                continue
            if L is None:
                L = g / merit
            counts[g] = counts.get(g, 0) + 1
            n += 1
            gmin = min(gmin, g)
            gmax = max(gmax, g)
    return L, counts, n, gmin, gmax


def shape_block(out_dir: Path, m0: float) -> None:
    tab = hl_model.table()
    gmax_tab = max(tab)
    for tag in ("f1", "f2"):
        path = ROOT / f"gap_hunt_records_{tag}.txt"
        if not path.exists():
            path = ROOT / "data" / f"gap_hunt_records_{tag}.txt"
        if not path.exists():
            print(f"[shape] {tag}: corpus missing, skipped")
            continue
        L, counts, n, gmin, gmax = load_corpus(path)
        g_lo = int(math.ceil(m0 * L))
        sel = {g: counts[g] for g in counts if g_lo <= g <= gmax_tab and g in tab}
        if not sel:
            print(f"[shape] {tag}: no corpus bins inside table range, skipped")
            continue
        mod = {g: _rho(tab[g], g, L, _primes_upto(gmax_tab)) for g in sel}
        so, sm = sum(sel.values()), sum(mod.values())
        ratio = {g: (sel[g] / so) / (mod[g] / sm) for g in sel}
        # slope fit on bins with enough counts
        pts = [(g, math.log(ratio[g])) for g in sel if sel[g] >= 10]
        ng = len(pts)
        sx = sum(g for g, _ in pts)
        sy = sum(y for _, y in pts)
        sxx = sum(g * g for g, _ in pts)
        sxy = sum(g * y for g, y in pts)
        slope = (ng * sxy - sx * sy) / (ng * sxx - sx * sx)
        sig_nat = hl_model.natural_sigma(L, gmax_tab)
        slope_th = (1.0 / sig_nat - 1.0 / 1.26) / L  # 1.26 = measured cover sigma
        print(f"\n[shape] {tag}: L={L:.2f} (shift~{round(L / LN2 - 255)}), "
              f"records={n:,}, g={gmin}..{gmax}, bins in-table (g>={g_lo}): {len(sel)}")
        print(f"        slope d ln(obs/model)/dg = {slope:+.3e}/adder   "
              f"(predict (1/sigma_nat-1/sigma_cover)/L = {slope_th:+.3e}; "
              f"sigma_nat={sig_nat:.4f}, sigma_cover~1.26)")
        pts.sort()
        glo2, ghi2 = (pts[0][0], pts[-1][0]) if len(pts) >= 20 else (min(sel), max(sel))
        qs = [glo2 + int(q * (ghi2 - glo2)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
        print("        response at quartiles: " + "  ".join(
            f"g={q}:x{ratio[min(sel, key=lambda g: abs(g - q))]:.2f}" for q in qs))
        out = out_dir / f"hl_cover_response_{tag}.csv"
        with out.open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["gap", "order", "S_g", "c1", "c2", "c3", "c4",
                         "model_norm", "obs_norm", "ratio"])
            for g in sorted(sel):
                r = tab[g]
                wr.writerow([g, r.get("order"), r.get("S_g"), r.get("c1"),
                             r.get("c2"), r.get("c3"), r.get("c4"),
                             f"{mod[g]/sm:.6e}", f"{sel[g]/so:.6e}", f"{ratio[g]:.4f}"])
        print(f"        curve -> {out}")


# ------------------------------------------------------------------ rates ---

def rates_block() -> None:
    tab = hl_model.table()
    primes = _primes_upto(int(max(tab)) + 1)
    print(f"\n[rates] model table: {len(tab)} gaps (max {max(tab)}), "
          f"run constants carry provenance inside the script")
    hdr = f"{'run':34s} {'L':>7s} {'m0':>3s} {'design':>6s} {'obs/win':>10s} {'natural/win':>12s} {'gain':>7s}"
    print("  " + hdr)
    print("  " + "-" * len(hdr))
    for r in RUNS:
        L = (255 + r["shift"]) * LN2
        nat, note = natural_rate_per_win(L, r["m0"], r["W"], tab)
        gain = r["rate"] / nat if nat else float("nan")
        sig = f"{r['sigma']:.3f}" if r.get("sigma") else "  -  "
        print(f"  {r['name']:34s} {L:7.1f} {r['m0']:3d} {r['design']:6d} "
              f"{r['rate']:10.3e} {nat:12.3e} {gain:7.2f}  sigma={sig}")
        print(f"  {'':34s} [{note}]  src: {r['src']}")
    print("\n  gain = observed/model at that (L, m0, cover).  To plan a new walk: take the\n"
          "  nearest-L row, scale by e^{(m0'-m0)*(1/sigma_nat-1/sigma_cover)} (first-order),\n"
          "  then check with scripts/winnability_map.py --hl.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shape-only", action="store_true")
    ap.add_argument("--rates-only", action="store_true")
    ap.add_argument("--m0-shape", type=float, default=8.0,
                    help="walk report floor m0 for the shape block (default 8)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    (ROOT / "data").mkdir(exist_ok=True)
    if not args.rates_only:
        shape_block(args.out_dir, args.m0_shape)
    if not args.shape_only:
        rates_block()
    return 0


if __name__ == "__main__":
    sys.exit(main())
