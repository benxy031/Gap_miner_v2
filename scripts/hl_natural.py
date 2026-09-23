#!/usr/bin/env python3
"""hl_natural.py — the Hardy-Littlewood NATURAL merit law, as a plot baseline.

WHY THIS EXISTS
---------------
Every sigma our hunt reports is an *assisted* number: a CRT cover decides which
candidates are ever offered to the primality test, so sigma_eff is the slope of
the distribution the covering produces, not the slope nature would give at the
same prime scale L = ln x.  Without a natural reference, "sigma = 1.33" is a
number with no scale.

The reference comes from the Hardy-Littlewood four-parameter consecutive-gap
model of P. Williams (titanV), "Hardy-Littlewood Four Parameter Consecutive
Prime Gap Model", 23 September 2026 — the PDF, the two coefficient CSVs and the
derivation script live in forum/.  That data is NOT vendored into this file: we
READ the CSVs at runtime (path configurable) and fit the growth of c1..c4 with
the gap length, so the provenance stays in one place and the author keeps the
attribution.

The model, in its own notation (equations 7/8 of the report):

    rho_g(L) = (S_g / L^2) * exp(-(c1/L + c2/L^2 + c3/L^3 + c4/L^4))
    Y_g(L)   = 1 / rho_g            (expected spacing in x)
    first occurrence: solve Y_g(L) = e^L

with L = ln x, g the (even) gap size, S_g the pair singular series and c1..c4
the inclusion-exclusion coefficients (c1 = B1, c2 = B1^2/2 - B2, ...).

WHAT WE NEED FROM IT
--------------------
At a fixed L the merit m = g/L, so the natural survival above a report
threshold m0 is a pure function of the exponent

    E(m) = c1/L + c2/L^2 + c3/L^3 + c4/L^4      (g = m*L)
    P_nat(merit >= m) / P_nat(merit >= m0) = exp(-(E(m) - E(m0)))

and its local slope dE/dm is the natural inverse-sigma.

MEASURED CONSEQUENCE (2026-09-23, our two corpus sizes)
------------------------------------------------------
    L = 528.9 (shift 507, 763-bit):  dE/dm = 1.0401  -> sigma_nat = 0.961
    L = 882.4 (shift 1017, 1273-bit): dE/dm = 1.0416 -> sigma_nat = 0.960

Two facts follow, and both matter more than the exact numbers:
  * the natural tail is essentially SIZE-FLAT over our whole range (0.15 %
    between the two corpora), so a measured size effect has to be argued, not
    assumed;
  * every sigma our hunts measure (1.26 .. 1.47) sits 31-53 % ABOVE it, so the
    ratio sigma_eff/sigma_nat is a covering-gain KPI: in slope terms ~1.39,
    i.e. x10 on the merit-28 rate, x330 on merit-40.

CAVEAT (must travel with every use)
-----------------------------------
The coefficient tables stop at g = 3600 (order 4) and g = 9990 (order 3), while
our gaps are 17k-25k, so ANY evaluation at our g is an extrapolation.  The
model itself diverges from the real record frontier beyond g ~ 1500-1800
(median model/real merit ratio 1.278, up to 1.97 at g ~ 3580), so treat the
baseline as a shape reference, never as an absolute prediction.  `slope()` and
`E()` never claim otherwise; callers should print `is_extrapolated(L, m)`.

Run directly for a table:
    scripts/hl_natural.py [L ...]
"""
import math
import os
import sys

# The CSVs use the "gap, ..., c1, ..., c4, ..., first_occurrence_ln_x, merit"
# layout; only the columns used here are named.
_COLS = ("gap", "S_g", "c1", "c2", "c3", "c4")

MIN_FIT_G = 100          # below this the coefficients are in a pre-asymptotic
                         # regime (c1/g moves from 0.36 to 0.99 by g=2310)
MAX_TABLE_G = 9990       # the largest gap any coefficient table reaches


def default_paths(root=None):
    """(order-4 csv, order-3 csv) inside <repo>/forum, or (None, None)."""
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    f = os.path.join(root, "forum")
    return (os.path.join(f, "hl_gap_4param_exact_parameters_with_first_occurrence.csv"),
            os.path.join(f, "hl_gap_3param_exact_parameters_with_first_occurrence.csv"))


def _read_csv(path):
    """{gap: {S_g, c1..c4}} for the rows that carry usable coefficients."""
    rows = {}
    with open(path, errors="ignore") as fh:
        head = fh.readline().rstrip("\n").split(",")
        idx = {name: i for i, name in enumerate(head)}
        missing = [c for c in _COLS if c not in idx]
        if missing:
            raise ValueError("%s: missing columns %s" % (path, missing))
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < len(head):
                parts += [""] * (len(head) - len(parts))
            try:
                g = int(parts[idx["gap"]])
            except ValueError:
                continue
            row = {}
            for c in _COLS[1:]:
                v = parts[idx[c]]
                row[c] = float(v) if v not in ("", "nan") else None
            rows[g] = row
    return rows


def _power_law(points):
    """Least-squares ln(v) = ln(a) + b*ln(g) -> (a, b).  Equal weights."""
    n = len(points)
    if n < 3:
        raise ValueError("need >=3 points for a power-law fit")
    sx = sum(math.log(g) for g, _ in points)
    sy = sum(math.log(v) for _, v in points)
    sxx = sum(math.log(g) ** 2 for g, _ in points)
    sxy = sum(math.log(g) * math.log(v) for g, v in points)
    den = n * sxx - sx * sx
    if den == 0.0:
        raise ValueError("degenerate power-law fit")
    b = (n * sxy - sx * sy) / den
    return math.exp((sy - b * sx) / n), b


class Natural:
    """The HL natural merit law for an arbitrary L = ln x.

    Instantiate with `load()`; `ok` is False when the forum CSVs are absent, in
    which case every method degrades to None and callers skip the baseline.
    """

    def __init__(self, rows4, rows3, order4_path, order3_path):
        self.rows4 = rows4
        self.rows3 = rows3
        self.order4_path = order4_path
        self.order3_path = order3_path
        self.max_g = max(max(rows4), max(rows3)) if (rows4 or rows3) else 0
        # Prefer order-4 rows (c1..c4); fall back to order-3 (c1..c3, c4 = 0).
        self._fits = {}
        for k in ("c1", "c2", "c3", "c4"):
            pts = []
            for g, row in rows4.items():
                if g >= MIN_FIT_G and row.get(k):
                    pts.append((float(g), row[k]))
            if len(pts) < 3:
                pts = []
                for g, row in rows3.items():
                    if g >= MIN_FIT_G and row.get(k):
                        pts.append((float(g), row[k]))
                self._has_order4 = False
            self._fits[k] = _power_law(pts) if len(pts) >= 3 else (0.0, 1.0)
        self.ok = True

    # ── coefficients and exponent ────────────────────────────────────────
    def c(self, k, g):
        """Extrapolated coefficient c_k at gap g (power law from the CSVs)."""
        a, b = self._fits[k]
        return a * (g ** b)

    def E(self, L, m):
        """The natural exponent at merit m for prime scale L (g = m*L)."""
        g = m * L
        return (self.c("c1", g) / L + self.c("c2", g) / L ** 2 +
                self.c("c3", g) / L ** 3 + self.c("c4", g) / L ** 4)

    def slope(self, L, m=24.0, dm=0.01):
        """dE/dm at merit m: the natural inverse-sigma (1/sigma_nat)."""
        return (self.E(L, m + dm) - self.E(L, m - dm)) / (2.0 * dm)

    def sigma(self, L, m=24.0):
        """sigma_nat: the natural tail scale at merit m, in merit units."""
        s = self.slope(L, m)
        return (1.0 / s) if s > 0 else float("inf")

    def survival(self, L, m0, ms):
        """P_nat(merit >= m) / P_nat(merit >= m0) for an iterable of merits."""
        e0 = self.E(L, m0)
        return [math.exp(-(self.E(L, m) - e0)) for m in ms]

    def is_extrapolated(self, L, m=24.0):
        """True when m*L lies beyond the largest tabulated gap."""
        return (m * L) > self.max_g

    def gain_slope(self, sigma_eff, L, m=24.0):
        """LOCAL ratio of our decay rate to nature's at merit m (>1 = gain).

        This is a per-merit-unit ratio, not a cumulative one; use gain() for
        the cumulative factor between two merits.
        """
        s = self.slope(L, m)
        if not sigma_eff or not s or sigma_eff <= 0:
            return None
        return s * sigma_eff

    def gain(self, merit, sigma_eff, L, m_ref=20.0):
        """Cumulative gain at `merit` relative to the report threshold m_ref.

        EXACT, not a local-slope extrapolation:

            gain(m) = P_ours(>=m)/P_ours(>=m_ref) / [P_nat(>=m)/P_nat(>=m_ref)]
                    = exp((E(m) - E(m_ref)) - (m - m_ref)/sigma_eff)

        i.e. the natural exponent difference minus our fitted exponential.
        Because dE/dm rises slowly with m, a local-slope product overstates
        the cumulative gain, so this form is the one to quote.
        """
        if not sigma_eff or sigma_eff <= 0:
            return None
        dE = self.E(L, merit) - self.E(L, m_ref)
        return math.exp(dE - (merit - m_ref) / sigma_eff)

    def describe(self, L, sigma_eff=None, m=24.0):
        """One-line, provenance-carrying summary for logs and captions."""
        s = self.sigma(L, m)
        txt = ("HL natural at L=%.1f: dE/dm=%.4f sigma_nat=%.3f"
               % (L, self.slope(L, m), s))
        if sigma_eff:
            txt += (" | sigma_eff=%.4f (local decay ratio %.3fx; cumulative "
                    "gain x%.1f at merit 28, x%.0f at 40)" % (
                        sigma_eff, self.gain_slope(sigma_eff, L, m) or 0.0,
                        self.gain(28.0, sigma_eff, L) or 0.0,
                        self.gain(40.0, sigma_eff, L) or 0.0))
        if self.is_extrapolated(L, m):
            txt += " [extrapolated beyond g=%d]" % self.max_g
        return txt


def load(root=None, path4=None, path3=None):
    """Build a Natural from the forum CSVs; returns None if unavailable."""
    p4, p3 = default_paths(root)
    path4 = path4 or p4
    path3 = path3 or p3
    rows4, rows3 = {}, {}
    try:
        if path4 and os.path.exists(path4):
            rows4 = _read_csv(path4)
        if path3 and os.path.exists(path3):
            rows3 = _read_csv(path3)
    except (OSError, ValueError):
        return None
    if not rows4 and not rows3:
        return None
    try:
        return Natural(rows4, rows3, path4, path3)
    except ValueError:
        return None


def _main(argv):
    hl = load()
    if hl is None:
        print("forum coefficient CSVs not found; nothing to report",
              file=sys.stderr)
        return 2
    print("coefficient source: %s" % os.path.basename(hl.order4_path))
    print("tabulated gaps: %d rows, max g=%d (fit uses g>=%d)"
          % (len(hl.rows4), max(hl.rows4), MIN_FIT_G))
    for k in ("c1", "c2", "c3", "c4"):
        a, b = hl._fits[k]
        print("  fit %s = %.6g * g^%.4f" % (k, a, b))
    Ls = [float(a) for a in argv] or [528.9, 882.4, 1404.9]
    print("\n%-10s %8s %10s %10s %8s" %
          ("L", "bits", "dE/dm@24", "sigma_nat", "extrap?"))
    for L in Ls:
        print("%-10.1f %8.0f %10.4f %10.3f %8s" %
              (L, L / math.log(2.0), hl.slope(L, 24.0), hl.sigma(L, 24.0),
               hl.is_extrapolated(L, 24.0)))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
