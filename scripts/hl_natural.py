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

SOURCES (precision order)
-------------------------
1. The distributed exact database
   (`forum/hl_gap_distributed/data/hl_gap_cumulants.csv`, community project,
   added 2026-10-04): B1 and B2 are EXACT for every even gap up to 90090,
   which covers our whole working range (15k..27k) -- and c1 = B1, c2 from B1,
   B2 are the two terms that dominate at our L.  c3 is exact for every gap up
   to 9990 plus 211 SAMPLED gaps from 10k to 90090: the code fits a power law
   to those (good to <0.8 % overall, ~0.3 % in our 15k-21k window; the forum
   tables' global fit is 11-15 % off there).  c4 stays fitted (its term is
   ~4e-5 at our L, i.e. irrelevant).
2. Fallback: power laws fitted to the forum/ four- and three-parameter tables
   (used when the database folder is absent -- e.g. fleet boxes -- or beyond
   g = 90090).

MEASURED CONSEQUENCE (updated 2026-10-04, exact c1,c2)
------------------------------------------------------
    L = 528.9 (shift 507):   dE/dm = 1.011  ->  sigma_nat = 0.990
    L = 676.2 (shift 720):   dE/dm = 1.009  ->  sigma_nat = 0.991
    L = 882.4 (shift 1017):  dE/dm = 1.006  ->  sigma_nat = 0.994

3-4 % larger than the old fitted reading (0.96): over 15k..21k the fit
over-predicted c1 by 2.0-2.2 % and c2 by 7-8 %, i.e. it decayed too fast.
The tail is still SIZE-FLAT (+0.5 % between the extreme sizes; the fitted law
claimed -0.15 %, an artifact of the fit), so a measured size effect has to be
argued, not assumed.  Our hunts measure sigma 1.25..1.5, i.e. 25-50 % above
nature -- that excess is the cover gain.  Measured on the live fleet s720 log
(2026-10-04, against this baseline): x1.23 per merit unit, cumulative x1.7 at
merit 24.4 -> x4.5 at 29.0.

NOTE ON slope(): with exact coefficients c1 carries an arithmetic term that
jumps between adjacent gaps (dc1 per +2 gap swings 0.2..3.6), so a POINTWISE
dE/dm oscillates by tens of percent.  E differences over >=1 merit unit are
unaffected (the term contributes <=0.003 there).  slope() therefore defaults
to a secant over +-1 merit unit whenever exact coefficients are active: a
trend value, accurate to ~0.3 %.  Pass dm explicitly for the raw derivative.

CAVEAT (must travel with every use)
-----------------------------------
c1 and c2 are exact only to g = 90090; c3, c4 and anything beyond that gap are
modelled.  The model diverges from the real record frontier at small g (median
model/real merit ratio 1.278 up to g ~ 3580), so treat the baseline as a SHAPE
reference, never as an absolute prediction; callers print `source_note(L, m)`
instead of assuming what is exact.

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


def default_db_path(root=None):
    """The distributed exact-coefficient database inside <repo>/forum."""
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "forum", "hl_gap_distributed", "data",
                        "hl_gap_cumulants.csv")


def _read_db(path):
    """{k: {gap: c_k}} from the distributed databases's cumulant CSV.

    Row = gap, order, c1..c12.  A row with order K carries EXACT c1..cK for
    that gap; higher cumulants are absent and must come from a fallback.
    """
    tables = {}
    with open(path, errors="ignore") as fh:
        head = fh.readline().rstrip("\n").split(",")
        idx = {name: i for i, name in enumerate(head)}
        if "gap" not in idx or "order" not in idx or "c1" not in idx:
            raise ValueError("%s: unexpected columns" % path)
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= idx["order"]:
                continue
            try:
                g = int(parts[idx["gap"]])
                o = int(parts[idx["order"]])
            except ValueError:
                continue
            for k in range(1, o + 1):
                name = "c%d" % k
                if name not in idx or idx[name] >= len(parts):
                    continue
                v = parts[idx[name]]
                if v:
                    try:
                        tables.setdefault(name, {})[g] = float(v)
                    except ValueError:
                        pass
    return tables


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

    def __init__(self, rows4, rows3, order4_path, order3_path,
                 db=None, db_path=None):
        self.rows4 = rows4
        self.rows3 = rows3
        self.order4_path = order4_path
        self.order3_path = order3_path
        self._db = db or {}
        self.db_path = db_path
        # Exact coverage of the distributed database: B1/B2 are complete to
        # g=90090 (c1, c2 = the dominant terms), so that is what E() trusts.
        self.exact_max_g = max(self._db.get("c1", {}), default=0)
        # c3: the SAMPLED exact rows above the exhaustive order-3 cutoff give
        # a power law that is ~40x more accurate than the forum tables' fit
        # in our evaluation window (0.3 % vs 11-15 %).
        self._c3_win = None
        self._c3_lo = self._c3_hi = 0
        win = sorted((g, v) for g, v in self._db.get("c3", {}).items()
                     if g > MAX_TABLE_G and v > 0)
        if len(win) >= 20:
            self._c3_lo, self._c3_hi = win[0][0], win[-1][0]
            try:
                self._c3_win = _power_law([(float(g), v) for g, v in win])
            except ValueError:
                self._c3_win = None
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
        """c_k at an even gap g (rounded to the nearest even integer).

        Exact from the distributed database when that gap is covered for this
        order; otherwise the sampled-window power law (c3) or the global
        power-law fit (all k, and everything beyond the database).
        """
        gk = int(round(g / 2.0)) * 2
        exact = self._db.get(k)
        if exact is not None:
            v = exact.get(gk)
            if v is not None:
                return v
        if (k == "c3" and self._c3_win is not None
                and self._c3_lo <= gk <= self._c3_hi):
            a, b = self._c3_win
            return a * (gk ** b)
        a, b = self._fits[k]
        return a * (gk ** b)

    def E(self, L, m):
        """The natural exponent at merit m for prime scale L (g = m*L)."""
        g = m * L
        return (self.c("c1", g) / L + self.c("c2", g) / L ** 2 +
                self.c("c3", g) / L ** 3 + self.c("c4", g) / L ** 4)

    def slope(self, L, m=24.0, dm=None):
        """dE/dm at merit m: the natural inverse-sigma (1/sigma_nat).

        dm=None picks the WIDE secant (+-1 merit unit) whenever exact
        coefficients are active at this (L, m): c1 carries an arithmetic term
        that jumps between adjacent gaps, so a pointwise derivative oscillates
        by tens of percent, while trend differences over >=1 merit unit are
        unaffected.  Pass dm explicitly for the raw pointwise derivative.
        """
        if dm is None:
            g = int(round((m * L) / 2.0)) * 2
            exact = self._db.get("c1") or {}
            dm = 1.0 if g in exact else 0.01
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

    def source_note(self, L=None, m=24.0):
        """One-line description of what is exact at this (L, m), for logs."""
        if not self._db:
            return ("power-law fits of the forum/ tables only (no exact "
                    "database); c1..c4 fitted, last tabulated g=%d"
                    % self.max_g)
        if L and L > 0 and m * L > self.exact_max_g:
            return ("all coefficients modelled beyond g=%d (exact c1,c2 only "
                    "to %d)" % (self.exact_max_g, self.exact_max_g))
        txt = "c1,c2 exact to g=%d (community DB)" % self.exact_max_g
        if self._c3_win:
            txt += ("; c3 sampled power law g=%d..%d"
                    % (self._c3_lo, self._c3_hi))
        else:
            txt += "; c3 fitted"
        return txt + "; c4 fitted"

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
        txt += " [%s]" % self.source_note(L, m)
        return txt


def load(root=None, path4=None, path3=None, db_path=None):
    """Build a Natural from the forum CSVs + exact DB; None if CSVs absent."""
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
    # The distributed exact database (community project, added 2026-10-04)
    # is OPTIONAL: absent on fleet boxes, and the tool then behaves exactly
    # as before (power-law fits only).
    db, used = None, None
    dbp = db_path or default_db_path(root)
    if dbp and os.path.exists(dbp):
        try:
            db = _read_db(dbp)
            used = dbp
        except (OSError, ValueError):
            db, used = None, None
    try:
        return Natural(rows4, rows3, path4, path3, db=db, db_path=used)
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
    if hl._db:
        print("exact DB: %s" % os.path.basename(hl.db_path or ""))
        print("  c1,c2 exact for every gap to g=%d (n=%d)"
              % (hl.exact_max_g, len(hl._db.get("c1", {}))))
        if hl._c3_win:
            print("  c3 sampled power law over g=%d..%d (n=%d)"
                  % (hl._c3_lo, hl._c3_hi,
                     sum(1 for g in hl._db.get("c3", {})
                         if g > MAX_TABLE_G)))
    else:
        print("exact DB not found -> power-law fits only (fleet behaviour)")
    Ls = [float(a) for a in argv] or [528.9, 882.4, 1404.9]
    print("\n%-10s %8s %10s %10s   %s"
          % ("L", "bits", "dE/dm@24", "sigma_nat", "source"))
    for L in Ls:
        print("%-10.1f %8.0f %10.4f %10.3f   %s" %
              (L, L / math.log(2.0), hl.slope(L, 24.0), hl.sigma(L, 24.0),
               hl.source_note(L, 24.0)))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
