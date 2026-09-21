#!/usr/bin/env python3
"""arith_ceiling.py -- paper model: could a DIFFERENT arithmetic backend beat our
Miller-Rabin kernel, and by how much, BEFORE anyone spends weeks rewriting it?

Two kinds of input, both hard to argue with:
  (1) measured integer retire rates of THIS GPU (tools/cuda_int_throughput.cu),
  (2) the exact sizes and op counts of our production kernel (AL limbs per shift,
      modmuls per MR test).

The model is deliberately pessimistic in three ways, because a model that only
charges the multiply always flatters the new backend:

  * every backend is charged a FULL operation mix per modular multiply, not just
    the multiply: the products, the reduction, the recombination of partial
    products, and the conditional subtract;
  * the tensor path is charged a packing efficiency ("util") because the
    m16n8k16 tile REDUCES over its k dimension while modular arithmetic is
    per-modulus -- the useful fraction of the 2048 MACs per instruction is a
    structural question, and it is swept, not assumed;
  * every backend is discounted by the SAME realisation factor: the fraction of
    its own paper ceiling that a real kernel actually reaches.  That factor is
    measured, not chosen -- our CGBN kernel reaches ~23% of its paper ceiling at
    768 bits, and the reasons (serial exponentiation chain, TPI structure,
    memory traffic, per-round overhead) are backend-independent in kind.

Usage:  python3 scripts/arith_ceiling.py [--json] [--realisation R]
        python3 scripts/arith_ceiling.py --help

Provenance of the rates: measured 2026-09-21, RTX 3070 (46 SM, sm_86, nvcc 12.4)
with `bin/cuda_int_throughput 400000`, two runs.  mul32 varies 4.8-5.3 T/s with
clock state; the others are stable to ~1%.
"""

import argparse
import json
import math

# ---------------------------------------------------------------- measured rates
GPU = dict(
    name="RTX 3070 (46 SM, sm_86, ~1.94 GHz)",
    mul64=1.10e12,      # 64-bit multiply /s          (CGBN limb primitive)
    imad=5.05e12,       # 32-bit IMAD + integer ALU /s (RNS 16-bit primitive)
    mul64hi=0.59e12,    # 64x64 -> high half /s
    dp4a=22.3e12,       # int8 MAC /s via __dp4a
    mma_int8=72.9e12,   # int8 MAC /s via m16n8k16 (89% of the INT8 peak)
)

# ------------------------------------------------------- our kernel's dimensions
HEADER_BITS = 256
SHIFTS = [26, 258, 720, 998]          # production / fleet shifts
REFERENCE_BITS = 768                  # 256 + shift 512: the AL=12 reference size
ACHIEVED_TESTS_S = 1.5e6              # measured CGBN kernel at the reference size
MR_WINDOW_SHARE = 0.79                # MR share of window time (77-84%)
MMA_UTILS = [1.0, 0.25, 1.0 / 16.0]   # packing efficiencies swept for the tensor row


def limbs(bits):
    """AL as the runtime picks it: ceil(bits/64) rounded up to even (CGBN)."""
    al = math.ceil(bits / 64.0)
    return al + (al % 2)


def modmuls_per_test(bits):
    """Base-2 Miller-Rabin on n-1: one squaring per exponent bit plus one
    multiply per set bit (about half of them)."""
    return 1.5 * bits


def rns_moduli(bits):
    """16-bit moduli needed to represent <bits>, plus one redundant channel."""
    return math.ceil(bits / 16.0) + 1


def backends(bits):
    """Per-backend operation mix for ONE <bits>-wide modular multiply.

    Each entry: (name, mac_ops, mac_unit, scalar_ops, note).
    mac_unit is the measured rate carrying mac_ops; scalar_ops are integer ALU /
    IMAD operations charged at the measured IMAD rate.
    """
    al = limbs(bits)
    k = rns_moduli(bits)
    return [
        # What we run today: CIOS/SOS Montgomery over 64-bit limbs.  No separate
        # scalar charge -- the limb multiplies ARE the whole cost.
        dict(name="CGBN 64-bit limbs (current)", mac=float(al * al), unit="mul64",
             rate=GPU["mul64"], scalar=0.0,
             note=f"AL={al}; SOS AL^2 limb muls (CIOS = 2x, shown below)"),
        # 16-bit RNS on the scalar ALU: per modulus one product IMAD, one
        # Montgomery reduction IMAD, ~3 integer ops (shift, add, conditional
        # subtract).  All carried by the same INT32 pipes.
        dict(name="RNS-16 scalar", mac=0.0, unit="imad", rate=GPU["imad"],
             scalar=5.0 * k,
             note=f"K={k} moduli x ~5 integer ops/modmul"),
        # 16-bit RNS via __dp4a: a 16x16->32 product needs 4 int8 products, done
        # as one dp4a; the reduction product is a second dp4a; recombination and
        # conditional subtract stay on the ALU (~5 ops).
        dict(name="RNS-16 dp4a", mac=8.0 * k, unit="dp4a", rate=GPU["dp4a"],
             scalar=5.0 * k,
             note=f"K={k} moduli x (2 dp4a + 5 ALU ops)"),
        # 16-bit RNS on int8 tensor cores: same op counts as dp4a, but the MACs
        # ride the tensor unit at a packing efficiency that is swept.
        dict(name="RNS-16 int8 MMA", mac=8.0 * k, unit="mma", rate=GPU["mma_int8"],
             scalar=5.0 * k,
             note=f"K={k} moduli x (8 int8 MACs + 5 ALU ops)"),
    ]


def cost_ns(b, mac_rate_multiplier=1.0):
    """Nanoseconds per modular multiply for one backend."""
    ns = 0.0
    if b["mac"]:
        ns += b["mac"] / (b["rate"] * mac_rate_multiplier) * 1e9
    if b["scalar"]:
        ns += b["scalar"] / GPU["imad"] * 1e9
    return ns


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--realisation", type=float, default=None,
                    help="fraction of its own paper ceiling a real kernel reaches "
                         "(default: measured from our CGBN kernel at 768 bits)")
    args = ap.parse_args()

    # The realisation factor is derived, not invented: look it up from the
    # measured CGBN rate at the reference size.
    al_ref = limbs(REFERENCE_BITS)
    ref_b = [b for b in backends(REFERENCE_BITS)
             if b["name"].startswith("CGBN")][0]
    nmod_ref = modmuls_per_test(REFERENCE_BITS)
    ref_ceiling = 1.0 / (cost_ns(ref_b) * nmod_ref * 1e-9)
    realisation = (args.realisation if args.realisation is not None
                   else ACHIEVED_TESTS_S / ref_ceiling)

    print(f"measured integer rates: {GPU['name']}")
    print(f"  mul64 {GPU['mul64']/1e12:.2f} T/s | 32-bit IMAD {GPU['imad']/1e12:.2f} T/s"
          f" | dp4a {GPU['dp4a']/1e12:.1f} T MAC/s"
          f" | int8 MMA {GPU['mma_int8']/1e12:.1f} T MAC/s")
    print(f"  one 64-bit limb mul = {GPU['imad']/GPU['mul64']:.2f} 32-bit MAC slots;"
          f"  int8 MMA = {GPU['mma_int8']/GPU['imad']:.1f}x the 32-bit MAC rate")
    print(f"\nreference size {REFERENCE_BITS} bits (AL={al_ref}):")
    print(f"  CGBN paper ceiling {ref_ceiling/1e6:.2f}M tests/s vs measured "
          f"{ACHIEVED_TESTS_S/1e6:.2f}M tests/s")
    print(f"  => realisation factor {realisation*100:.1f}% of the paper ceiling,"
          f" applied identically to every backend below")
    print(f"  (MR is {MR_WINDOW_SHARE*100:.0f}% of a window's time, so end-to-end"
          f" gain = 1 + (speedup-1) x {MR_WINDOW_SHARE:.2f})\n")

    out = dict(gpu=GPU, realisation=realisation, achieved_tests_s=ACHIEVED_TESTS_S,
               mr_window_share=MR_WINDOW_SHARE, sizes=[], verdict={})
    widths = (24, 9, 15, 11, 11, 11, 11)
    header = ("backend", "ops/mod", "unit", "ns/mod", "paper", "realist.",
              "vs CGBN")

    sizes = sorted(set([HEADER_BITS + s for s in SHIFTS] + [REFERENCE_BITS]))
    for bits in sizes:
        al = limbs(bits)
        nmod = modmuls_per_test(bits)
        print(f"=== {bits} bits (AL={al}, K={rns_moduli(bits)}, "
              f"{nmod:.0f} modmuls per MR test)" +
              ("   <-- reference size" if bits == REFERENCE_BITS else "") + " ===")
        print("  " + "  ".join(h.rjust(w) for h, w in zip(header, widths)))
        print("  " + "  ".join(("-" * len(h)).rjust(w)
                               for h, w in zip(header, widths)))
        rows = []
        for b in backends(bits):
            utils = MMA_UTILS if b["unit"] == "mma" else [1.0]
            for util in utils:
                ns = cost_ns(b, util)
                paper = 1.0 / (ns * nmod * 1e-9)
                real = paper * realisation
                label = b["name"] + (f" util={util:g}" if b["unit"] == "mma" else "")
                ops = b["mac"] + b["scalar"]
                rows.append(dict(name=label, unit=b["unit"], util=util,
                                 ops_per_modmul=ops, ns_per_modmul=ns,
                                 paper_tests_s=paper, realised_tests_s=real))
        # comparison column: every row against the CURRENT backend at THIS size
        cur = [r for r in rows if r["name"].startswith("CGBN")][0]
        for r in rows:
            rel = r["realised_tests_s"] / cur["realised_tests_s"]
            print("  " + "  ".join(c.rjust(w) for c, w in zip(
                (r["name"], f"{r['ops_per_modmul']:.0f}",
                 [b for b in backends(bits)
                  if r["name"].startswith(b["name"])][0]["unit"],
                 f"{r['ns_per_modmul']:.4f}", f"{r['paper_tests_s']/1e6:.2f}M",
                 f"{r['realised_tests_s']/1e6:.2f}M", f"{rel:.2f}x"), widths)))
        print(f"  note: {backends(bits)[0]['note']}; CIOS ordering would HALVE the "
              f"CGBN column: {cur['paper_tests_s']/2/1e6:.2f}M paper, "
              f"{cur['realised_tests_s']/2/1e6:.2f}M realised")
        mma_rows = [r for r in rows if "MMA" in r["name"]]
        best_mma = max(r["realised_tests_s"] for r in mma_rows)
        scalar_rns = [r for r in rows if "scalar" in r["name"]][0]
        out["sizes"].append(dict(bits=bits, al=al, modmuls=nmod, rows=rows,
                                 cios_paper_tests_s=cur["paper_tests_s"] / 2))
        print(f"  realised ceiling at {realisation*100:.0f}%: "
              f"CGBN {cur['realised_tests_s']/1e6:.2f}M | "
              f"RNS-16 scalar {scalar_rns['realised_tests_s']/1e6:.2f}M "
              f"({scalar_rns['realised_tests_s']/cur['realised_tests_s']:.2f}x) | "
              f"RNS-16 MMA util=1 {best_mma/1e6:.2f}M "
              f"({best_mma/cur['realised_tests_s']:.2f}x) tests/s")
        if bits == REFERENCE_BITS:
            print(f"  >>> the CGBN row here is the model's own anchor; the MEASURED "
                  f"rate is {ACHIEVED_TESTS_S/1e6:.2f}M tests/s.")
        print()

    # ------------------------------------------------------------- verdict
    ref_rows = [s for s in out["sizes"] if s["bits"] == REFERENCE_BITS][0]["rows"]
    ref_cur = [r for r in ref_rows if r["unit"] == "mul64"][0]
    ref_rns = [r for r in ref_rows if r["unit"] == "imad"][0]
    ref_mma1 = [r for r in ref_rows if r["unit"] == "mma" and r["util"] == 1.0][0]
    ref_mma_small = [r for r in ref_rows if r["unit"] == "mma"
                     and r["util"] == MMA_UTILS[-1]][0]
    ratio_rns = ref_rns["realised_tests_s"] / ref_cur["realised_tests_s"]
    ratio_mma = ref_mma1["realised_tests_s"] / ref_cur["realised_tests_s"]
    e2e = MR_WINDOW_SHARE
    print("=== verdict ===")
    print(f"1. The multiply carrier is NOT today's binding limit.  At "
          f"{REFERENCE_BITS} bits our kernel achieves {ACHIEVED_TESTS_S/1e6:.2f}M "
          f"tests/s")
    print(f"   against its own paper multiply ceiling of {ref_cur['paper_tests_s']/1e6:.2f}M"
          f" -- it uses {realisation*100:.0f}% of the multiply stream.  Whatever the"
          f" other")
    print("   ~77% of the time is, a backend that only speeds up multiplies"
          " inherits that share.")
    print(f"2. Multiply-count advantage at {REFERENCE_BITS} bits, same realisation"
          f" discount for all:")
    print(f"   RNS-16 scalar: {ratio_rns:.2f}x the current backend "
          f"({ref_rns['ops_per_modmul']:.0f} 32-bit ops vs "
          f"{ref_cur['ops_per_modmul']:.0f} limb muls, and one limb mul costs "
          f"{GPU['imad']/GPU['mul64']:.2f} IMAD slots)")
    print(f"   RNS-16 int8 MMA at util=1: {ratio_mma:.2f}x; at util=1/16: "
          f"{ref_mma_small['realised_tests_s']/ref_cur['realised_tests_s']:.2f}x")
    print("   The advantage GROWS with size (RNS uses K ~ bits channels while CGBN"
          " uses AL^2 ~ bits^2 limb muls):")
    for s in out["sizes"]:
        c = [r for r in s["rows"] if r["unit"] == "mul64"][0]
        r = [r for r in s["rows"] if r["unit"] == "imad"][0]
        m = [x for x in s["rows"] if x["unit"] == "mma" and x["util"] == 1.0][0]
        print(f"     {s['bits']:>5} bits: RNS-scalar {r['paper_tests_s']/c['paper_tests_s']:.1f}x,"
              f" RNS-MMA(util=1) {m['paper_tests_s']/c['paper_tests_s']:.1f}x vs CGBN-SOS"
              f" (paper, no realisation discount)")
    print(f"3. End-to-end cap: even a backend that made the MR stage infinitely fast"
          f" gains at most")
    print(f"   {1.0/(1.0-MR_WINDOW_SHARE):.2f}x window rate (MR is "
          f"{MR_WINDOW_SHARE*100:.0f}% of it).  A {ratio_rns:.1f}x multiply gain"
          f" becomes {1.0+(ratio_rns-1.0)*e2e:.2f}x end-to-end if nothing else"
          f" regresses.")
    print("4. The tensor road carries a structural tax this model can only bracket:"
          " the m16n8k16")
    print("   tile reduces over k, modular arithmetic is per-modulus, and the"
          " recombination and")
    print("   reduction of the partial products stay on the scalar ALU -- which is"
          " why the MMA rows")
    print("   are only ~2-3x the scalar RNS row even at util=1, and why util buys"
          " little beyond ~25%.")
    print("5. What would actually have to be true for this road to pay: the kernel"
          " must stop spending")
    print(f"   {100-realisation*100:.0f}% of its multiply budget elsewhere.  That is"
          " the same conclusion the")
    print("   recorded fingerprints reached: the open levers are a fuller/steadier"
          " MR batch and")
    print("   removing the residual per-round overhead -- not the multiply carrier.")

    out["verdict"] = dict(
        realisation=realisation,
        reference_bits=REFERENCE_BITS,
        current_paper_tests_s=ref_cur["paper_tests_s"],
        rns16_scalar_multiple=ratio_rns,
        rns16_mma_multiple_util1=ratio_mma,
        rns16_mma_multiple_util16=ref_mma_small["realised_tests_s"] /
                                  ref_cur["realised_tests_s"],
        e2e_cap=1.0 / (1.0 - MR_WINDOW_SHARE),
    )
    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
