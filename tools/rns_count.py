#!/usr/bin/env python3
"""rns_count.py -- "count before claiming" for the RNS-16 / integer-MMA road.

Answers ONE question: can an RNS-16 Montgomery multiplication for our widths
beat the MEASURED CGBN Montgomery multiplication, on the multiply stream alone?
It deliberately charges only the multiply stream -- everything else (modular
corrections, base-extension matrix memory, register pressure, and above all the
SERIAL dependency chain) is listed as uncharted, because that is where every
previous bignum path in this repo lost 85-99% (see bench_rns_mont.cu).

MEASURED ANCHORS (RTX 3070, this box, refreshed 2026-09-24):
  CGBN AL=12 montmul   0.717 ns   <- bin/bench_fermat 12 40000 20 = 1,394,233 cand/s
                                    (device-aggregate; the 1102 ns figure printed by
                                     cuda_int_throughput is a LATENCY artifact)
  mul32 (IMAD)         4592.68 GMAC/s   bin/cuda_int_throughput 200000
  dp4a                 20591.01 GMAC/s
  mma8 register        67399.26 GMAC/s
  mma8 + residue repack 57950.44 GMAC/s  (the realistic shape: 0.86x register)

BIJ PARAMETERS THAT MATTER (the correction that invalidates earlier estimates):
  the Bajard-Imbert-Jullien RNS Montgomery needs M > 4N and M' > 2N, i.e. for a
  b-bit N with 16-bit moduli  K = ceil((b+2)/16)  AND  K' = ceil((b+1)/16).
  So K' ~= K, NOT K'=10: the extension base is as big as the main base.  An
  earlier note assumed K'=10 and got 0.0117 ns/step -- a 40x understatement of
  the conversion work.

Usage: rns_count.py [bits ...]   (default: 768 1024 1280)
"""
import math, sys

# ---- measured anchors (provenance above) -----------------------------------
CGBN_NS_768 = 0.717          # ns per Montgomery multiplication, AL=12 (768-bit)
IMAD_GMAC   = 4592.68        # 32-bit MAC with 64-bit accumulate
MMA_PACK    = 57950.44       # int8 MACs/s with the residue repack in the loop
MMA_REG     = 67399.26       # int8 MACs/s register-resident (upper bound)


def count(bits):
    words = bits // 32                      # CGBN 32-bit limbs
    cgbn_macs = 2 * words * words           # CIOS: 2*b^2 word-MACs
    K = math.ceil((bits + 2) / 16.0)        # M > 4N
    Kp = math.ceil((bits + 1) / 16.0)       # M' > 2N
    products = K * K + 2 * K * Kp           # multiply + 2 base extensions
    int8_macs = 4 * products                # 16-bit product = 4 int8 MACs
    ns = int8_macs / MMA_PACK               # MMA_PACK is MACs per ns (GMAC/s == MAC/ns)
    conv = 2 * K * Kp / products            # share of work that IS conversion
    return dict(bits=bits, words=words, cgbn_macs=cgbn_macs, K=K, Kp=Kp,
                products=products, int8_macs=int8_macs, ns=ns, conv=conv)


def main():
    bits_list = [int(a) for a in sys.argv[1:]] or [768, 1024, 1280]
    print(f"anchors: CGBN {CGBN_NS_768} ns/montmul @768-bit | IMAD {IMAD_GMAC} GMAC/s "
          f"| mma8+repack {MMA_PACK} GMAC/s")
    print()
    print(f"{'bits':>5} {'words':>5} {'CGBN MACs':>10} {'K':>3} {'K\'':>4} "
          f"{'16b prod':>9} {'int8 MACs':>10} {'RNS ns':>8} {'vs CGBN':>8} {'conv%':>6}")
    for b in bits_list:
        c = count(b)
        # scale the 768-bit measured CGBN ns by the MAC ratio (same kernel family)
        cgbn_scaled = CGBN_NS_768 * (c['cgbn_macs'] / 1152.0)
        print(f"{b:5d} {c['words']:5d} {c['cgbn_macs']:10d} {c['K']:3d} {c['Kp']:4d} "
              f"{c['products']:9d} {c['int8_macs']:10d} {c['ns']:8.3f} "
              f"{cgbn_scaled / c['ns']:7.2f}x {100*c['conv']:5.0f}%")
    print()
    c = count(768)
    print(f"768-bit detail:")
    print(f"  CGBN : {c['cgbn_macs']} int32 MACs (2*{c['words']}*{c['words']}) in "
          f"{CGBN_NS_768} ns  => {c['cgbn_macs']/CGBN_NS_768:.0f} GMAC/s = "
          f"{100*c['cgbn_macs']/CGBN_NS_768/IMAD_GMAC:.0f}% of the IMAD peak")
    print(f"  RNS  : {c['products']} 16-bit products = {c['int8_macs']} int8 MACs in "
          f"{c['ns']:.3f} ns (multiply stream only, at the mma8+repack rate)")
    print(f"  ratio: {CGBN_NS_768/c['ns']:.2f}x on the multiply stream alone"
          f"  (register-resident mma8 would give {CGBN_NS_768/(c['int8_macs']/MMA_REG):.2f}x)")
    print(f"  the conversion share of that work is {100*c['conv']:.0f}% "
          f"(2*K*K' = {2*c['K']*c['Kp']} of {c['products']})")
    print()
    print("NOT CHARGED HERE (and where every previous bignum path lost):")
    print("  - partial-product recombination: an int16 product needs 3 adds + 2 shifts")
    print("    to combine the int8 partials -> more ops than the MACs counted above")
    print("  - base extension is a SEQUENTIAL matrix-vector step (not a free trickle")
    print("    of independent MACs), so the latency chain grows with K")
    print("  - K+K' ~= 100 moduli of state per candidate: register pressure and")
    print("    memory traffic the register-resident MMA rate does not include")
    print("  - CGBN's 0.717 ns already includes ALL of its own non-multiply work")
    print()
    print("VERDICT (2026-09-24): the RNS multiply stream buys <= ~1.4-1.8x in the best")
    print("case, while CGBN sits at ~35% of the SAME-box IMAD peak (2.9x unused scalar")
    print("headroom in its own domain) and is latency-bound, not multiply-bound.  The")
    print("road stays closed; reopen only with a MEASURED RNS-16 768-bit Montgomery step")
    print("(not a stream rate) that beats 0.717 ns/montmul end to end.")


if __name__ == '__main__':
    main()
