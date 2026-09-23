#!/usr/bin/env bash
# Mechanism experiment: what sets the REALIZED gap length?
#
# Three arms differ in exactly one knob, --sieve-primes P (the sieve prime
# limit).  The survivor density of the sieve is
#     u(P) = prod_{p<=P} (1 - 1/p)  ~  e^-(ln ln P + 0.2615)
#     P = 2e3   -> u ~ 10.1 %
#     P = 5e4   -> u ~  7.1 %
#     P = 2e6   -> u ~  5.1 %      (the value the 12 h run used)
# so the arms span a factor ~2 in u.
#
# WHY: if a gap forms between two survivors that happen to be prime, the first
# prime after a position appears ~L/u adders later, i.e. the merit distribution
# would have a mode near 1/u (~10, ~14, ~20 for the three arms) and differ in
# SHAPE.  If instead the covering design pins the run length, all three arms
# look the same.  Either answer is useful: the first turns u into a dial for
# aiming at a chosen gap length, the second says the spanning structure is
# elsewhere and aiming must come from the cover.
#
# Floor 10 (not 20): the mode must be VISIBLE, and the p0 arms showed a floor
# of 10 still walks at ~500-650 win/s.  Everything else is identical, arms run
# sequentially on the same GPU, one timeout each, so a slow arm cannot bias a
# faster one.  NEVER edit this file while it runs (bash re-reads scripts).
set -u

COVER=data/crt/m23/shift1017_p130_covermax_m30.txt
MIN=10
SECS=2400          # 40 min per arm -> 2 h total

echo "=== mech_sieve_ab start $(date -u +%FT%TZ)  cover=$COVER min_merit=$MIN secs=$SECS ==="
for P in 2000 50000 2000000; do
    OUT="/tmp/mech_P${P}"
    rm -f "${OUT}.txt" "${OUT}.log" "${OUT}.state"
    echo "=== arm sieve-primes=$P  start $(date -u +%T) ==="
    env GAP_HUNT_BATCH=1024 timeout --kill-after=10 "$SECS" \
        ./bin/gapminer --gap-hunt --crt-file "$COVER" --gap-hunt-device 0 \
        --gap-hunt-min-merit "$MIN" --sieve-primes "$P" \
        --gap-hunt-state "${OUT}.state" --gap-hunt-out "${OUT}.txt" \
        > "${OUT}.log" 2>&1
    rc=$?
    n=$(wc -l < "${OUT}.txt" 2>/dev/null || echo 0)
    echo "=== arm sieve-primes=$P  done rc=$rc gaps=$n $(date -u +%T) ==="
done
echo "=== mech_sieve_ab all arms done $(date -u +%FT%TZ) ==="
