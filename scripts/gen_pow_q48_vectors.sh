#!/usr/bin/env bash
#
# Regenerate tests/data/pow_q48_vectors.txt from the Gapcoin node's own
# proof-of-work implementation.
#
#   GAPCOIN_SRC=/path/to/Gapcoin scripts/gen_pow_q48_vectors.sh
#
# The oracle (tools/pow_q48_oracle.cpp) links src/PoWCore/PoWUtils.cpp from that
# tree, so the committed fixture is the node's arithmetic and not ours.  Never
# hand-edit the fixture: its entire value is that it comes from the node.
#
# Gapcoin is GPL-3.0-or-later, the same license as this repository.

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
src="${GAPCOIN_SRC:-$root/../Gapcoin}"
out="$root/tests/data/pow_q48_vectors.txt"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [ ! -f "$src/src/PoWCore/PoWUtils.cpp" ]; then
    echo "error: no Gapcoin source tree at $src (set GAPCOIN_SRC)" >&2
    exit 1
fi

g++ -O2 -o "$tmp/pow_q48_oracle" \
    "$root/tools/pow_q48_oracle.cpp" \
    "$src/src/PoWCore/PoWUtils.cpp" \
    -I"$src/src/PoWCore" -lgmpxx -lgmp

rev="$(git -C "$src" rev-parse --short HEAD 2>/dev/null || echo unknown)"

"$tmp/pow_q48_oracle" > "$out.tmp"

# Provenance line so a reviewer can see which node revision produced the file.
{
    head -1 "$out.tmp"
    echo "# gapcoin source: $src (rev $rev)"
    tail -n +2 "$out.tmp"
} > "$out"
rm -f "$out.tmp"

echo "wrote $out ($(grep -cv '^#' "$out") vectors) from $src (rev $rev)"
