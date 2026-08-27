#!/usr/bin/env bash
# Copyright (C) 2026  GapMiner V2 contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Refreshes data/prime_gap_merits.txt from the live record table published at
# https://primegaps.cloudygo.com/merits.txt. Run this occasionally to pick up
# newly discovered records; gapminer_v2 itself never fetches this over the
# network at runtime (see --merit-records in README.md).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/data/prime_gap_merits.txt"
URL="https://primegaps.cloudygo.com/merits.txt"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

echo "[update_merits] Downloading $URL ..."
curl -fsS --max-time 30 "$URL" -o "$TMP"

lines=$(wc -l < "$TMP")
if [ "$lines" -lt 1000 ]; then
    echo "[update_merits] Downloaded file looks too small ($lines lines); not overwriting $DEST" >&2
    exit 1
fi

mv "$TMP" "$DEST"
trap - EXIT
echo "[update_merits] Updated $DEST ($lines lines)"
