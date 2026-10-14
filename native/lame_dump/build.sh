#!/usr/bin/env bash
# Build mp3dump against an out-of-tree LAME source tree.
#
# LAME is not vendored into this repo (see README.md for the licensing reason).
# Point LAME_DIR at a configured+built LAME 3.100 tree.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default: the pinned, unmodified upstream tree this repo builds itself.
# Set up with: bash native/lame_dump/setup_lame.sh
HERE_REPO="$(cd "$HERE/../.." && pwd)"
LAME_DIR="${LAME_DIR:-$HERE_REPO/third_party/lame-3.100}"

if [[ ! -f "$LAME_DIR/config.h" ]]; then
    echo "error: $LAME_DIR is not a configured LAME tree (no config.h)" >&2
    echo "       run: bash native/lame_dump/setup_lame.sh" >&2
    exit 1
fi

LIB="$LAME_DIR/libmp3lame/.libs/libmp3lame.a"
if [[ ! -f "$LIB" ]]; then
    echo "error: $LIB not found — run 'make' in $LAME_DIR" >&2
    exit 1
fi

# Static link is required: hip_set_pinfo is declared in the internal header
# libmp3lame/util.h and is not part of the installed shared-library API.
gcc -O2 -Wall -Wextra -Wno-unused-parameter \
    -DHAVE_CONFIG_H \
    -I"$LAME_DIR" \
    -I"$LAME_DIR/include" \
    -I"$LAME_DIR/libmp3lame" \
    -I"$LAME_DIR/mpglib" \
    -o "$HERE/mp3dump" \
    "$HERE/mp3dump.c" \
    "$LIB" \
    -lm

echo "built $HERE/mp3dump (against $LAME_DIR)"
