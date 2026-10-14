#!/usr/bin/env bash
# Fetch and build the pinned, unmodified LAME the oracle is validated against.
#
# The repo deliberately does NOT build against a hand-edited LAME checkout: the
# dumped xr/xr_hat are only meaningful relative to a known encoder, and a thesis
# result has to be reproducible from a verifiable source. This downloads upstream
# lame-3.100, checks its MD5, and builds it into third_party/ (gitignored).
set -euo pipefail

VERSION=3.100
MD5=83e260acbe4389b54fe08e0bdbf7cddb
URL="https://downloads.sourceforge.net/project/lame/lame/${VERSION}/lame-${VERSION}.tar.gz"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$(cd "$HERE/../.." && pwd)/third_party"
mkdir -p "$TP"
cd "$TP"

if [[ ! -f "lame-${VERSION}.tar.gz" ]]; then
    echo "downloading lame-${VERSION}..."
    curl -sL -o "lame-${VERSION}.tar.gz" "$URL"
fi

got=$(md5sum "lame-${VERSION}.tar.gz" | cut -d' ' -f1)
if [[ "$got" != "$MD5" ]]; then
    echo "error: checksum mismatch for lame-${VERSION}.tar.gz" >&2
    echo "       got  $got" >&2
    echo "       want $MD5" >&2
    exit 1
fi
echo "checksum ok ($MD5)"

[[ -d "lame-${VERSION}" ]] || tar xzf "lame-${VERSION}.tar.gz"
cd "lame-${VERSION}"

# NOANALYSIS must stay undefined — the analysis path is what exports xr/xr_hat.
[[ -f config.h ]] || ./configure --disable-shared --enable-static --disable-frontend >/dev/null
grep -q '/\* #undef NOANALYSIS \*/' config.h || {
    echo "error: NOANALYSIS is defined; the analysis path would be compiled out" >&2
    exit 1
}
make -j"$(nproc)" >/dev/null
echo "built $TP/lame-${VERSION}/libmp3lame/.libs/libmp3lame.a"
