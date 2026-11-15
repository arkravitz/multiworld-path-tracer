#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
version=2.5.1
archive="oidn-$version.arm64.macos.tar.gz"
checksum=98e0aca8e7ab69e9f4f0191582a500fd8b0d9085d68662f18ab6469e934a6efd
[ "$(uname -sm)" = 'Darwin arm64' ] || { echo 'This installer is for Apple Silicon macOS. See the OIDN downloads for other platforms.' >&2; exit 1; }
mkdir -p .deps
curl --fail --location --retry 2 --output ".deps/$archive" "https://github.com/OpenImageDenoise/oidn/releases/download/v$version/$archive"
printf '%s  %s\n' "$checksum" ".deps/$archive" | shasum -a 256 -c -
tar -xzf ".deps/$archive" -C .deps
printf 'Installed OIDN %s for this project.\n' "$version"
