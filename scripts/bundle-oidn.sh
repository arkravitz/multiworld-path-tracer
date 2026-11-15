#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
app_bundle=${1:-"$project_dir/release/Multiworld Path Tracer.app"}
runtime_dir="$project_dir/.deps/oidn-2.5.1.arm64.macos"
[ -d "$app_bundle/Contents/MacOS" ] || { echo 'Build the app bundle first.' >&2; exit 1; }
[ -f "$runtime_dir/lib/libOpenImageDenoise.2.dylib" ] || { echo 'Run scripts/install-oidn.sh first.' >&2; exit 1; }
mkdir -p "$app_bundle/Contents/Frameworks" "$app_bundle/Contents/Resources/OIDN"
cp -a "$runtime_dir"/lib/*.dylib "$app_bundle/Contents/Frameworks/"
# The upstream package expects its sibling libraries in ../lib. In the app
# they all live together in Frameworks, including dynamically loaded devices.
for library in "$app_bundle"/Contents/Frameworks/*.dylib; do
    [ -L "$library" ] && continue
    if otool -l "$library" | /usr/bin/grep -q '@loader_path/../lib'; then
        install_name_tool -rpath '@loader_path/../lib' '@loader_path' "$library"
    fi
done
cp "$runtime_dir/doc/LICENSE.txt" "$runtime_dir"/doc/third-party-programs*.txt "$app_bundle/Contents/Resources/OIDN/"
codesign --force --deep --sign - "$app_bundle"
codesign --verify --deep --strict "$app_bundle"
printf 'Bundled OIDN 2.5.1 and its notices into %s\n' "$app_bundle"
