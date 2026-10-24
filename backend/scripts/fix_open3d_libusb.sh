#!/usr/bin/env bash
# open3d 0.20 links libusb-1.0.so.0 (RealSense), missing on stock WSL. No-sudo fix:
# fetch the .deb and drop the .so next to open3d's module (its RPATH has $ORIGIN).
set -euo pipefail
cd "$(dirname "$0")/.."
O=$(.venv/bin/python -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('open3d').origin))")
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
(cd "$T" && apt-get download libusb-1.0-0 >/dev/null && dpkg-deb -x libusb-1.0-0_*.deb x)
cp -L "$T"/x/usr/lib/x86_64-linux-gnu/libusb-1.0.so.0 "$O/libusb-1.0.so.0"
.venv/bin/python -c "import open3d; print('open3d', open3d.__version__, 'imports OK')"
