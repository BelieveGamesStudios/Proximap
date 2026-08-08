#!/usr/bin/env bash
# package_app_snap.sh
# Automates packaging Proximap as a Linux Snap package (.snap) for Canonical Snap Store.

set -e

APP_NAME="proximap"
VERSION="1.4.0"
SNAP_FILE="${APP_NAME}_${VERSION}_amd64.snap"

echo "=========================================================="
echo " Starting Proximap Snapcraft Packaging (.snap)"
echo " Target Package: ${SNAP_FILE}"
echo "=========================================================="

# 1. Clean old snap build artifacts
echo "[1/5] Cleaning up previous snap build artifacts..."
rm -f *.snap

# 2. Ensure dist/Proximap is fully populated with backend dependencies and assets
echo "[2/5] Preparing dist/Proximap payload directory..."
if [ ! -d "dist/Proximap" ]; then
    echo "ERROR: 'dist/Proximap' directory not found. Please run PyInstaller first."
    exit 1
fi

mkdir -p dist/Proximap/backend_bin
mkdir -p dist/Proximap/models
mkdir -p dist/Proximap/public

if [ -d "backend_bin" ]; then
    cp -r backend_bin/* dist/Proximap/backend_bin/ 2>/dev/null || true
fi
if [ -d "models" ]; then
    cp -r models/* dist/Proximap/models/ 2>/dev/null || true
fi
if [ -d "public" ]; then
    cp -r public/* dist/Proximap/public/ 2>/dev/null || true
fi
if [ -f "toolchain_map.json" ]; then
    cp toolchain_map.json dist/Proximap/
fi

# Prune Windows and macOS binary artifacts from Linux snap payload
find dist/Proximap/backend_bin -type f \( -name "*.exe" -o -name "*.dll" -o -name "*.bat" -o -name "*.pdb" -o -name "*.lib" -o -name "*.dylib" \) -delete 2>/dev/null || true
find dist/Proximap/backend_bin -type d -empty -delete 2>/dev/null || true

# 3. Setup snap/gui assets
echo "[3/5] Setting up snap desktop integration..."
mkdir -p snap/gui
if [ -f "public/app_icon.png" ]; then
    cp public/app_icon.png snap/gui/proximap.png
fi

cat <<EOF > snap/gui/proximap.desktop
[Desktop Entry]
Name=Proximap
Comment=3D Scene Reconstruction & Photogrammetry Desktop Dashboard
Exec=proximap
Icon=\${SNAP}/meta/gui/proximap.png
Terminal=false
Type=Application
Categories=Graphics;3DGraphics;Science;
Keywords=photogrammetry;3d;reconstruction;colmap;openmvs;
StartupWMClass=Proximap
EOF

# 4. Trigger Snapcraft build
echo "[4/5] Executing Snapcraft pack..."
if command -v snapcraft >/dev/null 2>&1; then
    snapcraft --destructive-mode || snapcraft
else
    echo "ERROR: snapcraft is not installed. Install with: sudo snap install snapcraft --classic"
    exit 1
fi

# 5. Output summary
echo "[5/5] Verification of generated snap artifact..."
if ls *.snap 1> /dev/null 2>&1; then
    echo "=========================================================="
    echo " SUCCESS! Proximap Snap package successfully created:"
    ls -lh *.snap
    echo "=========================================================="
else
    echo "ERROR: Snap package creation failed. No .snap file generated."
    exit 1
fi
