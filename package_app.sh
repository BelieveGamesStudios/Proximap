#!/usr/bin/env bash
# package_app.sh
# Automates the packaging of Proximap for commercial distribution on macOS.

set -e

echo "============================================="
echo " Starting Proximap Commercial Packaging (macOS)"
echo "============================================="

# 1. Cleanup old build directories and logs
echo "[1/5] Cleaning up old build/dist files and logs..."
rm -rf build dist *.log
echo "  Cleaned up old files."

# 2. Run PyInstaller to package the Python GUI
echo "[2/5] Compiling Python application using PyInstaller..."

ICON_FLAG=""
if [ -f "app_icon.icns" ]; then
    ICON_FLAG="--icon=app_icon.icns"
elif [ -f "app_icon.png" ]; then
    # Create icns from png using sips if possible
    echo "  Found app_icon.png. Attempting to convert to app_icon.icns..."
    mkdir -p app_icon.iconset
    sips -z 16 16     app_icon.png --out app_icon.iconset/icon_16x16.png > /dev/null
    sips -z 32 32     app_icon.png --out app_icon.iconset/icon_16x16@2x.png > /dev/null
    sips -z 32 32     app_icon.png --out app_icon.iconset/icon_32x32.png > /dev/null
    sips -z 64 64     app_icon.png --out app_icon.iconset/icon_32x32@2x.png > /dev/null
    sips -z 128 128   app_icon.png --out app_icon.iconset/icon_128x128.png > /dev/null
    sips -z 256 256   app_icon.png --out app_icon.iconset/icon_128x128@2x.png > /dev/null
    sips -z 256 256   app_icon.png --out app_icon.iconset/icon_256x256.png > /dev/null
    sips -z 512 512   app_icon.png --out app_icon.iconset/icon_256x256@2x.png > /dev/null
    sips -z 512 512   app_icon.png --out app_icon.iconset/icon_512x512.png > /dev/null
    sips -z 1024 1024 app_icon.png --out app_icon.iconset/icon_512x512@2x.png > /dev/null
    iconutil -c icns app_icon.iconset
    rm -rf app_icon.iconset
    if [ -f "app_icon.icns" ]; then
        ICON_FLAG="--icon=app_icon.icns"
    fi
fi

python3 -m PyInstaller --windowed --noconsole $ICON_FLAG --name Proximap \
    --collect-all numpy --collect-all rembg --collect-all scipy \
    --collect-all pymatting --collect-all vispy --copy-metadata pymatting main_window.py

if [ ! -d "dist/Proximap.app" ]; then
    echo "PyInstaller compilation failed! 'dist/Proximap.app' not found."
    exit 1
fi
echo "  PyInstaller compilation complete."

# 3. Create distribution backend directories
echo "[3/5] Setting up backend binary directories..."
MAC_OS_DIR="dist/Proximap.app/Contents/MacOS"
COLMAP_DIR="$MAC_OS_DIR/backend_bin/colmap"
OPENMVS_DIR="$MAC_OS_DIR/backend_bin/openMVS"

mkdir -p "$COLMAP_DIR"
mkdir -p "$OPENMVS_DIR"

# 4. Copy backend binaries selectively
echo "[4/5] Selectively copying backend toolchain dependencies..."

echo "  Copying COLMAP binaries..."
# macOS uses .dylib or no extension for executables. Assuming 'colmap' is the executable.
if [ -f "backend_bin/colmap/colmap" ]; then
    cp -r "backend_bin/colmap/"* "$COLMAP_DIR/"
else
    echo "  [WARNING] colmap executable not found in backend_bin/colmap"
fi

echo "  Copying OpenMVS binaries..."
if [ -f "backend_bin/openMVS/DensifyPointCloud" ]; then
    cp -r "backend_bin/openMVS/"* "$OPENMVS_DIR/"
else
    echo "  [WARNING] OpenMVS binaries not found in backend_bin/openMVS"
fi

echo "  Pruning Windows-only backend artifacts from macOS bundle..."
find "$MAC_OS_DIR/backend_bin" -type f \( -name "*.exe" -o -name "*.dll" -o -name "*.bat" -o -name "*.pdb" -o -name "*.lib" \) -delete
find "$MAC_OS_DIR/backend_bin" -type d -empty -delete

echo "  Copying toolchain map configuration..."
if [ -f "toolchain_map.json" ]; then
    cp "toolchain_map.json" "$MAC_OS_DIR/"
fi

echo "  Copying offline background removal models..."
if [ -d "models" ]; then
    cp -r "models" "$MAC_OS_DIR/"
else
    echo "  [WARNING] Models directory not found."
fi

echo "  Copying UI icons and public assets..."
if [ -d "public" ]; then
    cp -r "public" "$MAC_OS_DIR/"
else
    echo "  [WARNING] Public directory not found."
fi

# Fix permissions
chmod -R 755 "$MAC_OS_DIR/backend_bin"

# 5. Compress the finalized distribution folder
echo "[5/5] Compressing distribution into release ZIP..."
ZIP_FILE="Proximap_Mac_Release.zip"
rm -f "$ZIP_FILE"

cd dist
zip -r "../$ZIP_FILE" Proximap.app > /dev/null
cd ..

echo "============================================="
echo " Proximap successfully packaged for macOS!"
echo " Release package: $ZIP_FILE"
echo "============================================="
