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

# 1b. Extract PyMeshLab macOS wheel & download standalone Python 3.10 interpreter
echo "[1b/5] Setting up PyMeshLab and standalone Python 3.10 runtime for macOS..."
if [ -d "backend_bin/PymeshLab" ]; then
    python3 - << 'PYEOF'
import os, zipfile, glob, stat

out_dir = "backend_bin/pymeshlab_extracted"
os.makedirs(out_dir, exist_ok=True)
whl_candidates = glob.glob("backend_bin/PymeshLab/*macosx*.whl")
if whl_candidates:
    target_whl = whl_candidates[0]
    print(f"  Extracting macOS wheel: {target_whl}")
    with zipfile.ZipFile(target_whl, 'r') as z:
        for info in z.infolist():
            if info.filename.startswith("pymeshlab/"):
                z.extract(info, out_dir)
                target = os.path.join(out_dir, info.filename)
                if target.endswith(".so") or target.endswith(".dylib"):
                    os.chmod(target, os.stat(target).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
print("  PyMeshLab macOS wheel extraction complete.")
PYEOF
fi

PY310_DIR="backend_bin/python3.10"
PY310_BIN="$PY310_DIR/bin/python3.10"
if [ ! -f "$PY310_BIN" ]; then
    echo "  Downloading standalone Python 3.10 for macOS (aarch64)..."
    PY310_TARBALL="cpython-3.10.21+20260825-aarch64-apple-darwin-install_only_stripped.tar.gz"
    PY310_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260825/${PY310_TARBALL}"
    mkdir -p "$PY310_DIR"
    if curl -L --retry 3 --retry-delay 5 -o "/tmp/${PY310_TARBALL}" "$PY310_URL" 2>/dev/null; then
        tar -xzf "/tmp/${PY310_TARBALL}" --strip-components=1 -C "$PY310_DIR"
        rm -f "/tmp/${PY310_TARBALL}"
        echo "  Python 3.10 macOS standalone downloaded and extracted."
    fi
fi

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
    --collect-all numpy --collect-all scipy \
    --collect-all vispy --collect-all imgui_bundle \
    --collect-all trimesh --collect-all pyrr --collect-all cv2 \
    --collect-all rembg --collect-all onnxruntime \
    --exclude-module PySide6.QtWebEngineCore \
    --exclude-module PySide6.QtWebEngineWidgets \
    --exclude-module PySide6.QtWebEngineQuick \
    --exclude-module PySide6.Qt3DCore \
    --exclude-module PySide6.Qt3DRender \
    --exclude-module PySide6.QtMultimedia \
    --exclude-module PySide6.QtMultimediaWidgets \
    --exclude-module PySide6.QtCharts \
    --exclude-module PySide6.QtDataVisualization \
    --exclude-module PySide6.QtDesigner \
    --exclude-module PySide6.QtQml \
    --exclude-module PySide6.QtQuick \
    --exclude-module PySide6.QtQuickWidgets \
    --exclude-module PySide6.QtVirtualKeyboard \
    --exclude-module PySide6.QtSql \
    --exclude-module PySide6.QtXml \
    --exclude-module matplotlib \
    --add-data "mesh_editor/shaders:mesh_editor/shaders" \
    --add-data "models:models" \
    --add-data "pymeshlab_worker.py:." main_window.py

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
PYMESHLAB_DIR="$MAC_OS_DIR/backend_bin/PymeshLab"

mkdir -p "$COLMAP_DIR"
mkdir -p "$OPENMVS_DIR"
mkdir -p "$PYMESHLAB_DIR"
mkdir -p "$MAC_OS_DIR/_internal/backend_bin"

if [ -d "backend_bin/PymeshLab" ]; then
    cp -r backend_bin/PymeshLab/* "$PYMESHLAB_DIR/" 2>/dev/null || true
fi

if [ -d "backend_bin/pymeshlab_extracted" ]; then
    cp -r backend_bin/pymeshlab_extracted "$MAC_OS_DIR/backend_bin/" 2>/dev/null || true
    cp -r backend_bin/pymeshlab_extracted "$MAC_OS_DIR/_internal/backend_bin/" 2>/dev/null || true
fi

if [ -d "$PY310_DIR" ] && [ -f "$PY310_BIN" ]; then
    cp -r "$PY310_DIR" "$MAC_OS_DIR/backend_bin/" 2>/dev/null || true
    cp -r "$PY310_DIR" "$MAC_OS_DIR/_internal/backend_bin/" 2>/dev/null || true
    chmod -R 755 "$MAC_OS_DIR/backend_bin/python3.10" 2>/dev/null || true
    chmod -R 755 "$MAC_OS_DIR/_internal/backend_bin/python3.10" 2>/dev/null || true
fi

# 4. Copy backend binaries selectively
echo "[4/5] Selectively copying backend toolchain dependencies..."

echo "  Copying COLMAP binaries..."
# macOS uses .dylib or no extension for executables. Assuming 'colmap' is the executable.
if [ -f "backend_bin/colmap/colmap" ]; then
    cp -r "backend_bin/colmap/"* "$COLMAP_DIR/"
else
    echo "  [WARNING] colmap executable not found in backend_bin/colmap"
fi

echo "  Selectively copying required OpenMVS binaries..."
if [ -d "backend_bin/openMVS" ]; then
    for bin in InterfaceCOLMAP DensifyPointCloud ReconstructMesh RefineMesh TextureMesh; do
        if [ -f "backend_bin/openMVS/$bin" ]; then
            cp "backend_bin/openMVS/$bin" "$OPENMVS_DIR/"
        else
            echo "  [WARNING] Required OpenMVS binary not found: $bin"
        fi
    done
else
    echo "  [WARNING] OpenMVS directory not found."
fi

echo "  Pruning Windows-only backend artifacts from macOS bundle..."
find "$MAC_OS_DIR/backend_bin" -type f \( -name "*.exe" -o -name "*.dll" -o -name "*.bat" -o -name "*.pdb" -o -name "*.lib" \) -delete
find "$MAC_OS_DIR/backend_bin" -type d -empty -delete

echo "  Pruning unused WebEngine shared libraries from macOS bundle..."
rm -rf "$MAC_OS_DIR/_internal/libQt6WebEngineCore.dylib"* 2>/dev/null || true
rm -rf "$MAC_OS_DIR/_internal/PySide6/Qt/lib/libQt6WebEngineCore.dylib"* 2>/dev/null || true
rm -rf "$MAC_OS_DIR/_internal/PySide6/Qt/resources/qtwebengine*dylib"* 2>/dev/null || true
rm -rf "$MAC_OS_DIR/_internal/PySide6/Qt/resources/qtwebengine"* 2>/dev/null || true

echo "  Copying toolchain map configuration..."
if [ -f "toolchain_map.json" ]; then
    cp "toolchain_map.json" "$MAC_OS_DIR/"
fi

echo "  Copying application icon for runtime usage..."
if [ -f "app_icon.ico" ]; then
    cp "app_icon.ico" "$MAC_OS_DIR/"
fi
if [ -f "app_icon.png" ]; then
    cp "app_icon.png" "$MAC_OS_DIR/"
fi

echo "  Copying UI icons and public assets..."
if [ -d "public" ]; then
    cp -r "public" "$MAC_OS_DIR/"
else
    echo "  [WARNING] Public directory not found."
fi

echo "  Copying offline AI models..."
if [ -d "models" ]; then
    cp -r "models" "$MAC_OS_DIR/"
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
