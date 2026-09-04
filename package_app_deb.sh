#!/usr/bin/env bash
# package_app_deb.sh
# Automates the packaging of Proximap for Debian/Ubuntu-based Linux distributions (.deb & portable .zip).

set -e

APP_NAME="proximap"
DISPLAY_NAME="Proximap"
VERSION="1.5.0"
ARCH=$(dpkg --print-architecture 2>/dev/null || echo "amd64")
MAINTAINER="ProximaXR Spatial Technologies <fumz@proximaxr.space>"
DESCRIPTION="Intuitive desktop 3D photogrammetry app."

echo "=========================================================="
echo " Starting Proximap Linux Bundling & Packaging (.deb)"
echo " Target Architecture: ${ARCH}"
echo "=========================================================="

# 1. Cleanup old build directories and artifacts
echo "[1/7] Cleaning up old build/dist files and logs..."
rm -rf build dist proximap_deb *.deb Proximap_Linux_Release.zip *.log
echo "  Cleaned up old files."

# 1b. Extract PyMeshLab wheel into backend_bin/pymeshlab_extracted/
#     PyMeshLab only ships cp310 (Python 3.10) ABI wheels. We extract the wheel
#     contents here so they can be bundled as --add-data regardless of the host
#     Python version running PyInstaller. The pymeshlab_worker.py subprocess then
#     uses a bundled Python 3.10 interpreter to load the extracted .so files.
echo "[2/7] Extracting PyMeshLab wheel into backend_bin/pymeshlab_extracted/..."
MLWHL=""
for whl in backend_bin/PymeshLab/pymeshlab-*manylinux*.whl backend_bin/PymeshLab/pymeshlab-*linux*.whl; do
    [ -f "$whl" ] && { MLWHL="$whl"; break; }
done
if [ -z "$MLWHL" ]; then
    echo "  [WARNING] No manylinux PyMeshLab wheel found in backend_bin/PymeshLab/. Skipping extraction."
else
    echo "  Extracting: $MLWHL"
    rm -rf backend_bin/pymeshlab_extracted
    python3 - << 'PYEOF'
import zipfile, os, sys, stat
whl = sys.argv[1] if len(sys.argv) > 1 else ""
import glob
wheels = glob.glob("backend_bin/PymeshLab/pymeshlab-*manylinux*.whl") + glob.glob("backend_bin/PymeshLab/pymeshlab-*linux*.whl")
if not wheels: sys.exit(0)
whl = wheels[0]
out_dir = "backend_bin/pymeshlab_extracted"
os.makedirs(out_dir, exist_ok=True)
prefix = next((n.split('/purelib/')[0] + '/purelib/' for n in zipfile.ZipFile(whl).namelist() if '/purelib/' in n), None)
if not prefix: sys.exit(1)
with zipfile.ZipFile(whl) as z:
    for name in z.namelist():
        if name.startswith(prefix):
            rel = name[len(prefix):]
            target = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if not name.endswith('/'):
                with z.open(name) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
                if target.endswith('.so') or '/bin/' in target:
                    os.chmod(target, os.stat(target).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
print("[2/7] Extraction complete:", out_dir)
PYEOF
    echo "  PyMeshLab wheel extracted successfully."
fi

# 2b. Download and bundle a standalone Python 3.10 interpreter (python-build-standalone)
#     This gives end-user machines a guaranteed cp310 runtime for pymeshlab_worker.py,
#     regardless of whatever Python is installed on their system.
PY310_DIR="backend_bin/python3.10"
PY310_BIN="$PY310_DIR/bin/python3.10"
if [ ! -f "$PY310_BIN" ]; then
    echo "[2b/7] Downloading standalone Python 3.10 interpreter..."
    PY310_TARBALL="cpython-3.10.17+20250725-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
    PY310_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20250725/${PY310_TARBALL}"
    mkdir -p "$PY310_DIR"
    if command -v curl > /dev/null 2>&1; then
        curl -L --retry 3 --retry-delay 5 -o "/tmp/${PY310_TARBALL}" "$PY310_URL" && \
            tar -xzf "/tmp/${PY310_TARBALL}" --strip-components=1 -C "$PY310_DIR" && \
            rm -f "/tmp/${PY310_TARBALL}" && \
            echo "  Python 3.10 standalone downloaded and extracted." || \
            echo "  [WARNING] Failed to download Python 3.10 standalone. PyMeshLab will use system python3.10 if available."
    elif command -v wget > /dev/null 2>&1; then
        wget -q --tries=3 -O "/tmp/${PY310_TARBALL}" "$PY310_URL" && \
            tar -xzf "/tmp/${PY310_TARBALL}" --strip-components=1 -C "$PY310_DIR" && \
            rm -f "/tmp/${PY310_TARBALL}" && \
            echo "  Python 3.10 standalone downloaded and extracted." || \
            echo "  [WARNING] Failed to download Python 3.10 standalone."
    else
        echo "  [WARNING] Neither curl nor wget available. Python 3.10 standalone not bundled."
    fi
else
    echo "[2b/7] Standalone Python 3.10 already present at $PY310_BIN — skipping download."
fi

# PyMeshLab imports NumPy from its ABI-matched sidecar runtime. Installing it
# here is required even though the main PyInstaller application bundles its own
# (potentially different-ABI) NumPy build.
if [ -x "$PY310_BIN" ]; then
    echo "[2c/7] Installing PyMeshLab sidecar runtime dependencies..."
    "$PY310_BIN" -m pip install --disable-pip-version-check "numpy==2.2.6"
fi

# 2. Icon check and setup
ICON_PATH="public/app_icon.png"
ICON_FLAG=""
if [ -f "$ICON_PATH" ]; then
    ICON_FLAG="--icon=$ICON_PATH"
fi

# 3. Run PyInstaller to package the Python GUI
# NOTE: --collect-all pymeshlab is intentionally OMITTED here — PyMeshLab ships
#       only cp310 ABI wheels and cannot be collected by a non-cp310 PyInstaller run.
#       Instead, the extracted wheel is bundled via --add-data below, and
#       pymeshlab_worker.py is invoked as a subprocess using the bundled Python 3.10.
echo "[3/7] Freezing Python application with PyInstaller..."
python3 -m PyInstaller --windowed --noconsole $ICON_FLAG --name Proximap \
    --collect-all PySide6 --collect-all vispy --collect-all numpy \
    --collect-all pillow --collect-all cv2 --collect-all trimesh \
    --collect-all pyrr --collect-all OpenGL --collect-all qrcode \
    --collect-all scipy --collect-all skimage --collect-all open3d \
    --collect-all mesh_editor --collect-all addons \
    --exclude-module lightglue --exclude-module torch --exclude-module torchvision --exclude-module nvidia --exclude-module triton \
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
    --add-data "addons:addons" \
    --add-data "pymeshlab_worker.py:." \
    main_window.py

if [ ! -d "dist/Proximap" ]; then
    echo "ERROR: PyInstaller compilation failed! 'dist/Proximap' not found."
    exit 1
fi
echo "  PyInstaller compilation successful."

# 3b. Copy the extracted PyMeshLab wheel and Python 3.10 interpreter into the bundle.
#     These go into _internal/ so pymeshlab_worker.py can find them at runtime.
echo "[3b/7] Injecting PyMeshLab cp310 extension and Python 3.10 runtime into bundle..."
INTERNAL="dist/Proximap/_internal"

if [ -d "backend_bin/pymeshlab_extracted" ]; then
    mkdir -p "$INTERNAL/backend_bin"
    cp -r backend_bin/pymeshlab_extracted "$INTERNAL/backend_bin/"
    chmod -R 755 "$INTERNAL/backend_bin/pymeshlab_extracted"
    echo "  PyMeshLab extracted files copied to bundle."
else
    echo "  [WARNING] backend_bin/pymeshlab_extracted not found — PyMeshLab unavailable in bundle."
fi

if [ -d "$PY310_DIR" ] && [ -f "$PY310_BIN" ]; then
    mkdir -p "$INTERNAL/backend_bin"
    # Copy the entire standalone Python 3.10 tree
    cp -r "$PY310_DIR" "$INTERNAL/backend_bin/"
    chmod -R 755 "$INTERNAL/backend_bin/python3.10"
    echo "  Standalone Python 3.10 runtime copied to bundle."
else
    echo "  [WARNING] Standalone Python 3.10 not found — worker will fall back to system python3.10."
fi

echo "[4/7] Pruning unnecessary build bloat..."
rm -rf dist/Proximap/_internal/triton 2>/dev/null || true
rm -rf dist/Proximap/_internal/torch/testing 2>/dev/null || true
rm -rf dist/Proximap/_internal/torch/include 2>/dev/null || true
rm -rf dist/Proximap/_internal/nvidia/nccl 2>/dev/null || true
find dist/Proximap/_internal -name "*.a" -delete 2>/dev/null || true

echo "  Pruning unused WebEngine shared libraries and resources..."
rm -rf dist/Proximap/_internal/libQt6WebEngineCore.so* 2>/dev/null || true
rm -rf dist/Proximap/_internal/PySide6/Qt/lib/libQt6WebEngineCore.so* 2>/dev/null || true
rm -rf dist/Proximap/_internal/PySide6/Qt/resources/qtwebengine* 2>/dev/null || true


# 5. Construct Debian package folder structure
echo "[5/7] Setting up Debian package directory hierarchy..."
DEB_DIR="dist/deb_build"
OPT_DIR="${DEB_DIR}/opt/proximap"
BIN_DIR="${DEB_DIR}/usr/bin"
APP_DIR="${DEB_DIR}/usr/share/applications"
ICON_DIR="${DEB_DIR}/usr/share/icons/hicolor/256x256/apps"
CONTROL_DIR="${DEB_DIR}/DEBIAN"

mkdir -p "$OPT_DIR"
mkdir -p "$BIN_DIR"
mkdir -p "$APP_DIR"
mkdir -p "$ICON_DIR"
mkdir -p "$CONTROL_DIR"

# Copy compiled PyInstaller app output
cp -r dist/Proximap/* "$OPT_DIR/"

# Copy backend binaries and toolchain files
echo "[5b/7] Copying backend dependencies and assets..."
COLMAP_DIR="$OPT_DIR/backend_bin/colmap"
OPENMVS_DIR="$OPT_DIR/backend_bin/openMVS"
PYMESHLAB_DIR="$OPT_DIR/backend_bin/PymeshLab"
mkdir -p "$COLMAP_DIR" "$OPENMVS_DIR" "$PYMESHLAB_DIR"

if [ -d "backend_bin/PymeshLab" ]; then
    cp -r backend_bin/PymeshLab/* "$PYMESHLAB_DIR/" 2>/dev/null || true
fi

if [ -d "backend_bin/colmap" ]; then
    cp -r backend_bin/colmap/* "$COLMAP_DIR/" 2>/dev/null || true
fi

# If local colmap binary is missing, copy system colmap if available
if [ ! -f "$COLMAP_DIR/colmap" ] && command -v colmap >/dev/null 2>&1; then
    echo "  Copying system colmap binary ($(command -v colmap)) into build..."
    cp "$(command -v colmap)" "$COLMAP_DIR/colmap"
fi

if [ -d "backend_bin/openMVS" ]; then
    echo "  Selectively copying required OpenMVS binaries..."
    for bin in InterfaceCOLMAP DensifyPointCloud ReconstructMesh RefineMesh TextureMesh; do
        if [ -f "backend_bin/openMVS/$bin" ]; then
            cp "backend_bin/openMVS/$bin" "$OPENMVS_DIR/"
        else
            echo "  [WARNING] Required OpenMVS binary not found: $bin"
        fi
    done
fi

# Prune Windows/macOS-specific binary artifacts
find "$OPT_DIR/backend_bin" -type f \( -name "*.exe" -o -name "*.dll" -o -name "*.bat" -o -name "*.pdb" -o -name "*.lib" -o -name "*.dylib" \) -delete 2>/dev/null || true
find "$OPT_DIR/backend_bin" -type d -empty -delete 2>/dev/null || true

# Copy project config & asset folders
if [ -f "toolchain_map.json" ]; then
    cp "toolchain_map.json" "$OPT_DIR/"
fi
if [ -d "backend_bin/sp_lg_weights" ]; then
    mkdir -p "$OPT_DIR/backend_bin/sp_lg_weights"
    cp -r backend_bin/sp_lg_weights/* "$OPT_DIR/backend_bin/sp_lg_weights/" 2>/dev/null || true
fi

if [ -d "models" ]; then
    cp -r "models" "$OPT_DIR/"
fi
if [ -d "public" ]; then
    cp -r "public" "$OPT_DIR/"
fi
if [ -d "addons" ]; then
    cp -r "addons" "$OPT_DIR/"
fi
if [ -d "mesh_editor" ]; then
    mkdir -p "$OPT_DIR/mesh_editor"
    cp -r mesh_editor/* "$OPT_DIR/mesh_editor/" 2>/dev/null || true
fi

# Set executable permissions
chmod -R 755 "$OPT_DIR"
chmod +x "$OPT_DIR/Proximap"

# Create symlink in /usr/bin/proximap
ln -sf /opt/proximap/Proximap "${BIN_DIR}/${APP_NAME}"

# Copy icon for system desktop integration
if [ -f "$ICON_PATH" ]; then
    cp "$ICON_PATH" "${ICON_DIR}/${APP_NAME}.png"
fi

# 6. Generate DEBIAN/control file
echo "[6/7] Generating DEBIAN/control file and Desktop entry..."
cat <<EOF > "${CONTROL_DIR}/control"
Package: ${APP_NAME}
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: ${MAINTAINER}
Section: graphics
Priority: optional
Depends: libc6, libgl1-mesa-glx | libgl1, libegl1, libxcb-xinerama0, colmap, libglu1-mesa, libgomp1, libmuparser2v5, libtbb12
Description: ${DESCRIPTION}
 Proximap is an intuitive desktop 3D photogrammetry GUI dashboard.
 Automates COLMAP and OpenMVS pipelines for 3D reconstruction.
EOF

# Generate desktop menu entry file
cat <<EOF > "${APP_DIR}/${APP_NAME}.desktop"
[Desktop Entry]
Name=${DISPLAY_NAME}
Comment=${DESCRIPTION}
Exec=/usr/bin/${APP_NAME}
Icon=${APP_NAME}
Terminal=false
Type=Application
Categories=Graphics;3DGraphics;Science;
Keywords=photogrammetry;3d;reconstruction;colmap;openmvs;
StartupWMClass=Proximap
EOF

chmod 644 "${APP_DIR}/${APP_NAME}.desktop"
chmod 644 "${CONTROL_DIR}/control"

# 7. Build .deb package and portable ZIP
DEB_FILE="${DISPLAY_NAME}_${VERSION}_${ARCH}.deb"
echo "[7/7] Building native .deb package: ${DEB_FILE}..."
dpkg-deb -z1 --root-owner-group --build "$DEB_DIR" "$DEB_FILE"
rm -rf "$DEB_DIR"

echo "Creating portable distribution ZIP..."
ZIP_FILE="Proximap_Linux_Release.zip"
rm -f "$ZIP_FILE"
cd dist
zip -r "../$ZIP_FILE" Proximap > /dev/null
cd ..

echo "=========================================================="
echo " SUCCESS! Proximap Linux packaging completed."
echo " Generated artifacts:"
echo "   1. Debian Package:  ${DEB_FILE}"
echo "   2. Portable ZIP:    ${ZIP_FILE}"
echo "=========================================================="
