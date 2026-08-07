#!/usr/bin/env bash
# package_app_deb.sh
# Automates the packaging of Proximap for Debian/Ubuntu-based Linux distributions (.deb & portable .zip).

set -e

APP_NAME="proximap"
DISPLAY_NAME="Proximap"
VERSION="1.4.0"
ARCH=$(dpkg --print-architecture 2>/dev/null || echo "amd64")
MAINTAINER="ProximaXR Spatial Technologies <fumz@proximaxr.space>"
DESCRIPTION="Intuitive desktop 3D photogrammetry app."

echo "=========================================================="
echo " Starting Proximap Linux Bundling & Packaging (.deb)"
echo " Target Architecture: ${ARCH}"
echo "=========================================================="

# 1. Cleanup old build directories and artifacts
echo "[1/6] Cleaning up old build/dist files and logs..."
rm -rf build dist proximap_deb *.deb Proximap_Linux_Release.zip *.log
echo "  Cleaned up old files."

# 2. Icon check and setup
ICON_PATH="public/app_icon.png"
ICON_FLAG=""
if [ -f "$ICON_PATH" ]; then
    ICON_FLAG="--icon=$ICON_PATH"
fi

# 3. Run PyInstaller to package the Python GUI
echo "[2/6] Freezing Python application with PyInstaller..."
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
    main_window.py

if [ ! -d "dist/Proximap" ]; then
    echo "ERROR: PyInstaller compilation failed! 'dist/Proximap' not found."
    exit 1
fi
echo "  PyInstaller compilation successful."

echo "  Pruning unnecessary PyTorch & CUDA build bloat (Triton, C++ headers, test suites)..."
rm -rf dist/Proximap/_internal/triton 2>/dev/null || true
rm -rf dist/Proximap/_internal/torch/testing 2>/dev/null || true
rm -rf dist/Proximap/_internal/torch/include 2>/dev/null || true
rm -rf dist/Proximap/_internal/nvidia/nccl 2>/dev/null || true
find dist/Proximap/_internal -name "*.a" -delete 2>/dev/null || true

echo "  Pruning unused WebEngine shared libraries and resources..."
rm -rf dist/Proximap/_internal/libQt6WebEngineCore.so* 2>/dev/null || true
rm -rf dist/Proximap/_internal/PySide6/Qt/lib/libQt6WebEngineCore.so* 2>/dev/null || true
rm -rf dist/Proximap/_internal/PySide6/Qt/resources/qtwebengine* 2>/dev/null || true


# 4. Construct Debian package folder structure
echo "[3/6] Setting up Debian package directory hierarchy..."
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
echo "[4/6] Copying backend dependencies and assets..."
COLMAP_DIR="$OPT_DIR/backend_bin/colmap"
OPENMVS_DIR="$OPT_DIR/backend_bin/openMVS"
mkdir -p "$COLMAP_DIR" "$OPENMVS_DIR"

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

# 5. Generate DEBIAN/control file
echo "[5/6] Generating DEBIAN/control file and Desktop entry..."
cat <<EOF > "${CONTROL_DIR}/control"
Package: ${APP_NAME}
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: ${MAINTAINER}
Section: graphics
Priority: optional
Depends: libc6, libgl1-mesa-glx | libgl1, libegl1, libxcb-xinerama0, colmap
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

# 6. Build .deb package and portable ZIP
DEB_FILE="${DISPLAY_NAME}_${VERSION}_${ARCH}.deb"
echo "[6/6] Building native .deb package: ${DEB_FILE}..."
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
