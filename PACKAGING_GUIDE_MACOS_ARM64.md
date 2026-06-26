# Proximap: macOS ARM64 Packaging Guide

This guide covers the complete process for packaging Proximap on Apple Silicon (ARM64) macOS, from setting up your development environment to creating distributable packages. This guide is intended for **developers and contributors** building Proximap on macOS for local development or distribution.

**Note:** This is a *development/unsigned* packaging guide. For commercial distribution with code signing and Apple notarization, refer to `packaging_guide_commercial.md`.

---

## Table of Contents

1. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
2. [Step 1: Installing System Dependencies](#step-1-installing-system-dependencies)
3. [Step 2: Obtaining Backend Binaries](#step-2-obtaining-backend-binaries)
   - [Option A: Download Pre-compiled Binaries](#option-a-download-pre-compiled-binaries)
   - [Option B: Build from Source](#option-b-build-from-source)
4. [Step 3: Setting Up Python Environment](#step-3-setting-up-python-environment)
5. [Step 4: Compiling with PyInstaller](#step-4-compiling-with-pyinstaller)
6. [Step 5: Creating Distribution Packages](#step-5-creating-distribution-packages)
   - [ZIP Distribution](#zip-distribution)
   - [DMG Distribution](#dmg-distribution)
7. [Step 6: Verification & Testing](#step-6-verification--testing)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites & Environment Setup

### System Requirements

- **macOS Version:** 11.0 (Big Sur) or later
- **Architecture:** ARM64 (Apple Silicon / M1 / M2 / M3 and newer)
- **Disk Space:** 
  - 5 GB minimum (compiled app + dependencies)
  - 20-30 GB recommended if building from source (for COLMAP/OpenMVS builds)
- **RAM:** 8 GB minimum; 16+ GB recommended for parallel builds
- **Xcode Command Line Tools:** Required for compilation

### Verify Your Architecture

Open Terminal and verify you're on ARM64:

```bash
uname -m
```

Expected output: `arm64`

If you see `x86_64`, you're on an Intel Mac. This guide is ARM64-specific; Intel instructions differ significantly.

### Install Xcode Command Line Tools

```bash
xcode-select --install
```

Accept the license:

```bash
sudo xcodebuild -license accept
```

---

## Step 1: Installing System Dependencies

This guide uses Homebrew to manage dependencies. Install Homebrew if you haven't already:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install Required Packages

```bash
# Core build tools
brew install cmake ninja git

# Python (if not already installed)
brew install python@3.11

# Qt6 (required for COLMAP GUI and plugin support)
brew install qt

# Image libraries (for OpenMVS and general image processing)
brew install opencv boost libjpeg-turbo openexr webp

# Optional: for GPU acceleration (Metal support on Apple Silicon)
brew install metal-sdk

# Tools for DMG creation
brew install create-dmg

# Verification tools
brew install wget curl
```

### Verify Python Installation

```bash
python3.11 --version
```

Expected output: `Python 3.11.x`

---

## Step 2: Obtaining Backend Binaries

Proximap requires two main external binaries: **COLMAP** (Structure-from-Motion) and **OpenMVS** (Multi-View Stereo). You can either download pre-compiled binaries or build them from source.

### Option A: Download Pre-compiled Binaries

**Fastest path.** Pre-compiled ARM64 releases are available for both tools.

#### COLMAP Download

1. Visit the [COLMAP GitHub Releases page](https://github.com/colmap/colmap/releases)
2. Find the latest release with "macOS" in the assets (e.g., `COLMAP-x.y.z-mac-arm64.tar.gz` or similar)
3. Download the ARM64 macOS release:

```bash
cd ~/Downloads
wget https://github.com/colmap/colmap/releases/download/[VERSION]/COLMAP-[VERSION]-mac-arm64.tar.gz
```

4. Extract and copy to project:

```bash
tar -xzf COLMAP-*-mac-arm64.tar.gz
mkdir -p /path/to/Proximap/backend_bin/colmap
cp COLMAP.app/Contents/MacOS/colmap /path/to/Proximap/backend_bin/colmap/
cp -r COLMAP.app/Contents/Frameworks/* /path/to/Proximap/backend_bin/colmap/ 2>/dev/null || true
```

5. Verify:

```bash
/path/to/Proximap/backend_bin/colmap/colmap --version
```

#### OpenMVS Download

1. Visit the [OpenMVS GitHub Releases page](https://github.com/cdcseacave/openmvs/releases)
2. Find the latest macOS ARM64 release (if available; may be labeled as "macOS" or "Darwin ARM64")
3. Download:

```bash
cd ~/Downloads
wget https://github.com/cdcseacave/openmvs/releases/download/[VERSION]/OpenMVS-[VERSION]-macOS-arm64.tar.gz
```

4. Extract and copy to project:

```bash
tar -xzf OpenMVS-*-macOS-arm64.tar.gz
mkdir -p /path/to/Proximap/backend_bin/openMVS
cp OpenMVS/bin/* /path/to/Proximap/backend_bin/openMVS/
cp -r OpenMVS/lib/* /path/to/Proximap/backend_bin/openMVS/ 2>/dev/null || true
```

5. Verify each binary:

```bash
for tool in InterfaceCOLMAP DensifyPointCloud ReconstructMesh RefineMesh TextureMesh; do
    /path/to/Proximap/backend_bin/openMVS/$tool --help && echo "$tool: OK" || echo "$tool: FAILED"
done
```

#### Verify Binary Dependencies

Ensure all required libraries are present:

```bash
# Check COLMAP dependencies
otool -L /path/to/Proximap/backend_bin/colmap/colmap

# Check OpenMVS dependencies
otool -L /path/to/Proximap/backend_bin/openMVS/InterfaceCOLMAP
```

If any dylibs are missing, note the library names and install via Homebrew, then retry the binary download from a more recent release.

---

### Option B: Build from Source

**More involved but allows customization.** Choose this if you need specific features or pre-compiled binaries aren't available for your macOS version.

#### COLMAP Build from Source

1. Clone the repository:

```bash
cd ~/src
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout [latest-tag]
```

2. Install COLMAP-specific dependencies:

```bash
brew install glew cgal gflags google-glog sqlite
```

3. Create a build directory and configure with CMake:

```bash
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DBUILD_TESTING=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_OPENGL=ON \
  -DQt6_DIR=$(brew --prefix qt6)/lib/cmake/Qt6
```

**Flag Explanations:**
- `-DCMAKE_BUILD_TYPE=Release`: Optimized binary
- `-DCMAKE_OSX_ARCHITECTURES=arm64`: ARM64 only
- `-DBUILD_TESTING=OFF`: Skip test compilation (saves time)
- `-DWITH_CUDA=OFF`: Apple Silicon doesn't support native CUDA (Metal support available in newer COLMAP versions)
- `-DWITH_OPENGL=ON`: GUI rendering

4. Build:

```bash
make -j$(nproc)
```

This will take 30-60 minutes depending on your system. Progress will be visible; grab a coffee ☕.

5. Copy compiled binaries:

```bash
mkdir -p /path/to/Proximap/backend_bin/colmap
cp src/colmap /path/to/Proximap/backend_bin/colmap/
cp -r src/thirdparty/glfw/deps/* /path/to/Proximap/backend_bin/colmap/ 2>/dev/null || true
```

6. Verify:

```bash
/path/to/Proximap/backend_bin/colmap/colmap --version
```

#### OpenMVS Build from Source

1. Clone the repository:

```bash
cd ~/src
git clone https://github.com/cdcseacave/openmvs.git
cd openmvs
git checkout [latest-tag]
```

2. Install OpenMVS dependencies:

```bash
brew install boost openexr openimageio tbb

# Optional: for GPU support (if available)
# brew install cuda  # (not available natively on Apple Silicon)
```

3. Create build directory and configure:

```bash
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DBUILD_TESTS=OFF \
  -DENABLE_CUDA=OFF \
  -DENABLE_CERES=ON
```

4. Build (this can take 1-2 hours):

```bash
make -j$(nproc)
```

5. Copy compiled binaries:

```bash
mkdir -p /path/to/Proximap/backend_bin/openMVS
cp bin/* /path/to/Proximap/backend_bin/openMVS/
```

6. Verify:

```bash
/path/to/Proximap/backend_bin/openMVS/InterfaceCOLMAP --help | head -5
```

---

## Step 3: Setting Up Python Environment

### Create Virtual Environment

Navigate to your Proximap project directory and create a virtual environment:

```bash
cd /path/to/Proximap
python3.11 -m venv venv
source venv/bin/activate
```

### Install Python Dependencies

Upgrade pip and install project dependencies:

```bash
pip install --upgrade pip setuptools wheel

# Install PyInstaller and core dependencies
pip install PyInstaller pyside6 numpy scipy pillow psutil

# Install optional deep learning models (for background removal)
pip install rembg pymatting onnxruntime
```

### Create requirements.txt (Optional)

To save your current environment:

```bash
pip freeze > requirements_macos.txt
```

Later, you (or others) can install the same dependencies:

```bash
pip install -r requirements_macos.txt
```

### Verify Installation

```bash
python -c "import pyside6; import pyinstaller; print('Dependencies OK')"
```

---

## Step 4: Compiling with PyInstaller

PyInstaller converts your Python app into a standalone macOS `.app` bundle.

### Prepare the Build

Ensure your virtual environment is activated:

```bash
source venv/bin/activate
```

### Icon Conversion (PNG → ICNS)

The project requires a macOS icon file (`.icns` format). If you have a PNG icon:

```bash
# Convert PNG to ICNS using macOS native tools
# Assumes you have a 512x512 PNG icon file named 'icon.png'

mkdir icon.iconset
sips -z 16 16 icon.png --out icon.iconset/icon_16x16.png
sips -z 32 32 icon.png --out icon.iconset/icon_32x32.png
sips -z 64 64 icon.png --out icon.iconset/icon_64x64.png
sips -z 128 128 icon.png --out icon.iconset/icon_128x128.png
sips -z 256 256 icon.png --out icon.iconset/icon_256x256.png
sips -z 512 512 icon.png --out icon.iconset/icon_512x512.png

# Create the ICNS file
iconutil -c icns icon.iconset -o Proximap.icns
rm -rf icon.iconset
```

### Run PyInstaller

The project includes a `Proximap.spec` file that configures PyInstaller. Review it first:

```bash
cat Proximap.spec | head -50
```

Key macOS settings in the spec:
- `windowed=True`: Creates a GUI app without console window
- Icon reference (ensure correct path to `.icns` file)
- Hidden imports for NumPy and OpenCV
- Data files (UI resources, models)

Run PyInstaller:

```bash
pyinstaller Proximap.spec
```

This will take 5-15 minutes depending on dependency size and your system. Output will be in `dist/Proximap.app`.

### Post-Compilation Optimization

Remove unnecessary files to reduce package size (~200-300 MB savings):

```bash
# Navigate to the compiled app
cd dist/Proximap.app/Contents/Frameworks

# Remove test and debug files
find . -name "*test*" -delete
find . -name "*Test*" -delete
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove debug symbols (optional; slightly reduces size)
# find . -name "*.dSYM" -type d -exec rm -rf {} + 2>/dev/null || true

cd /path/to/Proximap
```

### Fix Executable Permissions

Ensure the app bundle has correct permissions:

```bash
chmod +x dist/Proximap.app/Contents/MacOS/Proximap
find dist/Proximap.app -type f -perm +111 -exec chmod +x {} \;
```

---

## Step 5: Creating Distribution Packages

### Common Setup for Both Formats

Create a staging directory for your distribution:

```bash
mkdir -p dist/Proximap_Distribution
cd dist/Proximap_Distribution

# Copy the compiled app
cp -r ../Proximap.app .

# Copy backend binaries and libraries
cp -r ../../backend_bin ./

# Create a README for users
cat > README.txt << 'EOF'
Proximap - Structure-from-Motion and Multi-View Stereo Pipeline
================================================================

Installation:
1. Extract this archive to your desired location
2. Double-click Proximap.app to run

Requirements:
- macOS 11.0 or later
- Apple Silicon (ARM64) Mac
- 8 GB RAM minimum (16+ GB recommended)
- Disk space for project files and reconstructions

For more information, see the main README.md in the project directory.
EOF

cd /path/to/Proximap
```

### ZIP Distribution

Simple, cross-platform format. Ideal for sharing among developers.

```bash
cd dist

# Create ZIP archive
zip -r Proximap_MacOS_ARM64_Release.zip Proximap_Distribution/

# Verify integrity
unzip -t Proximap_MacOS_ARM64_Release.zip | head -20
echo "ZIP created successfully: $(ls -lh Proximap_MacOS_ARM64_Release.zip | awk '{print $5, $9}')"
```

The resulting `Proximap_MacOS_ARM64_Release.zip` can be distributed directly.

### DMG Distribution

Professional macOS installer format. Users mount the DMG and drag the app to Applications.

#### Step 1: Create DMG Structure

```bash
cd dist/Proximap_Distribution

# Create a temporary folder for DMG contents
mkdir -p dmg_temp

# Copy app and documentation
cp -r Proximap.app dmg_temp/
cp -r backend_bin dmg_temp/
cp README.txt dmg_temp/

# Create a symlink to /Applications for easy drag-and-drop installation
ln -s /Applications dmg_temp/Applications

# Optional: Create a simple .webloc file pointing to documentation
cat > dmg_temp/Website.webloc << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>URL</key>
    <string>https://github.com/yourusername/proximap</string>
</dict>
</plist>
EOF
```

#### Step 2: Create DMG

Using the `create-dmg` tool (installed earlier) for automated creation:

```bash
create-dmg \
  --volname "Proximap" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "Proximap.app" 150 190 \
  --icon "Applications" 450 190 \
  --icon "README.txt" 300 320 \
  --hide-extension "README.txt" \
  Proximap_MacOS_ARM64_Release.dmg \
  dmg_temp/
```

**Alternative: Using `hdiutil` (lower-level, more control)**

```bash
# Create a sparse disk image
hdiutil create -size 2g -fs HFS+ -fsargs "-c c=64,a=16,e=16" \
  -layout SPUD -type SPARSE \
  -volname Proximap dmg_temp.dmg

# Mount it
hdiutil mount dmg_temp.dmg

# Copy files to mounted volume
cp -r dmg_temp/* /Volumes/Proximap/

# Create Applications symlink on the volume
ln -s /Applications /Volumes/Proximap/Applications

# Unmount
hdiutil unmount /Volumes/Proximap

# Convert to compressed final DMG
hdiutil convert dmg_temp.dmg -format UDZO -o Proximap_MacOS_ARM64_Release.dmg

# Clean up
rm dmg_temp.dmg
rm -rf dmg_temp
```

#### Step 3: Verify DMG

```bash
# List DMG contents
hdiutil imageinfo Proximap_MacOS_ARM64_Release.dmg

# Mount and verify (read-only)
hdiutil mount -readonly Proximap_MacOS_ARM64_Release.dmg
ls -la /Volumes/Proximap/
hdiutil unmount /Volumes/Proximap/

echo "DMG created successfully: $(ls -lh Proximap_MacOS_ARM64_Release.dmg | awk '{print $5, $9}')"
```

---

## Step 6: Verification & Testing

Before distributing, verify your package on a clean system (or in a virtual environment).

### Local Verification

#### Extract and Test ZIP

```bash
# Create a temporary directory
mkdir /tmp/proximap_test
cd /tmp/proximap_test

# Extract the ZIP
unzip /path/to/Proximap_MacOS_ARM64_Release.zip

# Verify structure
ls -la Proximap_Distribution/

# Launch the app
open Proximap_Distribution/Proximap.app
```

#### Mount and Test DMG

```bash
# Mount the DMG
hdiutil mount /path/to/Proximap_MacOS_ARM64_Release.dmg

# Navigate and launch
open /Volumes/Proximap/Proximap.app
```

### Runtime Verification Checklist

1. **App Launches**: Window appears without errors
2. **Hardware Detection**: Check that the app correctly detects:
   - Total RAM
   - Available disk space
   - GPU (if present) or CPU fallback
3. **Backend Tools Found**: Verify COLMAP and OpenMVS can be located and executed:
   - Attempt to run a small test reconstruction
   - Monitor log output for tool execution
4. **Qt Plugin Loading**: No warnings about missing Qt plugins
5. **File Permissions**: 
   ```bash
   # Verify app is executable
   ls -la /Volumes/Proximap/Proximap.app/Contents/MacOS/Proximap
   # Output should show: -rwxr-xr-x
   ```

### System Integrity Protection (SIP) Considerations

On newer macOS versions with enhanced security, unsigned apps may trigger additional checks. For development:

1. App will show a "Developer cannot be verified" dialog on first launch
2. User must go to **System Preferences > Security & Privacy** and click "Allow"
3. Subsequent launches will not show this warning

This is expected for unsigned, development builds. For commercial distribution, see `packaging_guide_commercial.md` for code signing instructions.

---

## Troubleshooting

### Common Issues

#### Issue: "command not found: colmap"

**Cause:** Binary not in `backend_bin/colmap/` or not in PATH.

**Solution:**
```bash
# Verify binary exists and is executable
ls -la /path/to/Proximap/backend_bin/colmap/colmap
file /path/to/Proximap/backend_bin/colmap/colmap

# Should show: Mach-O 64-bit executable arm64
```

If file is not executable, fix permissions:
```bash
chmod +x /path/to/Proximap/backend_bin/colmap/colmap
```

#### Issue: "dyld: Library not loaded: @rpath/..."

**Cause:** Missing dynamic library dependency.

**Solution:**
```bash
# Identify missing library
otool -L /path/to/Proximap/backend_bin/colmap/colmap

# Install the missing dependency via Homebrew
brew install [library-name]

# Download a newer pre-compiled binary that includes the dependency
```

#### Issue: "QT_PLUGIN_PATH not set" or missing Qt platform plugin

**Cause:** Qt plugins not properly bundled or COLMAP using wrong Qt.

**Solution:**

The PyInstaller spec should handle Qt6 plugin bundling. If issues persist:

```bash
# Find Qt6 plugins location
$(brew --prefix qt6)/plugins

# Manually copy to app bundle if needed
mkdir -p dist/Proximap.app/Contents/Plugins
cp -r $(brew --prefix qt6)/plugins/* dist/Proximap.app/Contents/Plugins/

# Set environment variable before running (temporary test)
export QT_PLUGIN_PATH="dist/Proximap.app/Contents/Plugins"
open dist/Proximap.app
```

#### Issue: PyInstaller build fails with "No module named 'X'"

**Cause:** Python dependency not installed or not included in spec.

**Solution:**
```bash
# Verify the dependency is installed
pip list | grep [package-name]

# If missing, install it
pip install [package-name]

# Add to Proximap.spec if not already listed in hiddenimports
# Then re-run: pyinstaller Proximap.spec
```

#### Issue: DMG creation fails with "Invalid mount point"

**Cause:** Old DMG still mounted or incorrect path.

**Solution:**
```bash
# Unmount any existing Proximap volumes
hdiutil unmount /Volumes/Proximap 2>/dev/null || true

# Try again
hdiutil create -size 2g ...
```

#### Issue: App bundle "appears damaged" on another Mac

**Cause:** Incorrect executable permissions or corrupted copy.

**Solution:**
```bash
# Fix permissions recursively
chmod -R +x /path/to/Proximap.app/Contents/MacOS/

# Re-verify on the target Mac
xattr -d com.apple.quarantine Proximap.app  # Remove quarantine attribute if present
```

### Performance Issues

#### Slow PyInstaller Build

- Use `--onedir` (default in spec) instead of `--onefile` for faster builds
- Reduce dependencies in the spec file
- Disable antivirus scanning of build directories temporarily

#### Slow App Startup

- Large bundle? Verify binary pruning was successful (Step 4)
- Check for unnecessary frameworks in `.app/Contents/Frameworks/`

#### Out of Memory During COLMAP Build from Source

- Reduce parallel build jobs: `make -j2` instead of `make -j$(nproc)`
- Increase swap space or use a Mac with more RAM

---

## Next Steps

Your packaged Proximap is ready for distribution or testing!

### For Further Development
- Refer to `pipeline_manager.py` for pipeline architecture details
- Consult `toolchain_map.json` for binary path configuration
- Check `main_window.py` for UI customization

### For Commercial Distribution
- See `packaging_guide_commercial.md` for code signing and Apple notarization
- Include proper licensing information (note: OpenMVS is AGPL v3)

### For Bug Reports or Contributions
- Test your packaged app thoroughly before publishing
- Document any platform-specific issues
- Share findings with the development team

---

## Additional Resources

- [COLMAP Documentation](https://colmap.github.io/)
- [OpenMVS GitHub](https://github.com/cdcseacave/openmvs)
- [PyInstaller Manual](https://pyinstaller.org/en/stable/)
- [Qt for Python (PySide6)](https://doc.qt.io/qtforpython/)
- [macOS App Bundle Format](https://developer.apple.com/library/archive/documentation/CoreFoundation/Conceptual/CFBundles/BundleTypes/BundleTypes.html)
- [Apple Notarization (for commercial builds)](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)

---

**Last Updated:** 2026-06-26  
**Tested On:** macOS 13+ (Ventura+) with Apple Silicon  
**Python Version:** 3.11+
