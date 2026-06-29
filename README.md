<img width="1918" height="1078" alt="Screenshot 2026-06-20 080015" src="https://github.com/user-attachments/assets/62a2b602-45c5-4e54-855d-76c43b8c07e0" />
# Proximap — 3D Photogrammetry Dashboard

Proximap is an intuitive desktop application built in Python and PySide6 that provides a step-by-step wizard dashboard for 3D photogrammetry reconstruction. It acts as a graphical pipeline coordinator that automates **COLMAP** (for Structure-from-Motion / camera registration) and **OpenMVS** (for dense point clouds, meshing, mesh refinement, and texturing) to turn flat images into detailed 3D scenes.

---

## Key Features

* **Drag-and-Drop Loader**: Simply drag a folder or selection of image files (`.png`, `.jpg`, `.jpeg`, `.tiff`) into the dashboard.
* **Camera Detection**: Autodetects your camera sensor make/model from image EXIF metadata.
* **Hardware Profiler**: Automatically profiles local system resources (RAM size, discrete GPU availability) and sets fallback limits for low-resource environments.
* **Reconstruction Quality Presets**: Select from Preview, Medium, High, or Ultra configuration presets to balance speed and fidelity.
* **Interactive Embedded 3D Viewer**: Interactively view reconstructed stages (Sparse Cloud & Cameras, Dense Cloud, and Textured Mesh) embedded directly inside the GUI application.
* **Multi-Format Export**: Export final textured 3D models directly as `.obj`, `.glb`, or `.ply`. GLB export and upload conversion are handled in-app without requiring a global `obj2gltf` install.

---

## Downloads

### macOS Client

Optimized for Apple Silicon (M1/M2/M3) and compatible with Intel Macs on macOS 11 or later. The macOS package includes the Proximap desktop client plus bundled COLMAP and OpenMVS reconstruction tools.

[Download Proximap for macOS](https://github.com/BelieveGamesStudios/Proximap/releases/latest/download/Proximap_Mac_Release.zip)

After downloading, unzip `Proximap_Mac_Release.zip`, open `Proximap.app`, and select a folder of overlapping photos to start a reconstruction.

---

## Repository Structure

```text
Proximap/
├── main_window.py          # PySide6 desktop GUI entry point
├── pipeline_manager.py     # Multi-stage photogrammetry pipeline coordinator
├── hardware_profiler.py    # Hardware checker (RAM and CUDA detection)
│
├── package_app.ps1         # PowerShell automated build and compilation pipeline
├── installer.nsi           # NSIS Setup Wizard installer configuration
├── toolchain_map.json      # Config mapping GUI calls to C++ binaries
│
├── LICENSE                 # GNU GPL v3 Source code license
└── THIRD_PARTY_LICENSES.md # License attributions for OpenMVS, COLMAP, etc.
```

---

## Getting Started (Developers)

### 1. Prerequisites
* **Python**: Version 3.9 or higher (64-bit recommended).
* **GPU (Optional but Recommended)**: NVIDIA GPU with CUDA drivers installed for GPU-accelerated mesh refinement and densification.

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/BelieveGamesStudios/Proximap.git
cd Proximap
pip install -r requirements.txt
```
*(Dependencies: `PySide6`, `numpy`, `pillow`, `pyinstaller` for packaging)*

### 3. Placing Backend Binaries
To run reconstructions, you must download precompiled C++ binaries for COLMAP and OpenMVS and place them in the `backend_bin` directory (which is ignored by Git to keep the repository lightweight):

1. **COLMAP**: Download the Windows release and extract it to `backend_bin/colmap/`.
2. **OpenMVS**: Download the Windows binary package and extract it to `backend_bin/openMVS/`.

Make sure your directories resemble the following structure:
```text
Proximap/
└── backend_bin/
    ├── colmap/
    │   ├── colmap.exe
    │   └── [Required DLLs...]
    └── openMVS/
        ├── DensifyPointCloud.exe
        ├── Viewer.exe
        └── [Required DLLs...]
```

### 4. Running the Dashboard
Launch the dashboard via Python:
```bash
python main_window.py
```

---

## Standalone Packaging & Distribution

You can bundle Proximap into a standalone zip file or platform installer.

### macOS

Follow the [macOS ARM64 Packaging Guide](PACKAGING_GUIDE_MACOS_ARM64.md) and run:

```bash
bash package_app.sh
```

The macOS package is written to `Proximap_Mac_Release.zip`.

### Windows

1. Place your custom logo as `app_icon.png` in the project root.
2. Run the automated PowerShell script as an Administrator:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\package_app.ps1
   ```
This script will automatically:
* Convert `app_icon.png` into a multi-resolution `.ico` icon using Pillow.
* Compile the Python scripts into a single directory layout using PyInstaller and embed the custom icon inside `Proximap.exe`.
* Strip out large developer libraries and test executables from COLMAP and OpenMVS to save ~450MB of space.
* Compile a Windows wizard installer (`Proximap_Setup.exe`) using NSIS (Nullsoft Scriptable Install System) if installed.
* Package the finalized distribution into a clean `Proximap_Commercial_Release.zip` folder.

---

## License

* The **Proximap** frontend source code is licensed under the **GNU General Public License v3 (GPL v3)**. See the [LICENSE](LICENSE) file for details.
* Native backend dependencies (OpenMVS, COLMAP, etc.) are subject to their own respective open-source licenses. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for full attribution.
