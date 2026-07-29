# Packaging Guide for Debian-based Linux (Ubuntu, Debian, Linux Mint, Pop!_OS)

This guide explains how to bundle **Proximap** into a native `.deb` installer package and a portable `.zip` release for Debian-based Linux operating systems.

---

## 1. Prerequisites

### System Tools
Make sure the standard Debian build tools, Python 3, and `zip` are installed:

```bash
sudo apt update
sudo apt install python3 python3-pip dpkg-dev zip
```

### Python Dependencies & PyInstaller
Install PyInstaller and the Python requirements:

```bash
python3 -m pip install --break-system-packages pyinstaller -r requirements.txt
```

*(Or install required libraries directly: `PySide6`, `vispy`, `numpy`, `pillow`, `opencv-python`, `trimesh`, `pyrr`, `PyOpenGL`)*

---

## 2. Backend Binaries (`backend_bin`)

For full 3D reconstruction functionality, ensure precompiled Linux binaries for **COLMAP** and **OpenMVS** are placed inside `backend_bin/`:

```text
Proximap/
└── backend_bin/
    ├── colmap/
    │   └── colmap (executable & libraries)
    └── openMVS/
        ├── DensifyPointCloud
        ├── ReconstructMesh
        ├── RefineMesh
        ├── TextureMesh
        └── Viewer
```

---

## 3. Automated Packaging

To generate the `.deb` installer and portable release zip, simply run:

```bash
./package_app_deb.sh
```

### Generated Release Files:
1. **`Proximap_1.0.0_amd64.deb`**: Native Debian package that installs Proximap into `/opt/proximap/` and registers a system desktop shortcut in the Application Menu.
2. **`Proximap_Linux_Release.zip`**: Portable standalone ZIP containing the binary directory.

---

## 4. Installing the Generated `.deb` Package

To install Proximap on your machine:

```bash
sudo dpkg -i Proximap_1.0.0_amd64.deb
sudo apt-get install -f  # Resolves any missing system dependencies if needed
```

Once installed, launch Proximap from your desktop application launcher or run:

```bash
proximap
```
