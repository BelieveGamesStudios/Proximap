# Publishing Proximap to the Ubuntu App Store (Snap Store)

The official app store for Ubuntu is the **Snap Store**. To publish **Proximap** to the Ubuntu Software Center, you package it as a `.snap` package using **Snapcraft** and upload it to Canonical's Snap Store.

---

## Step 1: Create a Developer Account & Install Tools

1. Create a free developer account at [snapcraft.io](https://snapcraft.io).
2. Install `snapcraft` on your Ubuntu machine:
   ```bash
   sudo snap install snapcraft --classic
   ```
3. Log in to your developer account from the terminal:
   ```bash
   snapcraft login
   ```

---

## Step 2: Register Your App Name

Reserve your unique application name on the Snap Store:
```bash
snapcraft register proximap
```

---

## Step 3: Add `snapcraft.yaml` Configuration

Create a `snap` folder in your project root and add `snapcraft/snapcraft.yaml`:

```bash
mkdir -p snap
```

Create `snap/snapcraft.yaml` with the following content:

```yaml
name: proximap
base: core22
version: '1.5.0'
summary: 3D Scene Reconstruction & Photogrammetry Desktop App
description: |
  Proximap is a high-performance desktop application for photogrammetry 
  and 3D point cloud reconstruction.

grade: stable
confinement: strict

apps:
  proximap:
    command: Proximap/Proximap
    desktop: usr/share/applications/proximap.desktop
    plugs:
      - home
      - desktop
      - desktop-legacy
      - x11
      - wayland
      - opengl
      - removable-media

parts:
  proximap:
    plugin: dump
    source: dist/Proximap
    organize:
      '*': Proximap/
```

---

## Step 4: Build the `.snap` Package

Run the automated Snapcraft build script:
```bash
chmod +x package_app_snap.sh
./package_app_snap.sh
```
This produces `proximap_1.5.0_amd64.snap`.

---

## Step 5: Test Locally Before Publishing

Install your `.snap` package locally to test execution:
```bash
sudo snap install --dangerous proximap_1.5.0_amd64.snap
```

Launch the app to verify GUI rendering and functionality:
```bash
proximap
```

---

## Step 6: Publish to the Ubuntu App Store

Upload the package to the Snap Store:

1. **Publish to `beta` channel first (recommended for testing)**:
   ```bash
   snapcraft upload --release=beta proximap-photogrammetry-app_1.5.0_amd64.snap
   ```

2. **Promote / Release to `stable` channel (Public Store)**:
   ```bash
   snapcraft upload --release=stable proximap-photogrammetry-app_1.5.0_amd64.snap
   ```

---

## Step 7: Manage & Monitor on Dashboard

Visit [snapcraft.io/dashboard](https://snapcraft.io/dashboard) to:
- Add high-resolution screenshots and app icons.
- Add marketing descriptions, developer website, and support links.
- View installation analytics and user feedback across Ubuntu releases.
