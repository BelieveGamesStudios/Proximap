# Packaging Proximap for Commercial Distribution

This guide details how to compile and package Proximap into a standalone, dependency-free ZIP archive for commercial distribution. 

Commercial distribution requires special considerations:
* **Selective Bundling**: Native C++ binaries (COLMAP and OpenMVS) contain hundreds of megabytes of testing utilities and compile-time static link libraries (`.lib`) that are completely unnecessary at runtime. These must be excluded to keep the release package lightweight.
* **Licensing**: Open-source toolchains (COLMAP, OpenMVS, Ceres Solver, Qt/PySide6, SQLite, and OpenEXR) have specific license notices that should be distributed along with the app.
* **Security & Reputation**: Using `--onedir` instead of `--onefile` dramatically reduces start-up latency and prevents false-positive warnings from antivirus engines.

---

## Workspace Directory Structure

The final packaged application layout inside the ZIP file must look like this:

```text
Proximap/
│
├── Proximap.exe             <- Main application executable
├── _internal/               <- Python runtime files, PySide6, NumPy, etc.
├── toolchain_map.json       <- Config mapping frontend calls to backend binaries
│
└── backend_bin/             <- Native C++ toolchain binaries
    ├── colmap/
    │   ├── colmap.exe       <- Main SfM tool
    │   ├── plugins/         <- Qt6 platform & image format plugins (Required!)
    │   └── [Required DLLs]  <- OpenBLAS, CUDA, Ceres, Qt, etc. (No test files!)
    │
    └── openMVS/
        ├── DensifyPointCloud.exe
        ├── InterfaceCOLMAP.exe
        ├── ReconstructMesh.exe
        ├── RefineMesh.exe
        ├── TextureMesh.exe
        ├── Viewer.exe
        └── [Required DLLs]  <- OpenCV, OpenEXR, WebP, etc. (No .lib or test files!)
```

---

## Method 1: Automated Packaging (Recommended)

An automation script has been provided at the project root to handle the cleaning, compiling, backend binary pruning, configuration copying, and ZIP creation in one command.

### Steps:
1. Open **PowerShell** as an administrator.
2. Navigate to your project root:
   ```powershell
   cd "C:\Users\AFOLABI FUNMILADE\Proximap"
   ```
3. Run the automated packaging script:
   ```powershell
   .\package_app.ps1
   ```
   *Note: If you receive an execution policy error, run the script bypassing restrictions:*
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\package_app.ps1
   ```

4. Once completed, a file named `Proximap_Commercial_Release.zip` will be generated in your project root, optimized and ready for distribution.

---

## Method 2: Manual Packaging Step-by-Step

If you need to customize the build, follow these manual commands:

### Step 1: Purge Caches and Temporary Test Files
Clean up temporary files, logs, and compiled cache directories to prevent packaging stale data:
```powershell
Remove-Item -Recurse -Force dist, build, *.log -ErrorAction SilentlyContinue
```

### Step 2: Compile the Python Application
Use PyInstaller to compile the entrypoint `main_window.py`. We use `--onedir` for directory output (improving startup speed and avoiding antivirus issues) and `--noconsole` to hide the terminal window behind the GUI:
```powershell
python -m PyInstaller --onedir --noconsole --name Proximap --collect-all numpy main_window.py
```

### Step 3: Set up Backend Binaries (Pruned Layout)
1. Create the backend binary directories in the new output folder:
   ```powershell
   New-Item -ItemType Directory -Force -Path "dist/Proximap/backend_bin/colmap"
   New-Item -ItemType Directory -Force -Path "dist/Proximap/backend_bin/openMVS"
   ```

2. Copy the **COLMAP** runtime files. Copy only the executables, DLLs, and Qt plugins, skipping the `*_test.exe` files (which save ~300MB of space):
   ```powershell
   Copy-Item -Path "backend_bin/colmap/*.dll" -Destination "dist/Proximap/backend_bin/colmap/"
   Copy-Item -Path "backend_bin/colmap/colmap.exe" -Destination "dist/Proximap/backend_bin/colmap/"
   Copy-Item -Path "backend_bin/colmap/plugins" -Destination "dist/Proximap/backend_bin/colmap/" -Recurse
   ```

3. Copy the **OpenMVS** runtime files. Exclude static libraries (`.lib`) and development files, copying only the required binaries (saving ~150MB of space):
   ```powershell
   # Copy DLLs
   Copy-Item -Path "backend_bin/openMVS/*.dll" -Destination "dist/Proximap/backend_bin/openMVS/"
   
   # Copy runtime executables
   $exes = @("DensifyPointCloud.exe", "InterfaceCOLMAP.exe", "ReconstructMesh.exe", "RefineMesh.exe", "TextureMesh.exe", "Viewer.exe")
   foreach ($exe in $exes) {
       Copy-Item -Path "backend_bin/openMVS/$exe" -Destination "dist/Proximap/backend_bin/openMVS/"
   }
   ```

4. Copy the toolchain configuration mapping file:
   ```powershell
   Copy-Item -Path "toolchain_map.json" -Destination "dist/Proximap/"
   ```

### Step 4: Add License & Attributions (Crucial for Commercial Releases)
Create a `LICENSE.txt` or `ATTRIBUTIONS.txt` in the root of `dist/Proximap/`. It should cite the licenses of the used toolchains:
* **Proximap**: Your proprietary license.
* **COLMAP**: New BSD License (3-clause BSD).
* **OpenMVS**: Affero GPL (AGPL v3) or proprietary license (if you purchased a commercial license from OpenMVS developers).
* **PySide6 (Qt)**: LGPL v3.
* **NumPy**: BSD License.

### Step 5: Archive the Cleaned Distribution Folder
Compress the `dist/Proximap` folder to ZIP format:
```powershell
Compress-Archive -Path "dist/Proximap" -DestinationPath "Proximap_Commercial_Release.zip"
```

---

## Method 3: Creating a Windows Setup Installer (NSIS)

If you want a professional distribution experience, you can package the compiled application directory into a single Windows Setup executable (`Proximap_Setup.exe`) using **NSIS (Nullsoft Scriptable Install System)**. 

An installer setup application allows users to install the app locally, add desktop and start menu shortcuts, register an uninstaller in Windows Control Panel, and configure custom icons.

### 1. Prerequisites
- Download and install **NSIS** from [nsis.sourceforge.io](https://nsis.sourceforge.io/).
- Ensure the `makensis` compiler executable is added to your Windows environment variables (PATH), or run it directly from its install directory.

### 2. Custom Icon Configuration
To use a custom icon for your installer:
1. Prepare a high-resolution `.ico` icon file (e.g. `app_icon.ico`) and place it in your project root.
2. If compiling the Python application via PyInstaller, you can also embed the custom icon inside `Proximap.exe` itself:
   ```powershell
   python -m PyInstaller --onedir --noconsole --icon=app_icon.ico --name Proximap --collect-all numpy main_window.py
   ```

### 3. NSIS Script Structure (`installer.nsi`)
An example configuration file named [installer.nsi](file:///c:/Users/AFOLABI%20FUNMILADE/Proximap/installer.nsi) has been created at your project root.

Key parts of the script to configure:
* **Installer Icon**: Set the custom icon file path:
  ```nsis
  !define MUI_ICON "app_icon.ico"
  !define MUI_UNICON "app_icon.ico"
  ```
* **Product Details**: Modify publisher, version, and folder names:
  ```nsis
  !define PRODUCT_NAME "Proximap"
  !define PRODUCT_PUBLISHER "Believe Games Studios"
  ```

### 4. Compile the Installer Setup Executable
To compile your installer:
1. Open PowerShell at your project root.
2. Run the NSIS compiler command:
   ```powershell
   makensis installer.nsi
   ```
   *(If `makensis` is not in your environment PATH, execute it by specifying the absolute path, e.g. `& "C:\Program Files (x86)\NSIS\makensis.exe" installer.nsi`)*
3. When compilation finishes, a standalone setup installer named **`Proximap_Setup.exe`** will be generated in your project root. Users can run this file to launch the install wizard, select directory, install files, and automatically add desktop/start menu shortcuts.

---

## Commercial Distribution Best Practices

1. **Code Signing**:
   Windows SmartScreen will flag the generated `Proximap.exe` as "Unknown Publisher" when users first download it. For a commercial release, buy a Code Signing Certificate (e.g., Sectigo or DigiCert) and sign the binary using `signtool.exe`:
   ```powershell
   signtool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 /a dist/Proximap/Proximap.exe
   ```
2. **Antivirus Whitelisting**:
   PyInstaller packages sometimes trigger false positives from heuristics. Using the `--onedir` output mode (which extracts files upfront) instead of `--onefile` (which unpacks to a temp folder dynamically) significantly minimizes antivirus flagging.
3. **Dedicated GPU Requirements**:
   State clearly in your commercial system requirements that for GPU acceleration (Cuda-enabled mapping and mesh refinement), the end-user must have a CUDA-enabled NVIDIA GPU with latest drivers installed. Otherwise, the app falls back to CPU simulation or slower CPU-bound processing.
