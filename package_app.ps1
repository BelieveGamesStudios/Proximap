# package_app.ps1
# Automates the packaging of Proximap for commercial distribution.

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Starting Proximap Commercial Packaging" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# 1. Cleanup old build directories and logs
Write-Host "[1/5] Cleaning up old build/dist files and logs..." -ForegroundColor Yellow
$cleanPaths = @("build", "dist", "*.log")
foreach ($path in $cleanPaths) {
    if (Test-Path $path) {
        Remove-Item -Recurse -Force $path
        Write-Host "  Removed: $path" -ForegroundColor DarkGray
    }
}

# 1.5 Convert app_icon.png to app_icon.ico if png exists
Write-Host "Checking for app_icon.png to convert to .ico..." -ForegroundColor Yellow
if (Test-Path "app_icon.png") {
    Write-Host "  Found app_icon.png. Converting to app_icon.ico using Pillow..." -ForegroundColor DarkGray
    python -c "from PIL import Image; img = Image.open('app_icon.png'); img.save('app_icon.ico', format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])"
    if (Test-Path "app_icon.ico") {
        Write-Host "  Successfully converted app_icon.png to app_icon.ico" -ForegroundColor Green
    } else {
        Write-Warning "Failed to convert app_icon.png to app_icon.ico"
    }
}

# 2. Run PyInstaller to package the Python GUI
Write-Host "[2/5] Compiling Python application using PyInstaller..." -ForegroundColor Yellow
$iconFlag = ""
if (Test-Path "app_icon.ico") {
    $iconFlag = "--icon=app_icon.ico"
}

$excludes = @(
    "--exclude-module=PySide6.QtWebEngineCore",
    "--exclude-module=PySide6.QtWebEngineWidgets",
    "--exclude-module=PySide6.QtWebEngineQuick",
    "--exclude-module=PySide6.Qt3DCore",
    "--exclude-module=PySide6.Qt3DRender",
    "--exclude-module=PySide6.QtMultimedia",
    "--exclude-module=PySide6.QtMultimediaWidgets",
    "--exclude-module=PySide6.QtCharts",
    "--exclude-module=PySide6.QtDataVisualization",
    "--exclude-module=PySide6.QtDesigner",
    "--exclude-module=PySide6.QtQml",
    "--exclude-module=PySide6.QtQuick",
    "--exclude-module=PySide6.QtQuickWidgets",
    "--exclude-module=PySide6.QtVirtualKeyboard",
    "--exclude-module=PySide6.QtSql",
    "--exclude-module=PySide6.QtXml",
    "--exclude-module=matplotlib"
)

if ($iconFlag) {
    python -m PyInstaller --onedir --noconsole $iconFlag --name Proximap --collect-all numpy --collect-all scipy --collect-all vispy --collect-all imgui_bundle --collect-all trimesh --collect-all pyrr --collect-all cv2 --collect-all pymeshlab --add-data "mesh_editor/shaders;mesh_editor/shaders" $excludes main_window.py
} else {
    python -m PyInstaller --onedir --noconsole --name Proximap --collect-all numpy --collect-all scipy --collect-all vispy --collect-all imgui_bundle --collect-all trimesh --collect-all pyrr --collect-all cv2 --collect-all pymeshlab --add-data "mesh_editor/shaders;mesh_editor/shaders" $excludes main_window.py
}


if (-not (Test-Path "dist/Proximap")) {
    Write-Error "PyInstaller compilation failed! 'dist/Proximap' not found."
    exit 1
}
Write-Host "  PyInstaller compilation complete." -ForegroundColor Green

Write-Host "  Pruning unused WebEngine DLLs and resources..." -ForegroundColor DarkGray
$webEngineFiles = @(
    "dist/Proximap/_internal/Qt6WebEngineCore.dll",
    "dist/Proximap/_internal/PySide6/Qt/bin/Qt6WebEngineCore.dll",
    "dist/Proximap/_internal/PySide6/Qt/resources/qtwebengine_resources.pak",
    "dist/Proximap/_internal/PySide6/Qt/resources/qtwebengine_devtools_resources.pak"
)
foreach ($file in $webEngineFiles) {
    if (Test-Path $file) {
        Remove-Item -Force $file
        Write-Host "  Removed unused file: $file" -ForegroundColor DarkGray
    }
}

# 3. Create distribution backend directories
Write-Host "[3/5] Setting up backend binary directories..." -ForegroundColor Yellow
$distColmapDir = "dist/Proximap/backend_bin/colmap"
$distOpenMvsDir = "dist/Proximap/backend_bin/openMVS"

New-Item -ItemType Directory -Force -Path $distColmapDir | Out-Null
New-Item -ItemType Directory -Force -Path $distOpenMvsDir | Out-Null

# 4. Copy backend binaries selectively (Pruning unneeded test/debug files)
Write-Host "[4/5] Selectively copying backend toolchain dependencies..." -ForegroundColor Yellow

# Copy COLMAP dependencies
Write-Host "  Copying COLMAP binaries and vocabulary trees..." -ForegroundColor DarkGray
# - DLLs
Copy-Item -Path "backend_bin/colmap/*.dll" -Destination $distColmapDir
# - Executable
Copy-Item -Path "backend_bin/colmap/colmap.exe" -Destination $distColmapDir
# - Vocab trees (.bin)
Get-ChildItem -Path "backend_bin/colmap/*.bin" -ErrorAction SilentlyContinue | Copy-Item -Destination $distColmapDir
# - Plugins folder (Qt plugins)
if (Test-Path "backend_bin/colmap/plugins") {
    Copy-Item -Path "backend_bin/colmap/plugins" -Destination $distColmapDir -Recurse
}

# Copy OpenMVS dependencies
Write-Host "  Copying OpenMVS binaries (excluding .lib, .log, and test executables)..." -ForegroundColor DarkGray
# - DLLs
Copy-Item -Path "backend_bin/openMVS/*.dll" -Destination $distOpenMvsDir
# - Needed executables
$mvsExes = @(
    "DensifyPointCloud.exe",
    "InterfaceCOLMAP.exe",
    "ReconstructMesh.exe",
    "RefineMesh.exe",
    "TextureMesh.exe"
)
foreach ($exe in $mvsExes) {
    if (Test-Path "backend_bin/openMVS/$exe") {
        Copy-Item -Path "backend_bin/openMVS/$exe" -Destination $distOpenMvsDir
    } else {
        Write-Warning "OpenMVS executable not found: $exe"
    }
}

# Copy toolchain mapping configuration
Write-Host "  Copying toolchain map configuration..." -ForegroundColor DarkGray
Copy-Item -Path "toolchain_map.json" -Destination "dist/Proximap/"

# Copy application icon for runtime usage
if (Test-Path "app_icon.ico") {
    Write-Host "  Copying app_icon.ico to dist directory..." -ForegroundColor DarkGray
    Copy-Item -Path "app_icon.ico" -Destination "dist/Proximap/"
}

# Copy public directory containing toolbar icons
Write-Host "  Copying UI icons and public assets..." -ForegroundColor DarkGray
if (Test-Path "public") {
    Copy-Item -Path "public" -Destination "dist/Proximap/public" -Recurse
} else {
    Write-Warning "Public directory not found. UI icons will fail to load."
}

# 5. Compress the finalized distribution folder to ZIP
Write-Host "[5/5] Compressing distribution into release ZIP..." -ForegroundColor Yellow
$zipFile = "Proximap_Commercial_Release.zip"
if (Test-Path $zipFile) {
    Remove-Item -Force $zipFile
}
Compress-Archive -Path "dist/Proximap" -DestinationPath $zipFile

# 6. Build NSIS Installer if makensis is available
Write-Host "[6/6] Checking for NSIS (makensis.exe) compiler..." -ForegroundColor Yellow
$makensis = Get-Command makensis -ErrorAction SilentlyContinue
if (-not $makensis) {
    # Check common install paths
    $commonPaths = @(
        "$env:ProgramFiles\NSIS\makensis.exe",
        "${env:ProgramFiles(x86)}\NSIS\makensis.exe"
    )
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            $makensis = $path
            break
        }
    }
}

if ($makensis) {
    if (-not (Test-Path "app_icon.ico")) {
        Write-Warning "Skipping automated NSIS installer build because 'app_icon.ico' was not found at the project root."
        Write-Warning "Please place a valid custom 'app_icon.ico' in the root or edit 'installer.nsi' to remove/update icon definitions, then run: makensis installer.nsi"
    } else {
        Write-Host "  Found NSIS compiler: $makensis" -ForegroundColor Green
        Write-Host "  Compiling installer setup executable..." -ForegroundColor DarkGray
        & $makensis "installer.nsi"
        if (Test-Path "Proximap_Setup.exe") {
            Write-Host "  NSIS Setup executable generated: Proximap_Setup.exe" -ForegroundColor Green
        } else {
            Write-Warning "NSIS compiler finished but Proximap_Setup.exe was not found."
        }
    }
} else {
    Write-Host "  NSIS compiler (makensis.exe) not found. Skipping installer setup build." -ForegroundColor DarkGray
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Proximap successfully packaged!" -ForegroundColor Green
Write-Host " Release package: $zipFile" -ForegroundColor Green
if (Test-Path "Proximap_Setup.exe") {
    Write-Host " Installer setup: Proximap_Setup.exe" -ForegroundColor Green
}
Write-Host "=============================================" -ForegroundColor Cyan
