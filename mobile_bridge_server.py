import os
import sys
import socket
import logging
import subprocess
import zipfile
import tempfile
import werkzeug.serving
from flask import Flask, request, jsonify, send_file, render_template_string
from PySide6.QtCore import QThread, Signal

log = logging.getLogger("mobile_bridge_server")

_FIREWALL_RULE_PREFIX = "ProximapMobileBridge"
_MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".mp4", ".mov", ".m4v", ".avi", ".mkv"}
_ZIP_MAX_FILES = 500
_ZIP_MAX_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB limit


def get_local_ips():
    """
    Returns a list of local IPv4 addresses sorted by likelihood of availability
    (Wi-Fi / LAN, Hotspot, USB Tethering). Excludes loopback (127.x.x.x) and link-local (169.254.x.x).
    """
    ips = []
    
    # Method 1: Connect to an external address to find default interface IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        default_ip = s.getsockname()[0]
        s.close()
        if default_ip and not default_ip.startswith("127.") and not default_ip.startswith("169.254."):
            ips.append(default_ip)
    except Exception:
        pass

    # Method 2: Inspect active network interfaces via psutil
    try:
        import psutil
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        
        wifi_hotspot_ips = []
        other_ips = []
        
        for name, stat in stats.items():
            name_lower = name.lower()
            if name_lower in ("lo", "loopback"):
                continue
            if stat.isup and name in addrs:
                for addr in addrs[name]:
                    if addr.family == socket.AF_INET:
                        ip = addr.address
                        if ip and not ip.startswith("127.") and not ip.startswith("169.254."):
                            # Prioritize wireless / hotspot / AP / tethering interfaces
                            if any(x in name_lower for x in ["wi-fi", "wifi", "wlan", "wireless", "802.11", "ap", "hotspot", "direct", "tether"]):
                                if ip not in wifi_hotspot_ips:
                                    wifi_hotspot_ips.append(ip)
                            else:
                                if ip not in other_ips:
                                    other_ips.append(ip)
                                    
        for ip in wifi_hotspot_ips:
            if ip not in ips:
                ips.append(ip)
        for ip in other_ips:
            if ip not in ips:
                ips.append(ip)
    except Exception as e:
        log.debug(f"[Network] psutil network lookup failed: {e}")

    # Method 3: Get all hostname IPs
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127.") and not ip.startswith("169.254.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass

    if not ips:
        ips.append("127.0.0.1")

    return ips


def get_wifi_ssid() -> str | None:
    """Helper to detect the active Wi-Fi SSID or Hotspot connection name across OSes."""
    try:
        if sys.platform == "win32":
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-NetConnectionProfile).Name"],
                capture_output=True,
                text=True,
                timeout=2,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            if res.returncode == 0:
                lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
                if lines:
                    return lines[0]
        elif sys.platform.startswith("linux"):
            # 1. Try nmcli active wifi SSID
            res = subprocess.run(
                ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if line.startswith("yes:"):
                        ssid = line.split("yes:", 1)[1].strip()
                        if ssid:
                            return ssid
            # 2. Try iwgetid -r
            res = subprocess.run(
                ["iwgetid", "-r"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
            # 3. Try nmcli active connections (catches Hotspots too)
            res = subprocess.run(
                ["nmcli", "-t", "-f", "NAME,TYPE", "connection", "show", "--active"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    parts = line.split(":")
                    if len(parts) >= 2 and any(t in parts[1].lower() for t in ["wireless", "wifi", "802-11-wireless"]):
                        return parts[0].strip()
        elif sys.platform == "darwin":
            res = subprocess.run(
                ["/System/Library/PrivateFrameworks/Apple80211.framework/Resources/airport", "-I"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if " SSID:" in line:
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None



def _is_admin() -> bool:
    """Check if the current process has administrator privileges."""
    if sys.platform != 'win32':
        return True
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def _run_netsh_elevated(args: list) -> bool:
    """
    Runs a netsh command. If the current process is not admin, spawns an elevated
    instance via ShellExecuteW (triggers a UAC prompt). Returns True on success.
    """
    if sys.platform != 'win32':
        return True

    cmd_str = " ".join(args)

    if _is_admin():
        # Already admin — run directly
        try:
            result = subprocess.run(
                ["netsh"] + args,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return result.returncode == 0
        except Exception as e:
            log.warning(f"[Firewall] netsh failed: {e}")
            return False
    else:
        # Not admin — use ShellExecuteW with "runas" to get UAC elevation
        try:
            import ctypes
            ret = ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",        # verb: request elevation
                "netsh",        # executable
                cmd_str,        # parameters
                None,           # working directory
                0,              # SW_HIDE: no console window
            )
            # ShellExecuteW returns a value > 32 on success
            return int(ret) > 32
        except Exception as e:
            log.warning(f"[Firewall] ShellExecuteW failed: {e}")
            return False


def _add_firewall_rule(port: int) -> bool:
    """
    Adds a Windows Firewall inbound allow rule for the given TCP port.
    Required so that phones on the same network can reach the Flask server.
    Triggers a UAC prompt if Proximap is not running as Administrator.
    Returns True on success.
    """
    if sys.platform != 'win32':
        return True
    rule_name = f"{_FIREWALL_RULE_PREFIX}_{port}"
    ok = _run_netsh_elevated([
        "advfirewall", "firewall", "add", "rule",
        f"name={rule_name}",
        "dir=in",
        "action=allow",
        "protocol=TCP",
        f"localport={port}",
    ])
    if ok:
        log.info(f"[Firewall] Inbound allow rule added for port {port}")
    else:
        log.warning(f"[Firewall] Could not add inbound rule for port {port}. Phone connections may be blocked.")
    return ok


def _remove_firewall_rule(port: int):
    """Removes the temporary Windows Firewall rule created for this session."""
    if sys.platform != 'win32':
        return
    rule_name = f"{_FIREWALL_RULE_PREFIX}_{port}"
    _run_netsh_elevated([
        "advfirewall", "firewall", "delete", "rule",
        f"name={rule_name}",
    ])
    log.info(f"[Firewall] Inbound rule removed for port {port}")


IMPORT_PORTAL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Proximap Mobile Import</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: #0D0D0D;
            color: #E0E0E0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        .app-header {
            width: 100%;
            max-width: 480px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 0 20px 0;
            border-bottom: 1px solid #1F1F1F;
            margin-bottom: 16px;
        }
        .brand-logo {
            font-size: 18px;
            font-weight: 800;
            letter-spacing: 1px;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .brand-logo span {
            color: #00E676;
            font-size: 22px;
            line-height: 0;
        }
        .container {
            width: 100%;
            max-width: 480px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .card {
            background-color: #141414;
            border: 1px solid #222222;
            border-radius: 12px;
            padding: 20px;
        }
        .tab-header {
            display: flex;
            border-bottom: 1px solid #222222;
            margin-bottom: 20px;
        }
        .tab-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 16px 12px 4px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.8px;
            color: #00E676;
            border-bottom: 2px solid #00E676;
            text-transform: uppercase;
        }
        .tab-item svg {
            width: 16px;
            height: 16px;
            fill: currentColor;
        }
        .picker-box {
            background-color: #181818;
            border: 2px dashed #2A2A2A;
            border-radius: 12px;
            padding: 36px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }
        .picker-box:active {
            background-color: #1E1E1E;
            border-color: #00E676;
        }
        .icon-square {
            width: 52px;
            height: 52px;
            background-color: #222222;
            border: 1px solid #2C2C2C;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .icon-square svg {
            width: 26px;
            height: 26px;
            fill: #9E9E9E;
            transition: fill 0.2s ease;
        }
        .picker-box:hover .icon-square svg,
        .picker-box:active .icon-square svg {
            fill: #00E676;
        }
        .picker-title {
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.8px;
            color: #FFFFFF;
            text-transform: uppercase;
        }
        .picker-subtitle {
            font-size: 12px;
            color: #757575;
            line-height: 1.4;
            max-width: 320px;
        }
        .picker-subtitle strong {
            color: #A5A5A5;
            font-weight: 600;
        }
        #fileInput { display: none; }

        .info-card {
            background-color: #181818;
            border: 1px solid #222222;
            border-radius: 8px;
            padding: 12px 14px;
            display: flex;
            gap: 12px;
            align-items: flex-start;
            margin-top: 16px;
        }
        .info-icon {
            width: 20px;
            height: 20px;
            min-width: 20px;
            color: #00E676;
            margin-top: 1px;
        }
        .info-text {
            font-size: 12px;
            color: #9E9E9E;
            line-height: 1.45;
        }
        .info-text strong {
            color: #D0D0D0;
        }

        /* Preview Grid */
        .grid-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 16px;
            margin-bottom: 8px;
        }
        .grid-title {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.8px;
            color: #A5A5A5;
            text-transform: uppercase;
        }
        .preview-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            max-height: 320px;
            overflow-y: auto;
            background-color: #181818;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #222222;
        }
        .preview-item {
            position: relative;
            aspect-ratio: 1;
            border-radius: 8px;
            overflow: hidden;
            background-color: #222222;
            border: 1px solid #2A2A2A;
        }
        .preview-item img, .preview-item video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        .video-badge {
            position: absolute;
            bottom: 6px;
            left: 6px;
            background: rgba(0,0,0,0.8);
            color: #00E676;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 0.5px;
            padding: 2px 6px;
            border-radius: 4px;
            backdrop-filter: blur(4px);
        }
        .zip-card {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 6px;
            padding: 8px;
            background-color: #1A1A1A;
            border-radius: 6px;
            text-align: center;
        }
        .zip-card svg {
            width: 32px;
            height: 32px;
            fill: #00E676;
        }
        .zip-name {
            font-size: 10px;
            font-weight: 600;
            color: #E0E0E0;
            word-break: break-all;
            max-height: 28px;
            overflow: hidden;
        }
        .btn-remove-file {
            position: absolute;
            top: 6px;
            right: 6px;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background-color: rgba(0, 0, 0, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.25);
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.15s ease;
            z-index: 5;
            backdrop-filter: blur(4px);
        }
        .btn-remove-file:active {
            background-color: #FF5252;
            border-color: #FF5252;
            transform: scale(1.1);
        }
        .btn-remove-file svg {
            width: 12px;
            height: 12px;
            fill: currentColor;
        }

        .btn-submit {
            background-color: #00E676;
            color: #0A0A0A;
            border: none;
            border-radius: 10px;
            padding: 16px;
            font-size: 15px;
            font-weight: 700;
            letter-spacing: 0.5px;
            cursor: pointer;
            width: 100%;
            margin-top: 16px;
            display: none;
            transition: background-color 0.2s;
        }
        .btn-submit:active {
            background-color: #00C853;
        }

        .progress-box {
            display: none;
            flex-direction: column;
            gap: 12px;
            background-color: #141414;
            border: 1px solid #222222;
            padding: 24px 20px;
            border-radius: 12px;
        }
        .progress-bar-bg {
            background-color: #222222;
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
        }
        .progress-bar-fill {
            background-color: #00E676;
            height: 100%;
            width: 0%;
            transition: width 0.2s;
        }
        .status-text {
            font-size: 13px;
            color: #B0B0B0;
            text-align: center;
        }

        .success-box {
            display: none;
            text-align: center;
            padding: 36px 20px;
            background-color: #141414;
            border: 1px solid #222222;
            border-radius: 12px;
        }
        .success-icon {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background-color: rgba(0, 230, 118, 0.15);
            border: 1px solid #00E676;
            color: #00E676;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 16px auto;
        }
        .success-box h2 {
            color: #FFFFFF;
            font-size: 18px;
            margin-bottom: 8px;
        }
        .success-box p {
            color: #9E9E9E;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="app-header">
        <div class="brand-logo">PROXIMAP <span>.</span></div>
    </div>

    <div class="container">
        <div id="uploadSection" class="card">
            <div class="tab-header">
                <div class="tab-item">
                    <svg viewBox="0 0 24 24"><path d="M9 16h6v-6h4l-7-7-7 7h4v6zm-4 2h14v2H5v-2z"/></svg>
                    Upload Local Files
                </div>
            </div>

            <div class="picker-box" onclick="document.getElementById('fileInput').click()">
                <div class="icon-square">
                    <svg viewBox="0 0 24 24"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/></svg>
                </div>
                <div class="picker-title">Drag & Drop or Click to Browse</div>
                <div class="picker-subtitle">
                    Supports <strong>.jpg</strong>, <strong>.jpeg</strong>, <strong>.png</strong>, <strong>.heic</strong>, <strong>.mp4</strong>, <strong>.mov</strong>, <strong>.zip</strong> files to send directly to desktop.
                </div>
            </div>
            <input type="file" id="fileInput" multiple accept="image/*,video/*,.zip,application/zip,application/x-zip-compressed" onchange="handleFilesSelected(this.files)">

            <div id="previewContainer" style="display: none;">
                <div class="grid-header">
                    <div class="grid-title">Staged Media Preview</div>
                </div>
                <div id="previewGrid" class="preview-grid"></div>
            </div>

            <button id="uploadBtn" class="btn-submit" onclick="startUpload()">Done — Send to Proximap</button>

            <div class="info-card">
                <svg class="info-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="16" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12.01" y2="8"></line>
                </svg>
                <div class="info-text">
                    Select photos, videos, or a .zip archive from your device. Zip archives will be automatically extracted directly to your active Proximap project.
                </div>
            </div>
        </div>

        <div id="progressSection" class="progress-box">
            <div class="status-text" id="statusText">Uploading files...</div>
            <div class="progress-bar-bg">
                <div id="progressFill" class="progress-bar-fill"></div>
            </div>
        </div>

        <div id="successSection" class="success-box">
            <div class="success-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
            </div>
            <h2>Transfer Complete!</h2>
            <p>Your media files were successfully sent to Proximap on your desktop.</p>
        </div>
    </div>

    <script>
        let selectedFiles = [];

        function handleFilesSelected(files) {
            if (!files || files.length === 0) return;
            const newFiles = Array.from(files);
            for (let nf of newFiles) {
                if (!selectedFiles.some(f => f.name === nf.name && f.size === nf.size)) {
                    selectedFiles.push(nf);
                }
            }
            renderPreviewGrid();
            document.getElementById('fileInput').value = '';
        }

        function removeFile(index) {
            if (index >= 0 && index < selectedFiles.length) {
                selectedFiles.splice(index, 1);
                renderPreviewGrid();
            }
        }

        function renderPreviewGrid() {
            const container = document.getElementById('previewContainer');
            const grid = document.getElementById('previewGrid');
            const btn = document.getElementById('uploadBtn');

            grid.innerHTML = '';

            if (selectedFiles.length === 0) {
                container.style.display = 'none';
                btn.style.display = 'none';
                return;
            }

            container.style.display = 'block';
            btn.style.display = 'block';
            btn.innerText = `Done — Send ${selectedFiles.length} File${selectedFiles.length > 1 ? 's' : ''} to Proximap`;

            selectedFiles.forEach((file, index) => {
                const item = document.createElement('div');
                item.className = 'preview-item';

                // Create remove button
                const removeBtn = document.createElement('button');
                removeBtn.className = 'btn-remove-file';
                removeBtn.type = 'button';
                removeBtn.title = 'Remove item';
                removeBtn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>`;
                removeBtn.onclick = function(e) {
                    e.stopPropagation();
                    removeFile(index);
                };
                item.appendChild(removeBtn);

                if (file.name.toLowerCase().endsWith('.zip') || file.type.includes('zip')) {
                    const zipDiv = document.createElement('div');
                    zipDiv.className = 'zip-card';
                    zipDiv.innerHTML = `
                        <svg viewBox="0 0 24 24"><path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm-6 10h-2v-2h2v2zm0-4h-2v-2h2v2z"/></svg>
                        <div class="zip-name">${file.name}</div>
                    `;
                    item.appendChild(zipDiv);

                    const badge = document.createElement('span');
                    badge.className = 'video-badge';
                    badge.innerText = 'ZIP ARCHIVE';
                    item.appendChild(badge);
                } else if (file.type.startsWith('image/') || /\\.(heic|heif|jpg|jpeg|png|webp)$/i.test(file.name)) {
                    const img = document.createElement('img');
                    img.src = URL.createObjectURL(file);
                    item.appendChild(img);
                } else {
                    const video = document.createElement('video');
                    video.src = URL.createObjectURL(file);
                    item.appendChild(video);

                    const badge = document.createElement('span');
                    badge.className = 'video-badge';
                    badge.innerText = 'VIDEO';
                    item.appendChild(badge);
                }

                grid.appendChild(item);
            });
        }

        function startUpload() {
            if (selectedFiles.length === 0) return;

            document.getElementById('uploadSection').style.display = 'none';
            document.getElementById('progressSection').style.display = 'flex';

            const fill = document.getElementById('progressFill');
            const status = document.getElementById('statusText');

            const totalBytes = selectedFiles.reduce((acc, file) => acc + file.size, 0);
            let previousCompletedBytes = 0;
            let currentFileIndex = 0;

            function uploadNextFile() {
                if (currentFileIndex >= selectedFiles.length) {
                    fetch('/done', { method: 'POST' })
                        .then(() => {
                            document.getElementById('progressSection').style.display = 'none';
                            document.getElementById('successSection').style.display = 'block';
                        })
                        .catch(err => console.error('Error completing upload session:', err));
                    return;
                }

                const file = selectedFiles[currentFileIndex];
                status.innerText = `Uploading ${currentFileIndex + 1} of ${selectedFiles.length}: ${file.name}...`;

                const formData = new FormData();
                formData.append('file', file);

                const xhr = new XMLHttpRequest();
                xhr.open('POST', '/upload', true);

                xhr.upload.onprogress = function(event) {
                    if (event.lengthComputable) {
                        const currentUploaded = event.loaded;
                        const totalUploadedBytes = previousCompletedBytes + currentUploaded;
                        const displayPct = Math.min(Math.round((totalUploadedBytes / totalBytes) * 100), 99);
                        fill.style.width = displayPct + '%';
                        status.innerText = `Uploading ${currentFileIndex + 1} of ${selectedFiles.length}: ${file.name} (${displayPct}%)`;
                    }
                };

                xhr.onload = function() {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        previousCompletedBytes += file.size;
                        currentFileIndex++;
                        uploadNextFile();
                    } else {
                        console.error('Failed to upload', file.name, xhr.statusText);
                        currentFileIndex++;
                        uploadNextFile();
                    }
                };

                xhr.onerror = function() {
                    console.error('Network error uploading', file.name);
                    currentFileIndex++;
                    uploadNextFile();
                };

                xhr.send(formData);
            }

            uploadNextFile();
        }
    </script>
</body>
</html>
"""

EXPORT_PORTAL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Proximap 3D Download</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: #121212;
            color: #E0E0E0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        .card {
            background-color: #1E1E1E;
            border: 1px solid #2C2C2C;
            border-radius: 16px;
            padding: 24px;
            width: 100%;
            max-width: 400px;
            text-align: center;
        }
        .icon {
            width: 64px;
            height: 64px;
            background-color: #2A2A2A;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 16px auto;
        }
        .icon svg { width: 32px; height: 32px; fill: #00E676; }
        h1 { font-size: 20px; color: #FFFFFF; margin-bottom: 8px; }
        p { font-size: 13px; color: #9E9E9E; margin-bottom: 24px; }
        .btn-download {
            display: inline-block;
            background-color: #00E676;
            color: #121212;
            text-decoration: none;
            font-weight: bold;
            font-size: 16px;
            padding: 16px 24px;
            border-radius: 10px;
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="icon">
            <svg viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
        </div>
        <h1>Download 3D Model</h1>
        <p>{{ filename }}</p>
        <a href="/download" class="btn-download" download>Download to Phone</a>
    </div>
</body>
</html>
"""


class MobileBridgeServer(QThread):
    """
    Flask HTTP server running in a background QThread.
    Safely shuts down on server.shutdown().
    """
    file_received = Signal(str)
    all_files_received = Signal(list)
    error = Signal(str)

    def __init__(self, save_dir: str = None, mode: str = "import", serve_file: str = None, parent=None):
        super().__init__(parent)
        self.save_dir = save_dir
        self.mode = mode
        self.serve_file = serve_file
        self.received_paths = []
        self.server = None

        self.app = Flask("MobileBridge")
        self._setup_routes()

        # Bind socket HERE (in main thread) so self.port is ready before run() is called.
        # We keep the socket open to prevent port stealing between bind and make_server.
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(('0.0.0.0', 0))
        self.port = self._sock.getsockname()[1]
        self._sock.listen(5)

    def _setup_routes(self):
        @self.app.route("/")
        def index():
            if self.mode == "import":
                return render_template_string(IMPORT_PORTAL_HTML)
            else:
                filename = os.path.basename(self.serve_file) if self.serve_file else "model.glb"
                return render_template_string(EXPORT_PORTAL_HTML, filename=filename)

        @self.app.route("/upload", methods=["POST"])
        def upload():
            if "file" not in request.files or not self.save_dir:
                return jsonify({"error": "No file uploaded"}), 400
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "Empty filename"}), 400

            filename_lower = file.filename.lower()

            if filename_lower.endswith(".zip"):
                temp_fd, temp_zip_path = tempfile.mkstemp(suffix=".zip", dir=self.save_dir)
                os.close(temp_fd)
                try:
                    file.save(temp_zip_path)
                    extracted_files = []

                    with zipfile.ZipFile(temp_zip_path, 'r') as zf:
                        infolist = zf.infolist()
                        if len(infolist) > _ZIP_MAX_FILES:
                            return jsonify({"error": f"ZIP contains too many files (max {_ZIP_MAX_FILES})"}), 400

                        total_bytes = 0
                        for member in infolist:
                            if member.is_dir():
                                continue
                            
                            clean_filename = os.path.basename(member.filename)
                            if not clean_filename or clean_filename.startswith("._") or "__MACOSX" in member.filename:
                                continue

                            ext = os.path.splitext(clean_filename)[1].lower()
                            if ext not in _MEDIA_EXTENSIONS:
                                continue

                            total_bytes += member.file_size
                            if total_bytes > _ZIP_MAX_BYTES:
                                log.warning(f"[MobileBridge] ZIP uncompressed size exceeds limit of {_ZIP_MAX_BYTES} bytes")
                                break

                            # Flatten structure & auto-suffix filename collisions (e.g. photo_1.jpg)
                            base, file_ext = os.path.splitext(clean_filename)
                            dest_path = os.path.join(self.save_dir, clean_filename)
                            counter = 1
                            while os.path.exists(dest_path):
                                dest_path = os.path.join(self.save_dir, f"{base}_{counter}{file_ext}")
                                counter += 1

                            # Security check: ensure path stays within save_dir
                            abs_dest = os.path.abspath(dest_path)
                            abs_save_dir = os.path.abspath(self.save_dir)
                            if not abs_dest.startswith(abs_save_dir):
                                log.warning(f"[MobileBridge] Skipping unsafe zip member path: {member.filename}")
                                continue

                            with zf.open(member) as source, open(dest_path, "wb") as target:
                                target.write(source.read())

                            self.received_paths.append(dest_path)
                            self.file_received.emit(dest_path)
                            extracted_files.append(os.path.basename(dest_path))

                    return jsonify({
                        "status": "ok",
                        "filename": file.filename,
                        "extracted": len(extracted_files),
                        "extracted_files": extracted_files
                    })
                except Exception as e:
                    log.error(f"[MobileBridge] Error extracting zip upload {file.filename}: {e}")
                    return jsonify({"error": f"Failed to extract ZIP archive: {str(e)}"}), 500
                finally:
                    if os.path.exists(temp_zip_path):
                        try:
                            os.remove(temp_zip_path)
                        except Exception:
                            pass
            else:
                dest_path = os.path.join(self.save_dir, file.filename)
                file.save(dest_path)
                self.received_paths.append(dest_path)
                self.file_received.emit(dest_path)
                return jsonify({"status": "ok", "filename": file.filename})

        @self.app.route("/done", methods=["POST"])
        def done():
            self.all_files_received.emit(list(self.received_paths))
            return jsonify({"status": "ok"})

        @self.app.route("/download")
        def download():
            if self.serve_file and os.path.exists(self.serve_file):
                return send_file(self.serve_file, as_attachment=True)
            return "File not found", 404

    def run(self):
        try:
            # Pass the already-bound socket to make_server (avoids race condition / port stealing)
            self.server = werkzeug.serving.make_server(
                '0.0.0.0', self.port, self.app, threaded=True, fd=self._sock.fileno()
            )
            self._sock.close()  # make_server now owns the fd
            self._sock = None

            # Open Windows Firewall so that phones on the local network can reach us
            fw_ok = _add_firewall_rule(self.port)
            if not fw_ok:
                log.warning(
                    f"[Firewall] Could not add inbound rule for port {self.port}. "
                    "Mobile devices may not be able to connect. "
                    "Try running Proximap as Administrator, or manually allow TCP port "
                    f"{self.port} in Windows Defender Firewall."
                )

            log.info(f"MobileBridgeServer started on port {self.port} (mode={self.mode})")
            self.server.serve_forever()
        except Exception as e:
            log.error(f"MobileBridgeServer failed: {e}")
            self.error.emit(str(e))
        finally:
            # Always clean up the firewall rule when the server exits
            _remove_firewall_rule(self.port)

    def stop(self):
        if hasattr(self, '_sock') and self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self.server:
            try:
                self.server.shutdown()
            except Exception as e:
                log.warning(f"Error shutting down Flask server: {e}")
        self.quit()
        self.wait(2000)
        # Firewall rule is also removed in run()'s finally block, but call here as belt-and-braces
        _remove_firewall_rule(self.port)

    def get_urls(self) -> list:
        ips = get_local_ips()
        return [f"http://{ip}:{self.port}" for ip in ips]
