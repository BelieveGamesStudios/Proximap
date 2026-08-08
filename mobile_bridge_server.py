import os
import sys
import socket
import logging
import subprocess
import werkzeug.serving
from flask import Flask, request, jsonify, send_file, render_template_string
from PySide6.QtCore import QThread, Signal

log = logging.getLogger("mobile_bridge_server")

_FIREWALL_RULE_PREFIX = "ProximapMobileBridge"


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
            background-color: #121212;
            color: #E0E0E0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            width: 100%;
            max-width: 480px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .header {
            text-align: center;
            padding: 16px 0;
            border-bottom: 1px solid #292929;
        }
        .header h1 {
            font-size: 20px;
            color: #00E676;
            margin-bottom: 4px;
        }
        .header p {
            font-size: 13px;
            color: #9E9E9E;
        }
        .picker-box {
            background-color: #1E1E1E;
            border: 2px dashed #00E676;
            border-radius: 12px;
            padding: 30px 20px;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .picker-box:active { background-color: #292929; }
        .picker-box svg {
            width: 48px;
            height: 48px;
            fill: #00E676;
            margin-bottom: 12px;
        }
        .picker-box span {
            display: block;
            font-size: 15px;
            font-weight: 600;
            color: #FFFFFF;
        }
        #fileInput { display: none; }
        .preview-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            max-height: 280px;
            overflow-y: auto;
            background-color: #1E1E1E;
            padding: 8px;
            border-radius: 8px;
        }
        .preview-item {
            position: relative;
            aspect-ratio: 1;
            border-radius: 6px;
            overflow: hidden;
            background-color: #2A2A2A;
        }
        .preview-item img, .preview-item video {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .video-badge {
            position: absolute;
            bottom: 4px;
            left: 4px;
            background: rgba(0,0,0,0.7);
            color: #00E676;
            font-size: 10px;
            padding: 2px 4px;
            border-radius: 3px;
        }
        .btn-submit {
            background-color: #00E676;
            color: #121212;
            border: none;
            border-radius: 8px;
            padding: 16px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            display: none;
        }
        .btn-submit:disabled {
            background-color: #444444;
            color: #888888;
        }
        .progress-box {
            display: none;
            flex-direction: column;
            gap: 8px;
            background-color: #1E1E1E;
            padding: 16px;
            border-radius: 8px;
        }
        .progress-bar-bg {
            background-color: #2A2A2A;
            height: 12px;
            border-radius: 6px;
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
            padding: 30px;
            background-color: #1E1E1E;
            border-radius: 12px;
        }
        .success-box h2 { color: #00E676; margin-bottom: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Proximap Mobile Import</h1>
            <p>Send photos & videos directly to your PC</p>
        </div>

        <div id="uploadSection">
            <div class="picker-box" onclick="document.getElementById('fileInput').click()">
                <svg viewBox="0 0 24 24"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/></svg>
                <span>Select Images / Videos</span>
            </div>
            <input type="file" id="fileInput" multiple accept="image/heic,image/heif,image/jpeg,image/jpg,image/png,image/webp,image/*,video/quicktime,video/mp4,video/x-m4v,video/*" onchange="handleFilesSelected(this.files)">
            <br>
            <div id="previewGrid" class="preview-grid" style="display: none;"></div>
            <br>
            <button id="uploadBtn" class="btn-submit" onclick="startUpload()">Done — Send to Proximap</button>
        </div>

        <div id="progressSection" class="progress-box">
            <div class="status-text" id="statusText">Uploading files...</div>
            <div class="progress-bar-bg">
                <div id="progressFill" class="progress-bar-fill"></div>
            </div>
        </div>

        <div id="successSection" class="success-box">
            <h2>✓ Transfer Complete!</h2>
            <p>Your media files were sent to Proximap on your desktop.</p>
        </div>
    </div>

    <script>
        let selectedFiles = [];

        function handleFilesSelected(files) {
            if (!files || files.length === 0) return;
            selectedFiles = Array.from(files);

            const grid = document.getElementById('previewGrid');
            grid.innerHTML = '';
            grid.style.display = 'grid';

            const btn = document.getElementById('uploadBtn');
            btn.style.display = 'block';
            btn.innerText = `Done — Send ${selectedFiles.length} File(s) to Proximap`;

            // Render thumbnails asynchronously in small batches to keep mobile UI fluid
            let index = 0;
            function renderBatch() {
                const batchSize = 6;
                const end = Math.min(index + batchSize, selectedFiles.length);
                for (let i = index; i < end; i++) {
                    const file = selectedFiles[i];
                    const item = document.createElement('div');
                    item.className = 'preview-item';
                    
                    if (file.type.startsWith('image/') || /\\.(heic|heif|jpg|jpeg|png|webp)$/i.test(file.name)) {
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
                }
                index = end;
                if (index < selectedFiles.length) {
                    requestAnimationFrame(renderBatch);
                }
            }
            renderBatch();
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
                    // Signal finish
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

                // Track upload progress for the current file
                xhr.upload.onprogress = function(event) {
                    if (event.lengthComputable) {
                        const currentUploaded = event.loaded;
                        const totalUploadedBytes = previousCompletedBytes + currentUploaded;
                        
                        // Limit display percentage to 99% until /done finishes
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
