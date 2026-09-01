"""
mesh_cleanup.py — Tiered 3D Mesh Processing Engine for Proximap

This module provides a unified abstraction (MeshProcessingBackend) for all 3D mesh repair,
decimation, distance-based merging, and Taubin smoothing operations across Proximap.

Architecture (v3 — Tiered MeshProcessingBackend Abstraction):
  1. PyMeshLabDirectBackend  — In-process PyMeshLab (Python 3.10 / ABI matched).
  2. PyMeshLabWorkerBackend  — Subprocess sidecar (pymeshlab_worker.py + Python 3.10).
  3. Open3DBackend           — In-process Open3D fallback.
  4. TrimeshBackend          — In-process Trimesh fallback (most limited).

Capabilities Reporting:
  The active backend exposes a `capabilities` property returning a set of supported
  BackendCapability flags. The UI layer uses `get_backend_capabilities()` to query
  available features (e.g. greying out unsupported actions or showing degraded fallback notices).
"""

import os
import sys
import json
import math
import warnings
import subprocess
import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Set, Optional, Callable

logger = logging.getLogger(__name__)


def _log(msg: str, callback=None):
    if callback:
        try:
            callback(msg)
        except Exception:
            pass
    logger.info(msg)


# ---------------------------------------------------------------------------
# Capability Flags & Exceptions
# ---------------------------------------------------------------------------

class BackendCapability(Enum):
    """Flags indicating operations supported by a MeshProcessingBackend."""
    MERGE_BY_DISTANCE  = auto()
    TAUBIN_SMOOTH      = auto()
    HOLE_CLOSING       = auto()
    NONMANIFOLD_REPAIR = auto()
    DECIMATION         = auto()


class NotSupportedError(Exception):
    """Raised when a requested mesh operation is not supported by the active backend."""
    pass


class DegradedFallbackWarning(UserWarning):
    """Issued when a backend executes an operation using a degraded fallback."""
    pass


# ---------------------------------------------------------------------------
# Path Resolution Helpers (for Subprocess Worker & Wheel Injection)
# ---------------------------------------------------------------------------

def _app_base_dir() -> str:
    """Return the app's effective root directory (handles PyInstaller _internal)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _internal_dir() -> str:
    """Return the _internal directory (PyInstaller) or the source dir."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def _find_python310() -> str:
    """Locate a Python 3.10 or 3.11 interpreter that can load the bundled cp310/cp311 extension."""
    base   = _app_base_dir()
    intdir = _internal_dir()

    # --- Priority 1: user-level pymeshlab venv (created by ensure_pymeshlab_venv) ---
    # This is a proper pip-installed environment — most reliable.
    venv_python = _get_venv_python()
    if venv_python and os.path.isfile(venv_python) and os.access(venv_python, os.X_OK):
        return venv_python

    # --- Priority 2: Bundled standalone Python 3.10 (from python-build-standalone) ---
    for candidate in [
        os.path.join(base,   "backend_bin", "python3.10", "bin", "python3.10"),
        os.path.join(base,   "backend_bin", "python3.11", "bin", "python3.11"),
        os.path.join(base,   "backend_bin", "python3.10", "bin", "python"),
        os.path.join(base,   "backend_bin", "python3.11", "bin", "python"),
        os.path.join(base,   "backend_bin", "venv", "bin", "python"),
        os.path.join(intdir, "backend_bin", "python3.10", "bin", "python3.10"),
        os.path.join(intdir, "backend_bin", "python3.11", "bin", "python3.11"),
        os.path.join(intdir, "python3.10",  "bin", "python3.10"),
        os.path.join(intdir, "python3.11",  "bin", "python3.11"),
        os.path.join(base,   "backend_bin", "python3.10", "python.exe"),
        os.path.join(base,   "backend_bin", "python3.11", "python.exe"),
        os.path.join(intdir, "python3.10",  "python.exe"),
        os.path.join(intdir, "python3.11",  "python.exe"),
    ]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    import shutil
    for name in ["python3.10", "python3.11", "py310", "py311"]:
        sys_py = shutil.which(name)
        if sys_py:
            return sys_py

    if sys.version_info[:2] in [(3, 10), (3, 11)]:
        return sys.executable

    return ""


def _get_venv_dir() -> str:
    """Return the path to the user-level pymeshlab venv."""
    # Store in XDG_DATA_HOME or ~/.local/share/proximap/pymeshlab_venv
    xdg = os.environ.get("XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share"))
    return os.path.join(xdg, "proximap", "pymeshlab_venv")


def _get_venv_python() -> str:
    """Return the python executable inside the pymeshlab venv (may not exist yet)."""
    venv_dir = _get_venv_dir()
    # Linux/macOS
    p = os.path.join(venv_dir, "bin", "python3")
    if os.path.isfile(p):
        return p
    p = os.path.join(venv_dir, "bin", "python")
    if os.path.isfile(p):
        return p
    # Windows
    p = os.path.join(venv_dir, "Scripts", "python.exe")
    if os.path.isfile(p):
        return p
    return ""


def _find_bundled_whl() -> str:
    """Return the path to the platform-appropriate bundled PyMeshLab .whl file."""
    base   = _app_base_dir()
    intdir = _internal_dir()

    import platform, glob
    machine = platform.machine().lower()  # x86_64, aarch64, arm64
    system  = sys.platform               # linux, darwin, win32

    # Map platform to wheel tag substrings
    if system == "linux":
        tags = ["manylinux", "linux"]
    elif system == "darwin":
        tags = ["macosx", "darwin"]
    else:
        tags = ["win"]

    search_dirs = [
        os.path.join(base,   "backend_bin", "PymeshLab"),
        os.path.join(intdir, "backend_bin", "PymeshLab"),
        os.path.join(intdir, "PymeshLab"),
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for tag in tags:
            for whl in glob.glob(os.path.join(d, f"pymeshlab-*{tag}*.whl")):
                return whl
    return ""


def ensure_pymeshlab_venv(log_callback=None) -> bool:
    """
    Create a user-level Python 3.10 venv and pip-install the bundled PyMeshLab .whl
    into it (if not already done). This is the canonical way to get a working
    pymeshlab that has correct Qt plugin paths, .dist-info metadata, and shared libs.

    Returns True if the venv is ready and pymeshlab can be imported, False otherwise.
    Called once at application startup (during the splash screen).
    """
    venv_dir    = _get_venv_dir()
    venv_python = _get_venv_python()
    sentinel    = os.path.join(venv_dir, ".pymeshlab_installed")

    # If sentinel exists, venv is already set up — do a quick sanity check.
    if os.path.isfile(sentinel) and venv_python:
        result = subprocess.run(
            [venv_python, "-c", "import pymeshlab; print('ok')"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and "ok" in result.stdout:
            _log("[PYMESHLAB] Venv already set up and verified.", log_callback)
            return True
        else:
            # Sentinel exists but import failed — re-create venv
            _log("[PYMESHLAB] Venv check failed, recreating...", log_callback)
            import shutil as _shutil
            _shutil.rmtree(venv_dir, ignore_errors=True)
            try:
                os.remove(sentinel)
            except FileNotFoundError:
                pass

    # Find bundled standalone Python 3.10 to create the venv
    base   = _app_base_dir()
    intdir = _internal_dir()
    bundled_py310 = ""
    for candidate in [
        os.path.join(base,   "backend_bin", "python3.10", "bin", "python3.10"),
        os.path.join(intdir, "backend_bin", "python3.10", "bin", "python3.10"),
        os.path.join(intdir, "python3.10",  "bin", "python3.10"),
        os.path.join(base,   "backend_bin", "python3.10", "python.exe"),
    ]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            bundled_py310 = candidate
            break

    if not bundled_py310:
        import shutil
        bundled_py310 = shutil.which("python3.10") or ""

    if not bundled_py310:
        _log("[PYMESHLAB] No Python 3.10 runtime found — cannot create pymeshlab venv.", log_callback)
        return False

    whl = _find_bundled_whl()
    if not whl:
        _log("[PYMESHLAB] No bundled .whl file found — cannot install pymeshlab.", log_callback)
        return False

    _log(f"[PYMESHLAB] Setting up mesh tools (first launch only)...", log_callback)
    _log(f"[PYMESHLAB] Python: {bundled_py310}", log_callback)
    _log(f"[PYMESHLAB] Wheel:  {os.path.basename(whl)}", log_callback)

    os.makedirs(venv_dir, exist_ok=True)

    # Step 1: Create the venv
    try:
        result = subprocess.run(
            [bundled_py310, "-m", "venv", "--without-pip", venv_dir],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            _log(f"[PYMESHLAB] venv creation failed: {result.stderr[:500]}", log_callback)
            return False
        _log("[PYMESHLAB] Venv created.", log_callback)
    except Exception as e:
        _log(f"[PYMESHLAB] venv creation exception: {e}", log_callback)
        return False

    # Step 2: Bootstrap pip into the venv using ensurepip via the venv's own python
    new_python = _get_venv_python()
    if not new_python:
        _log("[PYMESHLAB] Venv python not found after creation.", log_callback)
        return False

    try:
        result = subprocess.run(
            [new_python, "-m", "ensurepip", "--upgrade"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            # ensurepip might fail if pip is missing from the standalone; try get-pip
            _log("[PYMESHLAB] ensurepip failed, trying get-pip fallback...", log_callback)
            import urllib.request
            get_pip_path = os.path.join(venv_dir, "get-pip.py")
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip_path)
            subprocess.run([new_python, get_pip_path], capture_output=True, text=True, timeout=120)
        _log("[PYMESHLAB] pip bootstrapped.", log_callback)
    except Exception as e:
        _log(f"[PYMESHLAB] ensurepip failed (non-fatal): {e}", log_callback)

    # Step 3: pip install numpy first (pymeshlab depends on it), then pymeshlab from the .whl
    try:
        # Install numpy from PyPI (required dependency for pymeshlab)
        _log("[PYMESHLAB] Installing numpy dependency...", log_callback)
        result = subprocess.run(
            [new_python, "-m", "pip", "install", "--quiet", "numpy"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            _log(f"[PYMESHLAB] numpy install failed: {result.stderr[-300:]}", log_callback)
            # Non-fatal: pymeshlab may still work if numpy is already available

        # Now install pymeshlab itself from the bundled .whl
        _log("[PYMESHLAB] Installing pymeshlab from bundled wheel...", log_callback)
        result = subprocess.run(
            [new_python, "-m", "pip", "install", "--no-deps", "--force-reinstall", whl],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            _log(f"[PYMESHLAB] pip install failed:\n{result.stdout[-500:]}\n{result.stderr[-500:]}", log_callback)
            return False
        _log("[PYMESHLAB] pymeshlab installed into venv.", log_callback)
    except Exception as e:
        _log(f"[PYMESHLAB] pip install exception: {e}", log_callback)
        return False

    # Step 4: Verify the install
    try:
        result = subprocess.run(
            [new_python, "-c", "import pymeshlab; print('ok')"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and "ok" in result.stdout:
            # Write sentinel so we skip setup on future launches
            with open(sentinel, "w") as f:
                f.write("1")
            _log("[PYMESHLAB] Venv verified — pymeshlab ready.", log_callback)
            return True
        else:
            _log(f"[PYMESHLAB] Verification failed: {result.stderr[:500]}", log_callback)
            return False
    except Exception as e:
        _log(f"[PYMESHLAB] Verification exception: {e}", log_callback)
        return False


def _find_worker_script() -> str:
    """Locate pymeshlab_worker.py relative to app root / _internal."""
    base   = _app_base_dir()
    intdir = _internal_dir()

    for candidate in [
        os.path.join(base,   "pymeshlab_worker.py"),
        os.path.join(intdir, "pymeshlab_worker.py"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return ""


def _try_direct_import_pymeshlab():
    """Attempt direct in-process import of pymeshlab."""
    try:
        import pymeshlab
        return pymeshlab
    except ImportError:
        pass

    intdir = _internal_dir()
    base   = _app_base_dir()
    for ml_dir in [
        os.path.join(base,   "backend_bin", "pymeshlab_extracted"),
        os.path.join(intdir, "pymeshlab_extracted"),
        os.path.join(intdir, "backend_bin", "pymeshlab_extracted"),
    ]:
        if os.path.isdir(ml_dir) and ml_dir not in sys.path:
            sys.path.insert(0, ml_dir)
            lib_dir = os.path.join(ml_dir, "pymeshlab", "lib")
            if os.path.isdir(lib_dir):
                old_ld = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + old_ld if old_ld else "")
            try:
                import pymeshlab
                return pymeshlab
            except ImportError:
                sys.path.pop(0)

    return None


# ---------------------------------------------------------------------------
# Abstract Base Class: MeshProcessingBackend
# ---------------------------------------------------------------------------

class MeshProcessingBackend(ABC):
    """Abstract base class for tiered 3D mesh processing engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable identifier name for this backend."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> Set[BackendCapability]:
        """Return set of capabilities supported by this backend."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can execute in the current runtime environment."""
        pass

    @abstractmethod
    def cleanup(self, input_ply_path: str, output_ply_path: str,
                log_callback: Optional[Callable[[str], None]] = None,
                cleanup_params: Optional[dict] = None) -> bool:
        """Execute mesh repair, decimation, and cleanup."""
        pass

    @abstractmethod
    def merge_by_distance(self, input_ply_path: str, output_ply_path: str,
                          threshold_pct: float = 1.0, bbox_diagonal: float = 0.0,
                          log_callback: Optional[Callable[[str], None]] = None) -> bool:
        """Merge close vertices within threshold_pct of bbox diagonal."""
        pass

    @abstractmethod
    def smooth_taubin(self, input_ply_path: str, output_ply_path: str,
                      lambda_factor: float = 0.5, mu_factor: Optional[float] = None,
                      iterations: int = 10,
                      log_callback: Optional[Callable[[str], None]] = None) -> bool:
        """Apply Taubin surface smoothing without shrinkage."""
        pass


# ---------------------------------------------------------------------------
# Backend 1: PyMeshLabDirectBackend (In-process PyMeshLab)
# ---------------------------------------------------------------------------

class PyMeshLabDirectBackend(MeshProcessingBackend):
    """In-process PyMeshLab backend (Python 3.10 / ABI matched)."""

    def __init__(self):
        self._ml = None

    @property
    def name(self) -> str:
        return "PyMeshLab (Direct)"

    @property
    def capabilities(self) -> Set[BackendCapability]:
        return {
            BackendCapability.MERGE_BY_DISTANCE,
            BackendCapability.TAUBIN_SMOOTH,
            BackendCapability.HOLE_CLOSING,
            BackendCapability.NONMANIFOLD_REPAIR,
            BackendCapability.DECIMATION,
        }

    def is_available(self) -> bool:
        if self._ml is None:
            self._ml = _try_direct_import_pymeshlab()
        return self._ml is not None

    def _get_ml(self):
        if not self.is_available():
            raise RuntimeError("PyMeshLab is not available for direct in-process execution.")
        return self._ml

    def cleanup(self, input_ply_path: str, output_ply_path: str,
                log_callback: Optional[Callable[[str], None]] = None,
                cleanup_params: Optional[dict] = None) -> bool:
        ml = self._get_ml()
        if cleanup_params is None:
            cleanup_params = {}

        enable_reduction     = bool(cleanup_params.get("enable_reduction", True))
        target_reduction_pct = float(cleanup_params.get("target_reduction_pct", 50))
        remove_duplicates    = bool(cleanup_params.get("remove_duplicates", True))
        repair_nonmanifold   = bool(cleanup_params.get("repair_nonmanifold", True))
        close_holes          = bool(cleanup_params.get("close_holes", True))
        max_hole_size        = int(cleanup_params.get("max_hole_size", 30))

        _log(f"[CLEANUP] Starting Auto Cleanup on: {os.path.basename(input_ply_path)}", log_callback)
        try:
            ms = ml.MeshSet()
            ms.load_new_mesh(input_ply_path)

            init_mesh = ms.current_mesh()
            init_v    = init_mesh.vertex_number()
            init_f    = init_mesh.face_number()
            _log(f"[CLEANUP] Initial mesh: {init_v:,} vertices, {init_f:,} faces", log_callback)

            _log("[CLEANUP] Applying mesh repair filters...", log_callback)
            if remove_duplicates:
                ms.meshing_remove_duplicate_vertices()
                ms.meshing_remove_duplicate_faces()
                ms.meshing_remove_unreferenced_vertices()
                ms.meshing_remove_null_faces()
            if repair_nonmanifold:
                ms.meshing_repair_non_manifold_edges()
                ms.meshing_repair_non_manifold_vertices()
            if close_holes and max_hole_size > 0:
                ms.meshing_close_holes(maxholesize=max_hole_size)

            ms.meshing_remove_connected_component_by_face_number(mincomponentsize=25)
            try:
                ms.meshing_re_orient_faces_coherently()
            except Exception as e:
                _log(f"[CLEANUP] Note: Face re-orientation skipped: {e}", log_callback)
            ms.meshing_merge_close_vertices()

            if enable_reduction and target_reduction_pct > 0:
                target_perc = max(0.05, min(0.95, (100.0 - target_reduction_pct) / 100.0))
                _log(f"[CLEANUP] Applying {int(target_reduction_pct)}% Quadric Edge Collapse Decimation...",
                     log_callback)
                ms.meshing_decimation_quadric_edge_collapse(
                    targetperc=target_perc,
                    qualitythr=0.3,
                    preserveboundary=True,
                    preservenormal=True,
                    preservetopology=True,
                )
            else:
                _log("[CLEANUP] Face reduction disabled. Keeping original face resolution.", log_callback)

            final_mesh = ms.current_mesh()
            final_v    = final_mesh.vertex_number()
            final_f    = final_mesh.face_number()

            os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
            ms.save_current_mesh(output_ply_path)

            if os.path.isfile(output_ply_path) and os.path.getsize(output_ply_path) > 0:
                pct = ((init_f - final_f) / init_f * 100.0) if init_f > 0 else 0.0
                _log(
                    f"[CLEANUP] Auto Cleanup complete: {final_v:,} vertices, {final_f:,} faces "
                    f"({pct:.1f}% face reduction). Saved to {os.path.basename(output_ply_path)}",
                    log_callback,
                )
                return True
            return False
        except Exception as e:
            _log(f"[WARNING] Error during PyMeshLab in-process cleanup: {e}", log_callback)
            return False

    def merge_by_distance(self, input_ply_path: str, output_ply_path: str,
                          threshold_pct: float = 1.0, bbox_diagonal: float = 0.0,
                          log_callback: Optional[Callable[[str], None]] = None) -> bool:
        ml = self._get_ml()
        try:
            ms = ml.MeshSet()
            ms.load_new_mesh(input_ply_path)
            if bbox_diagonal <= 0.0:
                bbox = ms.current_mesh().bounding_box()
                bbox_diagonal = math.sqrt(bbox.dim_x()**2 + bbox.dim_y()**2 + bbox.dim_z()**2)

            abs_threshold = (threshold_pct / 100.0) * bbox_diagonal
            _log(f"[MERGE] PyMeshLab Direct Merge: threshold={threshold_pct:.2f}% ({abs_threshold:.5f})", log_callback)

            if hasattr(ms, "meshing_merge_close_vertices"):
                ms.meshing_merge_close_vertices(threshold=ml.PureValue(abs_threshold))
            elif hasattr(ms, "apply_coord_merge_close_vertices"):
                ms.apply_coord_merge_close_vertices(threshold=ml.PureValue(abs_threshold))
            else:
                ms.meshing_merge_close_vertices()

            # Clean duplicate, null/degenerate faces, and unreferenced vertices produced by vertex merge
            for fn_name in ["meshing_remove_duplicate_faces", "meshing_remove_null_faces", 
                            "meshing_remove_duplicate_vertices", "meshing_remove_unreferenced_vertices",
                            "meshing_repair_non_manifold_edges"]:
                if hasattr(ms, fn_name):
                    try:
                        getattr(ms, fn_name)()
                    except Exception:
                        pass

            os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
            ms.save_current_mesh(output_ply_path)
            return os.path.isfile(output_ply_path) and os.path.getsize(output_ply_path) > 0
        except Exception as e:
            _log(f"[WARNING] PyMeshLab direct merge_by_distance failed: {e}", log_callback)
            return False

    def smooth_taubin(self, input_ply_path: str, output_ply_path: str,
                      lambda_factor: float = 0.5, mu_factor: Optional[float] = None,
                      iterations: int = 10,
                      log_callback: Optional[Callable[[str], None]] = None) -> bool:
        ml = self._get_ml()
        if mu_factor is None:
            mu_factor = -(lambda_factor + 0.01)
        try:
            ms = ml.MeshSet()
            ms.load_new_mesh(input_ply_path)

            taubin_fn = getattr(ms, "apply_coord_taubin_smoothing", None) or \
                        getattr(ms, "meshing_apply_coord_taubin_smoothing", None)
            if taubin_fn is None:
                raise RuntimeError("PyMeshLab Taubin smoothing filter (apply_coord_taubin_smoothing) not available in this PyMeshLab version.")

            _log(f"[SMOOTH] PyMeshLab Direct Taubin Smooth: lambda={lambda_factor}, mu={mu_factor}, steps={iterations}", log_callback)
            try:
                taubin_fn(lambda_val=lambda_factor, mu_val=mu_factor, steps=iterations)
            except TypeError:
                try:
                    taubin_fn(lambda_val=lambda_factor, steps=iterations)
                except TypeError:
                    taubin_fn()

            os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
            ms.save_current_mesh(output_ply_path)
            return os.path.isfile(output_ply_path) and os.path.getsize(output_ply_path) > 0
        except Exception as e:
            _log(f"[WARNING] PyMeshLab direct smooth_taubin failed: {e}", log_callback)
            raise


# ---------------------------------------------------------------------------
# Backend 2: PyMeshLabWorkerBackend (Subprocess Sidecar)
# ---------------------------------------------------------------------------

class PyMeshLabWorkerBackend(MeshProcessingBackend):
    """Subprocess sidecar backend (pymeshlab_worker.py + Python 3.10)."""

    @property
    def name(self) -> str:
        return "PyMeshLab (Worker Subprocess)"

    @property
    def capabilities(self) -> Set[BackendCapability]:
        return {
            BackendCapability.MERGE_BY_DISTANCE,
            BackendCapability.TAUBIN_SMOOTH,
            BackendCapability.HOLE_CLOSING,
            BackendCapability.NONMANIFOLD_REPAIR,
            BackendCapability.DECIMATION,
        }

    def is_available(self) -> bool:
        interpreter = _find_python310()
        worker      = _find_worker_script()
        return bool(interpreter and worker)

    def _run_worker(self, params: dict, log_callback=None) -> bool:
        interpreter = _find_python310()
        worker      = _find_worker_script()

        if not interpreter or not worker:
            return False

        intdir = _internal_dir()
        base   = _app_base_dir()
        env = os.environ.copy()
        for ml_dir in [
            os.path.join(base,   "backend_bin", "pymeshlab_extracted"),
            os.path.join(intdir, "pymeshlab_extracted"),
            os.path.join(intdir, "backend_bin", "pymeshlab_extracted"),
        ]:
            lib_dir = os.path.join(ml_dir, "pymeshlab", "lib")
            if os.path.isdir(lib_dir):
                old_ld = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = lib_dir + (":" + old_ld if old_ld else "")
                break

        cmd = [interpreter, worker, json.dumps(params)]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            result_ok = False
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "log" in obj:
                        _log(obj["log"], log_callback)
                    if "result" in obj:
                        result_ok = bool(obj["result"])
                        if not result_ok and "error" in obj:
                            _log(f"[WARNING] Worker error: {obj['error']}", log_callback)
                except json.JSONDecodeError:
                    _log(f"[CLEANUP] {line}", log_callback)

            proc.wait()
            stderr_out = proc.stderr.read().strip()
            if stderr_out and not result_ok:
                _log(f"[WARNING] Worker stderr: {stderr_out[:500]}", log_callback)

            return result_ok and proc.returncode == 0
        except Exception as e:
            _log(f"[WARNING] PyMeshLab worker subprocess launch failed: {e}", log_callback)
            return False

    def cleanup(self, input_ply_path: str, output_ply_path: str,
                log_callback: Optional[Callable[[str], None]] = None,
                cleanup_params: Optional[dict] = None) -> bool:
        if cleanup_params is None:
            cleanup_params = {}
        worker_params = {
            "action":               "cleanup",
            "input_ply":            input_ply_path,
            "output_ply":           output_ply_path,
            "enable_reduction":     bool(cleanup_params.get("enable_reduction", True)),
            "target_reduction_pct": float(cleanup_params.get("target_reduction_pct", 50)),
            "remove_duplicates":    bool(cleanup_params.get("remove_duplicates", True)),
            "repair_nonmanifold":   bool(cleanup_params.get("repair_nonmanifold", True)),
            "close_holes":          bool(cleanup_params.get("close_holes", True)),
            "max_hole_size":        int(cleanup_params.get("max_hole_size", 30)),
        }
        return self._run_worker(worker_params, log_callback)

    def merge_by_distance(self, input_ply_path: str, output_ply_path: str,
                          threshold_pct: float = 1.0, bbox_diagonal: float = 0.0,
                          log_callback: Optional[Callable[[str], None]] = None) -> bool:
        worker_params = {
            "action":        "merge_by_distance",
            "input_ply":     input_ply_path,
            "output_ply":    output_ply_path,
            "threshold_pct": float(threshold_pct),
            "bbox_diagonal": float(bbox_diagonal),
        }
        return self._run_worker(worker_params, log_callback)

    def smooth_taubin(self, input_ply_path: str, output_ply_path: str,
                      lambda_factor: float = 0.5, mu_factor: Optional[float] = None,
                      iterations: int = 10,
                      log_callback: Optional[Callable[[str], None]] = None) -> bool:
        if mu_factor is None:
            mu_factor = -(lambda_factor + 0.01)
        worker_params = {
            "action":        "smooth_taubin",
            "input_ply":     input_ply_path,
            "output_ply":    output_ply_path,
            "lambda_factor": float(lambda_factor),
            "mu_factor":     float(mu_factor),
            "iterations":    int(iterations),
        }
        return self._run_worker(worker_params, log_callback)


# ---------------------------------------------------------------------------
# Backend 3: Open3DBackend (In-process Open3D)
# ---------------------------------------------------------------------------

class Open3DBackend(MeshProcessingBackend):
    """In-process Open3D backend."""

    def __init__(self):
        self._o3d = None

    @property
    def name(self) -> str:
        return "Open3D"

    @property
    def capabilities(self) -> Set[BackendCapability]:
        return {
            BackendCapability.MERGE_BY_DISTANCE,
            BackendCapability.TAUBIN_SMOOTH,
            BackendCapability.DECIMATION,
        }

    def is_available(self) -> bool:
        if self._o3d is None:
            try:
                import open3d as o3d
                self._o3d = o3d
            except ImportError:
                self._o3d = False
        return self._o3d is not False and self._o3d is not None

    def _get_o3d(self):
        if not self.is_available():
            raise RuntimeError("Open3D is not available in the current environment.")
        return self._o3d

    def cleanup(self, input_ply_path: str, output_ply_path: str,
                log_callback: Optional[Callable[[str], None]] = None,
                cleanup_params: Optional[dict] = None) -> bool:
        o3d = self._get_o3d()
        if cleanup_params is None:
            cleanup_params = {}

        enable_reduction     = bool(cleanup_params.get("enable_reduction", True))
        target_reduction_pct = float(cleanup_params.get("target_reduction_pct", 50))
        remove_duplicates    = bool(cleanup_params.get("remove_duplicates", True))

        try:
            _log(f"[CLEANUP] Open3D Engine: Processing {os.path.basename(input_ply_path)}...", log_callback)
            mesh = o3d.io.read_triangle_mesh(input_ply_path)
            if not mesh.has_vertices() or len(mesh.vertices) == 0:
                _log("[WARNING] Open3D failed to load vertices from mesh.", log_callback)
                return False

            init_v = len(mesh.vertices)
            init_f = len(mesh.triangles)
            _log(f"[CLEANUP] Initial mesh: {init_v:,} vertices, {init_f:,} faces", log_callback)

            if remove_duplicates:
                mesh.remove_duplicated_vertices()
                mesh.remove_duplicated_triangles()
                mesh.remove_unreferenced_vertices()
                mesh.remove_degenerate_triangles()

            if enable_reduction and target_reduction_pct > 0 and init_f > 0:
                target_perc      = max(0.05, min(0.95, (100.0 - target_reduction_pct) / 100.0))
                target_triangles = max(10, int(init_f * target_perc))
                _log(f"[CLEANUP] Applying {int(target_reduction_pct)}% Quadric Edge Collapse Decimation "
                     f"(target {target_triangles:,} faces)...", log_callback)
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
            else:
                _log("[CLEANUP] Face reduction disabled. Keeping original face resolution.", log_callback)

            os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
            o3d.io.write_triangle_mesh(output_ply_path, mesh)

            if os.path.isfile(output_ply_path) and os.path.getsize(output_ply_path) > 0:
                final_v = len(mesh.vertices)
                final_f = len(mesh.triangles)
                pct = ((init_f - final_f) / init_f * 100.0) if init_f > 0 else 0.0
                _log(
                    f"[CLEANUP] Auto Cleanup complete (Open3D): {final_v:,} vertices, {final_f:,} faces "
                    f"({pct:.1f}% face reduction). Saved to {os.path.basename(output_ply_path)}",
                    log_callback,
                )
                return True
            return False
        except Exception as e:
            _log(f"[WARNING] Open3D cleanup failed: {e}", log_callback)
            return False

    def merge_by_distance(self, input_ply_path: str, output_ply_path: str,
                          threshold_pct: float = 1.0, bbox_diagonal: float = 0.0,
                          log_callback: Optional[Callable[[str], None]] = None) -> bool:
        o3d = self._get_o3d()
        try:
            mesh = o3d.io.read_triangle_mesh(input_ply_path)
            if not mesh.has_vertices():
                return False

            if bbox_diagonal <= 0.0:
                extent = mesh.get_axis_aligned_bounding_box().get_extent()
                bbox_diagonal = math.sqrt(extent[0]**2 + extent[1]**2 + extent[2]**2)

            eps = (threshold_pct / 100.0) * bbox_diagonal
            _log(f"[MERGE] Open3D Merge by Distance: threshold_pct={threshold_pct:.2f}%, eps={eps:.5f}", log_callback)

            mesh = mesh.merge_close_vertices(eps)
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_unreferenced_vertices()
            mesh.remove_non_manifold_edges()

            os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
            o3d.io.write_triangle_mesh(output_ply_path, mesh)
            return os.path.isfile(output_ply_path) and os.path.getsize(output_ply_path) > 0
        except Exception as e:
            _log(f"[WARNING] Open3D merge_by_distance failed: {e}", log_callback)
            return False

    def smooth_taubin(self, input_ply_path: str, output_ply_path: str,
                      lambda_factor: float = 0.5, mu_factor: Optional[float] = None,
                      iterations: int = 10,
                      log_callback: Optional[Callable[[str], None]] = None) -> bool:
        o3d = self._get_o3d()
        if mu_factor is None:
            mu_factor = -(lambda_factor + 0.01)
        try:
            mesh = o3d.io.read_triangle_mesh(input_ply_path)
            if not mesh.has_vertices():
                return False

            _log(f"[SMOOTH] Open3D Taubin Smooth: lambda_filter={lambda_factor}, mu={mu_factor}, iterations={iterations}", log_callback)
            mesh = mesh.filter_smooth_taubin(number_of_iterations=iterations, lambda_filter=lambda_factor, mu=mu_factor)

            os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
            o3d.io.write_triangle_mesh(output_ply_path, mesh)
            return os.path.isfile(output_ply_path) and os.path.getsize(output_ply_path) > 0
        except Exception as e:
            _log(f"[WARNING] Open3D smooth_taubin failed: {e}", log_callback)
            return False


# ---------------------------------------------------------------------------
# Backend 4: TrimeshBackend (In-process Trimesh — most limited)
# ---------------------------------------------------------------------------

class TrimeshBackend(MeshProcessingBackend):
    """In-process Trimesh backend (fallback engine)."""

    def __init__(self):
        self._tm = None

    @property
    def name(self) -> str:
        return "Trimesh"

    @property
    def capabilities(self) -> Set[BackendCapability]:
        # Per specification: Trimesh supports neither quadric decimation nor distance merge,
        # hole closing, or non-manifold repair. capabilities returns an empty set.
        return set()

    def is_available(self) -> bool:
        if self._tm is None:
            try:
                import trimesh
                self._tm = trimesh
            except ImportError:
                self._tm = False
        return self._tm is not False and self._tm is not None

    def _get_tm(self):
        if not self.is_available():
            raise RuntimeError("Trimesh is not available in the current environment.")
        return self._tm

    def cleanup(self, input_ply_path: str, output_ply_path: str,
                log_callback: Optional[Callable[[str], None]] = None,
                cleanup_params: Optional[dict] = None) -> bool:
        tm = self._get_tm()
        if cleanup_params is None:
            cleanup_params = {}
        remove_duplicates = bool(cleanup_params.get("remove_duplicates", True))

        try:
            _log(f"[CLEANUP] Trimesh Engine: Processing {os.path.basename(input_ply_path)}...", log_callback)
            mesh = tm.load(input_ply_path, force="mesh")
            init_v = len(mesh.vertices)
            init_f = len(mesh.faces)
            _log(f"[CLEANUP] Initial mesh: {init_v:,} vertices, {init_f:,} faces", log_callback)

            if remove_duplicates:
                mesh.merge_vertices()
                mesh.remove_duplicate_faces()
                mesh.remove_unreferenced_vertices()

            os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
            mesh.export(output_ply_path, file_type="ply")

            if os.path.isfile(output_ply_path) and os.path.getsize(output_ply_path) > 0:
                _log(
                    f"[CLEANUP] Auto Cleanup complete (Trimesh): {len(mesh.vertices):,} vertices, "
                    f"{len(mesh.faces):,} faces. Saved to {os.path.basename(output_ply_path)}",
                    log_callback,
                )
                return True
            return False
        except Exception as e:
            _log(f"[WARNING] Trimesh cleanup failed: {e}", log_callback)
            return False

    def merge_by_distance(self, input_ply_path: str, output_ply_path: str,
                          threshold_pct: float = 1.0, bbox_diagonal: float = 0.0,
                          log_callback: Optional[Callable[[str], None]] = None) -> bool:
        raise NotSupportedError("Merge by distance is not supported by the Trimesh backend.")

    def smooth_taubin(self, input_ply_path: str, output_ply_path: str,
                      lambda_factor: float = 0.5, mu_factor: Optional[float] = None,
                      iterations: int = 10,
                      log_callback: Optional[Callable[[str], None]] = None) -> bool:
        tm = self._get_tm()
        warn_msg = "Smoothing with Laplacian (Taubin unavailable in current backend)"
        _log(f"[WARNING] {warn_msg}", log_callback)
        warnings.warn(warn_msg, DegradedFallbackWarning, stacklevel=2)

        try:
            import trimesh.smoothing
            mesh = tm.load(input_ply_path, force="mesh")
            trimesh.smoothing.filter_laplacian(mesh, lamb=lambda_factor, iterations=iterations)

            os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
            mesh.export(output_ply_path, file_type="ply")
            return os.path.isfile(output_ply_path) and os.path.getsize(output_ply_path) > 0
        except Exception as e:
            _log(f"[WARNING] Trimesh Laplacian smoothing failed: {e}", log_callback)
            return False


# ---------------------------------------------------------------------------
# Factory & Caching Registry
# ---------------------------------------------------------------------------

_CACHED_BACKEND: Optional[MeshProcessingBackend] = None


def select_backend(force_reselect: bool = False) -> MeshProcessingBackend:
    """
    Evaluates backends in priority order and returns the first functional engine:
      1. PyMeshLabDirectBackend  (in-process PyMeshLab)
      2. PyMeshLabWorkerBackend  (subprocess sidecar via pymeshlab_worker.py)
      3. Open3DBackend           (in-process Open3D)
      4. TrimeshBackend          (in-process Trimesh)
    """
    global _CACHED_BACKEND
    if _CACHED_BACKEND is not None and not force_reselect:
        return _CACHED_BACKEND

    # 1. Direct PyMeshLab
    direct = PyMeshLabDirectBackend()
    if direct.is_available():
        _log(f"[BACKEND] Selected 3D processing backend: {direct.name}")
        _CACHED_BACKEND = direct
        return direct

    # 2. PyMeshLab Worker Subprocess
    worker = PyMeshLabWorkerBackend()
    if worker.is_available():
        _log(f"[BACKEND] Selected 3D processing backend: {worker.name}")
        _CACHED_BACKEND = worker
        return worker

    # 3. Open3D
    o3d = Open3DBackend()
    if o3d.is_available():
        _log(f"[BACKEND] Selected 3D processing backend: {o3d.name}")
        _CACHED_BACKEND = o3d
        return o3d

    # 4. Trimesh
    tm = TrimeshBackend()
    if tm.is_available():
        _log(f"[BACKEND] Selected 3D processing backend: {tm.name}")
        _CACHED_BACKEND = tm
        return tm

    raise RuntimeError("No mesh processing backend is available in the current environment.")


def get_backend() -> MeshProcessingBackend:
    """Return the active cached MeshProcessingBackend instance."""
    return select_backend()


def get_backend_capabilities() -> Set[BackendCapability]:
    """Return capabilities set for the active backend."""
    return get_backend().capabilities


# ---------------------------------------------------------------------------
# Legacy Public API Compatibility
# ---------------------------------------------------------------------------

def run_mesh_cleanup(input_ply_path: str, output_ply_path: str,
                     log_callback=None, cleanup_params: dict = None) -> bool:
    """
    Executes automated mesh repair and decimation on input PLY file via active backend.

    Args:
        input_ply_path (str): Path to input mesh PLY file.
        output_ply_path (str): Path to save cleaned PLY file.
        log_callback (callable, optional): Function for logging progress messages.
        cleanup_params (dict, optional): Cleanup options dict.

    Returns:
        bool: True if cleanup succeeded; False otherwise.
    """
    if not os.path.isfile(input_ply_path):
        _log(f"[WARNING] Auto Cleanup input file not found: {input_ply_path}", log_callback)
        return False

    return get_backend().cleanup(input_ply_path, output_ply_path, log_callback, cleanup_params)
