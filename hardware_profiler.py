"""
Hardware Profiler Utility Module
This module handles hardware diagnostics, memory constraint checks, OpenMVG version validations,
and GPU benchmarking.
"""

import os
import sys
import time
import signal
import atexit
import subprocess
import psutil
import numpy as np

# Invariant check: Under no circumstances should the system query total static RAM
# via platform or os libraries. Only psutil is permitted for dynamic memory checks.

# Track active subprocesses to clean them up on exit to prevent leaks
_active_subprocesses = set()

def _cleanup_subprocesses():
    """Fallback handler to terminate any running subprocesses on interpreter exit."""
    for proc in list(_active_subprocesses):
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass

atexit.register(_cleanup_subprocesses)


class InitializationBlockError(Exception):
    """Exception raised when an initialization constraint or dependency check fails."""
    pass


def get_available_memory() -> int:
    """
    Returns the dynamic available system memory in bytes.
    CRITICAL: Adheres to workspace rule by exclusively using psutil.
    """
    return psutil.virtual_memory().available


def get_total_memory() -> int:
    """
    Returns the total physical system memory in bytes.
    CRITICAL: Adheres to workspace rule by exclusively using psutil.
    """
    return psutil.virtual_memory().total


def run_safe_subprocess(cmd: list, timeout: float = 30.0, **kwargs) -> tuple:
    """
    Runs a subprocess safely with a fallback termination signal handler
    to prevent dangling zombie processes or memory leaks.
    
    Returns:
        tuple: (returncode, stdout, stderr)
    """
    creationflags = 0
    if sys.platform == 'win32':
        creationflags = subprocess.CREATE_NO_WINDOW

    try:
        # Avoid console window flashing on Windows if needed, but standard defaults are fine.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
            **kwargs
        )
        _active_subprocesses.add(proc)
        
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return proc.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            # Terminate first, then kill if it doesn't respond
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise TimeoutError(f"Subprocess timed out after {timeout} seconds: {' '.join(cmd)}")
        finally:
            _active_subprocesses.discard(proc)
    except Exception as e:
        raise e


def validate_colmap_version(colmap_binary_path: str) -> bool:
    """
    Validates that the COLMAP binary is executable and supports CLI help.
    Throws InitializationBlockError if execution fails.
    """
    if not os.path.exists(colmap_binary_path):
        raise FileNotFoundError(f"COLMAP binary does not exist at: {colmap_binary_path}")
        
    try:
        ret, stdout, stderr = run_safe_subprocess([colmap_binary_path, "help"], timeout=5.0)
    except Exception as e:
        raise InitializationBlockError(f"Failed to execute COLMAP binary: {e}")
        
    if ret != 0:
        raise InitializationBlockError(
            f"COLMAP binary execution failed with exit code {ret}.\n"
            f"Output: {(stdout or '') + (stderr or '')}"
        )
        
    return True


def check_nvidia_gputil() -> list:
    """Attempts to discover NVIDIA GPUs using GPUtil."""
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        return [gpu.name for gpu in gpus]
    except Exception:
        return []


def check_nvidia_smi() -> list:
    """Attempts to discover NVIDIA GPUs using nvidia-smi command line utility."""
    paths = [
        "nvidia-smi",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        r"C:\Windows\System32\nvidia-smi.exe"
    ]
    for path in paths:
        try:
            ret, stdout, stderr = run_safe_subprocess(
                [path, "--query-gpu=name", "--format=csv,noheader"],
                timeout=5.0
            )
            if ret == 0 and stdout:
                return [line.strip() for line in stdout.splitlines() if line.strip()]
        except Exception:
            pass
    return []


def check_rocm_smi() -> list:
    """Attempts to discover AMD GPUs using rocm-smi command line utility."""
    try:
        ret, stdout, stderr = run_safe_subprocess(
            ["rocm-smi", "--showproductname"],
            timeout=5.0
        )
        if ret == 0 and stdout:
            names = []
            for line in stdout.splitlines():
                if "Product Name:" in line or "Card" in line:
                    names.append(line.strip())
            return names
    except Exception:
        pass
    return []


def check_gpus_wmic() -> list:
    """Queries Windows WMI for installed video controllers."""
    gpus = []
    try:
        ret, stdout, stderr = run_safe_subprocess(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            timeout=5.0
        )
        if ret == 0 and stdout:
            lines = [line.strip() for line in stdout.splitlines() if line.strip()]
            if len(lines) > 1:
                gpus = lines[1:]  # Skip header "Name"
    except Exception:
        pass
    return gpus


def check_gpus_powershell() -> list:
    """Queries Windows PowerShell Get-CimInstance for installed video controllers."""
    gpus = []
    if sys.platform != 'win32':
        return gpus
    try:
        ret, stdout, stderr = run_safe_subprocess(
            ["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            timeout=5.0
        )
        if ret == 0 and stdout:
            gpus = [line.strip() for line in stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return gpus


def detect_gpus() -> tuple:
    """
    Scans the system for NVIDIA and AMD GPUs.
    Returns:
        tuple: (list of dedicated GPUs, list of integrated GPUs)
    """
    gputil_names = check_nvidia_gputil()
    smi_names = check_nvidia_smi()
    rocm_names = check_rocm_smi()
    wmic_names = check_gpus_wmic()
    powershell_names = check_gpus_powershell()
    
    all_names = set(gputil_names + smi_names + rocm_names + wmic_names + powershell_names)
    
    dgpus = []
    igpus = []
    
    for name in all_names:
        name_upper = name.upper()
        # Classify as integrated/fallback if it fits integrated signatures
        is_integrated = False
        if "INTEL" in name_upper or "IRIS" in name_upper or "HD GRAPHICS" in name_upper:
            is_integrated = True
        elif "AMD RADEON(TM) GRAPHICS" in name_upper or "RADEON VEGA" in name_upper or "APU" in name_upper:
            is_integrated = True
        elif "APPLE" in name_upper:
            is_integrated = True
            
        # dGPU checks
        is_dgpu_brand = (
            "NVIDIA" in name_upper or 
            "AMD" in name_upper or 
            "RADEON" in name_upper or 
            "GEFORCE" in name_upper or 
            "QUADRO" in name_upper or 
            "TESLA" in name_upper
        )
        
        if is_dgpu_brand and not is_integrated:
            dgpus.append(name)
        else:
            igpus.append(name)
            
    return dgpus, igpus


def run_matrix_multiplication_benchmark() -> dict:
    """
    Runs a matrix multiplication compute micro-benchmark targeting exactly 1.0 second.
    Attempts to run on CUDA/GPU via torch or cupy if available, with a NumPy fallback.
    Returns a dict with performance stats.
    """
    # 1. Try PyTorch (CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # 2048x2048 is a solid matrix size for modern GPUs
            N = 2048
            a = torch.randn(N, N, device=device, dtype=torch.float32)
            b = torch.randn(N, N, device=device, dtype=torch.float32)
            
            # Warmup
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            iterations = 0
            start_time = time.time()
            end_time = start_time + 1.0
            
            while time.time() < end_time:
                _ = torch.matmul(a, b)
                iterations += 1
                
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            # 2 * N^3 operations per multiplication
            ops_per_iter = 2 * (N ** 3)
            total_ops = iterations * ops_per_iter
            gflops = (total_ops / elapsed) / 1e9
            
            return {
                "backend": "PyTorch (CUDA)",
                "iterations": iterations,
                "elapsed_seconds": elapsed,
                "gflops": gflops
            }
    except Exception:
        pass

    # 2. Try CuPy (CUDA/ROCm)
    try:
        import cupy as cp
        N = 2048
        a = cp.random.randn(N, N, dtype=cp.float32)
        b = cp.random.randn(N, N, dtype=cp.float32)
        
        # Warmup
        _ = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        
        iterations = 0
        start_time = time.time()
        end_time = start_time + 1.0
        
        while time.time() < end_time:
            _ = cp.dot(a, b)
            iterations += 1
            
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start_time
        
        ops_per_iter = 2 * (N ** 3)
        total_ops = iterations * ops_per_iter
        gflops = (total_ops / elapsed) / 1e9
        
        return {
            "backend": "CuPy",
            "iterations": iterations,
            "elapsed_seconds": elapsed,
            "gflops": gflops
        }
    except Exception:
        pass

    # 3. Fallback to NumPy (CPU)
    # Using a smaller matrix size (e.g. 512x512) for CPU to avoid excessive thread load
    try:
        N = 512
        a = np.random.randn(N, N).astype(np.float32)
        b = np.random.randn(N, N).astype(np.float32)
        
        # Warmup
        _ = np.dot(a, b)
        
        iterations = 0
        start_time = time.time()
        end_time = start_time + 1.0
        
        while time.time() < end_time:
            _ = np.dot(a, b)
            iterations += 1
            
        elapsed = time.time() - start_time
        ops_per_iter = 2 * (N ** 3)
        total_ops = iterations * ops_per_iter
        gflops = (total_ops / elapsed) / 1e9
        
        return {
            "backend": "NumPy (CPU-Fallback)",
            "iterations": iterations,
            "elapsed_seconds": elapsed,
            "gflops": gflops
        }
    except Exception as e:
        return {
            "backend": "None",
            "iterations": 0,
            "elapsed_seconds": 0.0,
            "gflops": 0.0,
            "error": str(e)
        }


# Global/module initialization
dedicated_gpus, integrated_gpus = detect_gpus()

if len(dedicated_gpus) > 0:
    use_low_hardware_fallback = False
    gpu_perf_stats = run_matrix_multiplication_benchmark()
else:
    use_low_hardware_fallback = True
    gpu_perf_stats = None


if __name__ == "__main__":
    print("=== Hardware Profiler Status ===")
    print(f"Available Memory: {get_available_memory() / (1024**3):.2f} GB")
    print(f"Dedicated GPUs Found: {dedicated_gpus}")
    print(f"Integrated/Other GPUs Found: {integrated_gpus}")
    print(f"Use Low Hardware Fallback: {use_low_hardware_fallback}")
    if gpu_perf_stats:
        print(f"Micro-Benchmark Stats:")
        for k, v in gpu_perf_stats.items():
            print(f"  {k}: {v}")
