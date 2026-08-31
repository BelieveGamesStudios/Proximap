"""
CCCoreLib Registration Backend for Proximap STE (Spatial Texture Engine)

This module provides isolated Python bindings via ctypes to CloudCompare's CCCoreLib,
supporting:
1. Horn Absolute Orientation (closed-form point-pair registration with rigid and scale estimation).
2. ICP (Iterative Closest Point fine registration with optional scale adjustment).
"""

import os
import ctypes
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# Path to native shared library
_BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend_bin")
_LIB_CANDIDATES = [
    os.path.join(_BACKEND_DIR, "libcccorelib_registration.so"),
    os.path.join(_BACKEND_DIR, "libcccorelib_registration.dylib"),
    os.path.join(_BACKEND_DIR, "cccorelib_registration.dll"),
    os.path.join(_BACKEND_DIR, "libcccorelib_registration.dll"),
]

_LIB: Optional[ctypes.CDLL] = None

for _cand in _LIB_CANDIDATES:
    if os.path.exists(_cand):
        try:
            _LIB = ctypes.CDLL(_cand)
            break
        except Exception:
            _LIB = None

if _LIB is not None:
    # int cc_is_available(void)
    _LIB.cc_is_available.argtypes = []
    _LIB.cc_is_available.restype = ctypes.c_int

    # int cc_register_point_pairs(...)
    _LIB.cc_register_point_pairs.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # p_aligned
        ctypes.POINTER(ctypes.c_double),  # p_ref
        ctypes.c_int,                     # count
        ctypes.c_int,                     # adjust_scale (1 or 0)
        ctypes.POINTER(ctypes.c_double),  # out_transform4x4 (16 doubles)
        ctypes.POINTER(ctypes.c_double),  # out_scale (1 double)
        ctypes.POINTER(ctypes.c_double),  # out_rms (1 double)
    ]
    _LIB.cc_register_point_pairs.restype = ctypes.c_int

    # int cc_icp_register(...)
    _LIB.cc_icp_register.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # model_pts
        ctypes.c_int,                     # num_model
        ctypes.POINTER(ctypes.c_double),  # data_pts
        ctypes.c_int,                     # num_data
        ctypes.c_int,                     # adjust_scale
        ctypes.c_double,                  # min_rms_decrease
        ctypes.c_int,                     # max_iterations
        ctypes.c_int,                     # sampling_limit
        ctypes.c_double,                  # overlap_ratio
        ctypes.POINTER(ctypes.c_double),  # out_transform4x4 (16 doubles)
        ctypes.POINTER(ctypes.c_double),  # out_scale
        ctypes.POINTER(ctypes.c_double),  # out_rms
        ctypes.POINTER(ctypes.c_uint),    # out_point_count
    ]
    _LIB.cc_icp_register.restype = ctypes.c_int


@dataclass
class CCCoreLibRegistrationResult:
    """Registration result from CCCoreLib."""
    transform: np.ndarray        # 4x4 matrix representing: P_target = s * R * P_source + T
    rotation: np.ndarray         # 3x3 orthonormal rotation matrix R
    translation: np.ndarray      # 3D translation vector T (shape (3,))
    scale: float                 # Recovered scale factor s
    rms: float                   # Root Mean Square error
    success: bool                # Whether registration succeeded
    point_count: int = 0         # Points used during computation
    error_message: str = ""      # Diagnostic description on failure

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Apply the recovered transformation to an (N, 3) point array:
        P' = s * (points @ R.T) + T
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected points shape (N, 3), got {pts.shape}")
        if pts.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        # Using 4x4 matrix multiplication in homogeneous coordinates
        homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
        transformed = homo @ self.transform.T
        return transformed[:, :3]


def is_cccorelib_available() -> bool:
    """Check whether CCCoreLib native library is loaded and operational."""
    if _LIB is None:
        return False
    try:
        return _LIB.cc_is_available() == 1
    except Exception:
        return False


def register_point_pairs(
    p_aligned: np.ndarray,
    p_ref: np.ndarray,
    adjust_scale: bool = True
) -> CCCoreLibRegistrationResult:
    """
    Perform point-pair registration using CCCoreLib's Horn Absolute Orientation algorithm.
    Target equation: P_ref = s * R * P_aligned + T

    Args:
        p_aligned: (N, 3) array of source points to be aligned.
        p_ref: (N, 3) array of reference target points.
        adjust_scale: If True, computes optimal scale factor s. If False, enforces rigid registration (s=1.0).

    Returns:
        CCCoreLibRegistrationResult containing 4x4 transformation matrix, R, T, scale, and RMS.
    """
    if not is_cccorelib_available():
        return CCCoreLibRegistrationResult(
            transform=np.eye(4, dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
            rms=-1.0,
            success=False,
            error_message="CCCoreLib native library is not available."
        )

    pts_a = np.ascontiguousarray(p_aligned, dtype=np.float64)
    pts_r = np.ascontiguousarray(p_ref, dtype=np.float64)

    if pts_a.ndim != 2 or pts_a.shape[1] != 3 or pts_r.ndim != 2 or pts_r.shape[1] != 3:
        return CCCoreLibRegistrationResult(
            transform=np.eye(4, dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
            rms=-1.0,
            success=False,
            error_message=f"Input point arrays must be (N, 3). Got {pts_a.shape} and {pts_r.shape}."
        )

    if pts_a.shape[0] != pts_r.shape[0]:
        return CCCoreLibRegistrationResult(
            transform=np.eye(4, dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
            rms=-1.0,
            success=False,
            error_message=f"Point count mismatch: {pts_a.shape[0]} aligned vs {pts_r.shape[0]} reference points."
        )

    count = int(pts_a.shape[0])
    if count < 3:
        return CCCoreLibRegistrationResult(
            transform=np.eye(4, dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
            rms=-1.0,
            success=False,
            error_message=f"At least 3 point pairs are required for registration. Provided: {count}."
        )

    if not np.all(np.isfinite(pts_a)) or not np.all(np.isfinite(pts_r)):
        return CCCoreLibRegistrationResult(
            transform=np.eye(4, dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
            rms=-1.0,
            success=False,
            error_message="Input points contain non-finite values (NaN or Inf)."
        )

    out_mat = np.zeros(16, dtype=np.float64)
    out_scale = ctypes.c_double(1.0)
    out_rms = ctypes.c_double(-1.0)

    p_a_ptr = pts_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    p_r_ptr = pts_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_mat_ptr = out_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    ret = _LIB.cc_register_point_pairs(
        p_a_ptr,
        p_r_ptr,
        count,
        1 if adjust_scale else 0,
        out_mat_ptr,
        ctypes.byref(out_scale),
        ctypes.byref(out_rms)
    )

    if ret != 0:
        return CCCoreLibRegistrationResult(
            transform=np.eye(4, dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
            rms=-1.0,
            success=False,
            error_message=f"CCCoreLib Horn registration returned error code {ret}."
        )

    mat_4x4 = out_mat.reshape((4, 4))
    scale_val = float(out_scale.value)
    
    # Extract rotation and translation
    if scale_val > 1e-12:
        r_mat = mat_4x4[:3, :3] / scale_val
    else:
        r_mat = mat_4x4[:3, :3]
    t_vec = mat_4x4[:3, 3]

    return CCCoreLibRegistrationResult(
        transform=mat_4x4,
        rotation=r_mat,
        translation=t_vec,
        scale=scale_val,
        rms=float(out_rms.value),
        success=True,
        point_count=count
    )


def refine_icp(
    model_pts: np.ndarray,
    data_pts: np.ndarray,
    adjust_scale: bool = False,
    min_rms_decrease: float = 1e-5,
    max_iterations: int = 30,
    sampling_limit: int = 50000,
    overlap_ratio: float = 1.0
) -> CCCoreLibRegistrationResult:
    """
    Perform fine ICP registration using CCCoreLib's ICP algorithm (Besl et al.).
    Aligns data_pts (source/moving) to model_pts (reference/fixed).

    Target equation: P_model = s * R * P_data + T
    """
    if not is_cccorelib_available():
        return CCCoreLibRegistrationResult(
            transform=np.eye(4, dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
            rms=-1.0,
            success=False,
            error_message="CCCoreLib native library is not available."
        )

    pts_m = np.ascontiguousarray(model_pts, dtype=np.float64)
    pts_d = np.ascontiguousarray(data_pts, dtype=np.float64)

    if pts_m.ndim != 2 or pts_m.shape[1] != 3 or pts_d.ndim != 2 or pts_d.shape[1] != 3:
        return CCCoreLibRegistrationResult(
            transform=np.eye(4, dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
            rms=-1.0,
            success=False,
            error_message=f"Input point clouds must be (N, 3). Got {pts_m.shape} and {pts_d.shape}."
        )

    if pts_m.shape[0] < 3 or pts_d.shape[0] < 3:
        return CCCoreLibRegistrationResult(
            transform=np.eye(4, dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
            rms=-1.0,
            success=False,
            error_message="At least 3 points in each cloud are required for ICP."
        )

    if not np.all(np.isfinite(pts_m)) or not np.all(np.isfinite(pts_d)):
        return CCCoreLibRegistrationResult(
            transform=np.eye(4, dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
            rms=-1.0,
            success=False,
            error_message="Input points contain non-finite values (NaN or Inf)."
        )

    out_mat = np.zeros(16, dtype=np.float64)
    out_scale = ctypes.c_double(1.0)
    out_rms = ctypes.c_double(-1.0)
    out_pts = ctypes.c_uint(0)

    p_m_ptr = pts_m.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    p_d_ptr = pts_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_mat_ptr = out_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    ret = _LIB.cc_icp_register(
        p_m_ptr,
        int(pts_m.shape[0]),
        p_d_ptr,
        int(pts_d.shape[0]),
        1 if adjust_scale else 0,
        float(min_rms_decrease),
        int(max_iterations),
        int(sampling_limit),
        float(overlap_ratio),
        out_mat_ptr,
        ctypes.byref(out_scale),
        ctypes.byref(out_rms),
        ctypes.byref(out_pts)
    )

    if ret != 0:
        return CCCoreLibRegistrationResult(
            transform=np.eye(4, dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            scale=1.0,
            rms=-1.0,
            success=False,
            error_message=f"CCCoreLib ICP registration returned error code {ret}."
        )

    mat_4x4 = out_mat.reshape((4, 4))
    scale_val = float(out_scale.value)
    
    if scale_val > 1e-12:
        r_mat = mat_4x4[:3, :3] / scale_val
    else:
        r_mat = mat_4x4[:3, :3]
    t_vec = mat_4x4[:3, 3]

    return CCCoreLibRegistrationResult(
        transform=mat_4x4,
        rotation=r_mat,
        translation=t_vec,
        scale=scale_val,
        rms=float(out_rms.value),
        success=True,
        point_count=int(out_pts.value)
    )
