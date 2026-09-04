from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

from mesh_cleanup import _app_base_dir, _find_python310, _find_worker_script, _internal_dir


class PyMeshLabPipeline:
    """Strict subprocess client for alignment, fusion, and reconstruction."""

    def __init__(self, log_fn: Optional[Callable[[str], None]] = None):
        self.log = log_fn or (lambda _message: None)

    def run(self, params: Dict) -> Dict:
        interpreter = _find_python310()
        worker = _find_worker_script()
        if not interpreter or not worker:
            raise RuntimeError("The bundled PyMeshLab worker runtime is unavailable")

        environment = os.environ.copy()
        for root in (_app_base_dir(), _internal_dir()):
            for relative in (("backend_bin", "pymeshlab_extracted"), ("pymeshlab_extracted",)):
                package_dir = os.path.join(root, *relative)
                library_dir = os.path.join(package_dir, "pymeshlab", "lib")
                if os.path.isdir(library_dir):
                    previous = environment.get("LD_LIBRARY_PATH", "")
                    environment["LD_LIBRARY_PATH"] = library_dir + ((":" + previous) if previous else "")
                    break

        process = subprocess.Popen(
            [interpreter, worker, json.dumps(params)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        result = None
        try:
            assert process.stdout is not None
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self.log("[PYMESHLAB] " + line)
                    continue
                if "log" in message:
                    self.log(message["log"])
                if "result" in message:
                    result = message
            process.wait()
            stderr = process.stderr.read().strip() if process.stderr else ""
            if process.returncode != 0 or not result or not result.get("result"):
                detail = (result or {}).get("error") or stderr or "unknown worker failure"
                raise RuntimeError("PyMeshLab pipeline failed: " + detail)
            return result
        finally:
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()

    def align(self, inputs: List[str], output_dir: Path, reference_index: int, voxel_size: float,
              quality_distance_multiplier: float = 2.5) -> Dict:
        return self.run({
            "action": "align_point_clouds",
            "input_plys": [str(Path(path).resolve()) for path in inputs],
            "output_dir": str(output_dir.resolve()),
            "reference_index": int(reference_index),
            "voxel_size": float(voxel_size),
            "quality_distance_multiplier": float(quality_distance_multiplier),
        })

    def fuse(self, inputs: List[str], output_path: Path, voxel_size: float,
             normal_neighbors: int, outlier_neighbors: int,
             rejected_output_path: Optional[Path] = None,
             sensor_origins: Optional[List[List[float]]] = None) -> Dict:
        return self.run({
            "action": "fuse_point_clouds",
            "input_plys": [str(Path(path).resolve()) for path in inputs],
            "output_ply": str(output_path.resolve()),
            "voxel_size": float(voxel_size),
            "normal_neighbors": int(normal_neighbors),
            "outlier_neighbors": int(outlier_neighbors),
            "sensor_origins": sensor_origins or [],
            "rejected_output_ply": str(rejected_output_path.resolve()) if rejected_output_path else "",
        })

    def reconstruct(self, input_path: Path, output_path: Path, depth: int, scale: float,
                    normal_neighbors: int, min_component_faces: int,
                    samples_per_node: float = 3.0, support_distance: float = 0.0) -> Dict:
        return self.run({
            "action": "screened_poisson",
            "input_ply": str(input_path.resolve()),
            "output_ply": str(output_path.resolve()),
            "poisson_depth": int(depth),
            "poisson_scale": float(scale),
            "normal_neighbors": int(normal_neighbors),
            "min_component_faces": int(min_component_faces),
            "samples_per_node": float(samples_per_node),
            "support_distance": float(support_distance),
        })
