from __future__ import annotations

import hashlib
import html
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .models import (
    DeepMeshFusionConfig,
    TourQualityCheck,
    TourReadinessIssue,
    TourReadinessResult,
    TourReadinessSummary,
)


@dataclass
class TourReadinessOutput:
    summary: TourReadinessSummary
    checks: List[TourQualityCheck]
    issues: List[TourReadinessIssue]
    assets: List[Dict]
    sources: List[Dict]


class TourReadinessQualityGate:
    """Aggregate all derived evidence into a deterministic virtual-tour handoff decision."""

    REQUIRED_FINAL_ASSETS = {
        "final_obj_path": "geometry-and-uv",
        "final_material_path": "material",
        "final_texture_path": "albedo-texture",
        "final_confidence_path": "texture-confidence",
        "review_map_path": "review-map",
        "report_path": "final-validation",
    }

    def __init__(self, config: DeepMeshFusionConfig):
        self.config = config

    def evaluate(self, workspace_root: str) -> TourReadinessOutput:
        root = Path(workspace_root).resolve()
        manifest = self._read_json(root / "workspace.json")
        issues: List[TourReadinessIssue] = []
        issue_number = 1

        def add(category, severity, blocking, stage, message, regions=1, artifact=None):
            nonlocal issue_number
            issues.append(TourReadinessIssue(
                issue_id=f"gate-issue-{issue_number:04d}", category=category, severity=severity,
                blocking=blocking, source_stage=stage, message=message, region_count=int(regions),
                artifact_path=str(artifact) if artifact else None,
            ))
            issue_number += 1

        geometry_path = root / "analysis" / "geometry_validation.json"
        gap_path = root / "analysis" / "gap_recovery.json"
        photo_path = root / "photogrammetry" / "photogrammetry_registration.json"
        final_path = root / "final" / "final_asset_validation.json"
        geometry_report = self._optional_json(geometry_path, add, "geometry-validation")
        gap_report = self._optional_json(gap_path, add, "gap-recovery")
        photo_report = self._optional_json(photo_path, add, "photogrammetry-registration")
        final_report = self._optional_json(final_path, add, "final-asset")

        geometry_scores = geometry_report.get("summary", {}).get("scores", {})
        final_summary = final_report.get("summary", {})
        final_quality = final_summary.get("quality", {})
        geometry = self._score(geometry_scores.get("overall"))
        completeness = self._score(geometry_scores.get("completeness"))
        consistency = self._score(geometry_scores.get("consistency"))
        texture_coverage = self._score(final_quality.get("coverage"))
        texture_quality = self._score(final_quality.get("texture"))

        critical_defects = sum(
            int(item.get("count", 1)) for item in geometry_report.get("issues", [])
            if item.get("severity") == "critical"
        ) + sum(
            1 for item in final_report.get("remaining_issues", []) if item.get("severity") == "critical"
        )
        geometry_review = int(geometry_report.get("summary", {}).get("review_region_count", 0))
        final_review = int(final_summary.get("review_region_count", 0))
        review_regions = geometry_review + final_review

        unresolved_gaps = int(gap_report.get("summary", {}).get("unresolved_gap_count", 0))
        if unresolved_gaps:
            add("geometry-gaps", "error", True, "gap-recovery", f"{unresolved_gaps} unresolved geometry gaps remain", unresolved_gaps, gap_path)
        if critical_defects:
            add("critical-defects", "critical", True, "geometry-validation", f"{critical_defects} critical mesh defects remain", critical_defects, geometry_path)
        final_errors = [item for item in final_report.get("remaining_issues", []) if item.get("severity") == "error"]
        if final_errors:
            add("texture-confidence-regions", "error", True, "final-asset", f"{len(final_errors)} final texture regions contain blocking defects", len(final_errors), final_path)
        registration = photo_report.get("registration", {})
        if not registration.get("accepted", False):
            add("registration-conflict", "error", True, "photogrammetry-registration", "Photogrammetry-to-LiDAR registration is not accepted", 1, photo_path)
        rejected_passes = [item for item in manifest.get("passes", []) if item.get("enabled", True) and not item.get("registration", {}).get("accepted", False)]
        if rejected_passes:
            add("excluded-scan-passes", "warning", False, "lidar-registration", f"{len(rejected_passes)} enabled LiDAR passes were excluded by registration quality gates", len(rejected_passes), root / "workspace.json")

        assets, artifact_integrity = self._verify_assets(manifest.get("final_asset"), add)
        sources, source_integrity = self._verify_sources(manifest, photo_report, add)

        checks = [
            self._score_check("geometry", "Geometry", geometry, self.config.tour_min_geometry),
            self._score_check("completeness", "Completeness", completeness, self.config.tour_min_completeness),
            self._score_check("surface-consistency", "Surface consistency", consistency, self.config.tour_min_surface_consistency),
            self._score_check("texture-coverage", "Texture coverage", texture_coverage, self.config.tour_min_texture_coverage),
            self._score_check("texture-quality", "Texture quality", texture_quality, self.config.tour_min_texture_quality),
            TourQualityCheck("critical-defects", "Critical defects", float(critical_defects), float(self.config.tour_max_critical_defects), critical_defects <= self.config.tour_max_critical_defects, True, f"{critical_defects} critical defects"),
            TourQualityCheck("review-regions", "Review regions", float(review_regions), float(self.config.tour_max_review_regions), review_regions <= self.config.tour_max_review_regions, True, f"{review_regions} regions require review"),
            TourQualityCheck("source-integrity", "Source integrity", 1.0 if source_integrity else 0.0, 1.0, source_integrity or not self.config.tour_require_source_integrity, self.config.tour_require_source_integrity, "All immutable sources verified" if source_integrity else "One or more source hashes could not be verified"),
            TourQualityCheck("artifact-integrity", "Final artifact integrity", 1.0 if artifact_integrity else 0.0, 1.0, artifact_integrity, True, "All handoff artifacts verified" if artifact_integrity else "Required final artifacts are missing or invalid"),
        ]
        for check in checks:
            if check.blocking and not check.passed and not any(issue.category == check.check_id for issue in issues):
                add(check.check_id, "error", True, "quality-gate", f"{check.label} failed: {check.message}")
        if not final_summary.get("polished_asset_ready", False):
            add("final-polish", "error", True, "final-asset", "Final surface/texture repair has not declared the asset polished", max(final_review, 1), final_path)

        blocking = sum(issue.blocking for issue in issues)
        advisory = len(issues) - blocking
        tour_ready = bool(all(check.passed or not check.blocking for check in checks) and blocking == 0 and final_summary.get("polished_asset_ready", False))
        summary = TourReadinessSummary(
            profile=self.config.tour_quality_profile, geometry=geometry, completeness=completeness,
            surface_consistency=consistency, texture_coverage=texture_coverage,
            texture_quality=texture_quality, critical_defect_count=critical_defects,
            review_region_count=review_regions, blocking_issue_count=blocking,
            advisory_issue_count=advisory, source_integrity_verified=source_integrity,
            artifact_integrity_verified=artifact_integrity, tour_ready=tour_ready,
        )
        return TourReadinessOutput(summary, checks, issues, assets, sources)

    def export(self, output: TourReadinessOutput, root: str) -> TourReadinessResult:
        target = Path(root).resolve(); target.mkdir(parents=True, exist_ok=True)
        report_path = target / "tour_readiness.json"
        html_path = target / "tour_readiness.html"
        asset_manifest_path = target / "tour_asset_manifest.json"
        generated_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema_version": 1, "generated_at": generated_at,
            "summary": asdict(output.summary), "checks": [asdict(item) for item in output.checks],
            "issues": [asdict(item) for item in output.issues], "assets": output.assets,
            "sources": output.sources,
        }
        self._write_json(report_path, payload)
        self._write_json(asset_manifest_path, {
            "schema_version": 1, "generated_at": generated_at,
            "quality_profile": output.summary.profile, "tour_ready": output.summary.tour_ready,
            "assets": output.assets, "source_integrity_verified": output.summary.source_integrity_verified,
        })
        temporary = html_path.with_suffix(".html.tmp")
        temporary.write_text(self._html(output), encoding="utf-8"); temporary.replace(html_path)
        return TourReadinessResult(str(report_path), str(html_path), str(asset_manifest_path), output.summary)

    def _verify_assets(self, final_artifact, add):
        assets, valid = [], True
        if not isinstance(final_artifact, dict):
            add("artifact-integrity", "critical", True, "final-asset", "Final asset manifest entry is missing")
            return assets, False
        for key, role in self.REQUIRED_FINAL_ASSETS.items():
            raw = final_artifact.get(key); path = Path(raw).resolve() if raw else None
            exists = bool(path and path.is_file() and path.stat().st_size > 0)
            item = {"role": role, "path": str(path) if path else None, "exists": exists, "size": path.stat().st_size if exists else 0, "sha256": self._sha256(path) if exists else None}
            assets.append(item)
            if not exists:
                valid = False; add("artifact-integrity", "critical", True, "final-asset", f"Required {role} artifact is missing or empty", 1, path)
        return assets, valid

    def _verify_sources(self, manifest, photo_report, add):
        sources, valid = [], True
        expected = {}
        for scan_pass in manifest.get("passes", []):
            if scan_pass.get("source_path") and scan_pass.get("source_sha256"):
                expected[scan_pass["source_path"]] = (scan_pass["source_sha256"], "lidar-pass")
        for path, digest in photo_report.get("source_hashes", {}).items():
            expected[path] = (digest, "photogrammetry-source")
        if not expected:
            valid = False; add("source-integrity", "error", self.config.tour_require_source_integrity, "provenance", "No immutable source provenance was available")
        for raw, (expected_hash, role) in expected.items():
            path = Path(raw); actual = self._sha256(path) if path.is_file() else None
            matches = actual == expected_hash
            sources.append({"role": role, "path": str(path), "exists": path.is_file(), "expected_sha256": expected_hash, "actual_sha256": actual, "verified": matches})
            if not matches:
                valid = False; add("source-integrity", "error", self.config.tour_require_source_integrity, "provenance", f"Immutable {role} source is missing or changed", 1, path)
        return sources, valid

    def _score_check(self, check_id, label, score, threshold):
        passed = score >= threshold
        return TourQualityCheck(check_id, label, score, threshold, passed, True, f"{score:.0%} against {threshold:.0%} minimum")

    @staticmethod
    def _score(value):
        try: return float(max(0.0, min(1.0, float(value))))
        except (TypeError, ValueError): return 0.0

    @staticmethod
    def _read_json(path):
        if not path.is_file(): return {}
        try: return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): return {}

    def _optional_json(self, path, add, stage):
        payload = self._read_json(path)
        if not payload: add("missing-stage-report", "critical", True, stage, f"Required {stage} report is missing or invalid", 1, path)
        return payload

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _write_json(path, payload):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"); temporary.replace(path)

    @staticmethod
    def _html(output):
        ready = output.summary.tour_ready
        status = "TOUR READY" if ready else "NOT TOUR READY"
        color = "#198754" if ready else "#b42318"
        rows = "".join(
            f"<tr><td>{html.escape(item.label)}</td><td>{'✓' if item.passed else '⚠'}</td><td>{html.escape(item.message)}</td></tr>"
            for item in output.checks
        )
        issue_rows = "".join(
            f"<li class='{html.escape(item.severity)}'><strong>{html.escape(item.category)}</strong> — {html.escape(item.message)} <small>({html.escape(item.source_stage)})</small></li>"
            for item in output.issues
        ) or "<li>No review issues.</li>"
        return f"""<!doctype html><html><head><meta charset='utf-8'><title>Proximap Quality Report</title>
<style>body{{font:16px system-ui;max-width:900px;margin:40px auto;padding:0 24px;color:#1d2939}}h1{{letter-spacing:.04em}}table{{width:100%;border-collapse:collapse}}td{{padding:12px;border-bottom:1px solid #e4e7ec}}.status{{margin:28px 0;padding:20px;text-align:center;font-size:28px;font-weight:700;color:white;background:{color};border-radius:10px}}li{{margin:10px 0}}.critical,.error{{color:#b42318}}.warning{{color:#b54708}}small{{color:#667085}}</style></head>
<body><h1>PROXIMAP QUALITY REPORT</h1><p>Profile: {html.escape(output.summary.profile)}</p><table>{rows}</table>
<div class='status'>{status}</div><h2>Review Issues</h2><ul>{issue_rows}</ul></body></html>"""


TourReadinessService = TourReadinessQualityGate
