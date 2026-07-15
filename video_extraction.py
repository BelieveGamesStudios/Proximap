"""
video_extraction.py

Headless video-to-frames extractor for photogrammetry input prep.
Extracts frames at a fixed time interval, with optional blur rejection
via Laplacian variance. Designed to be called standalone or imported
and driven from a QThread worker.

Usage:
    python video_extraction.py --video input.mp4 --output frames/ --interval 0.5
    python video_extraction.py --video input.mp4 --output frames/ --interval 0.5 --blur-threshold 25
"""

import argparse
import cv2
import logging
import sys
from dataclasses import dataclass
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("video_extraction")


@dataclass
class ExtractionResult:
    total_frames_scanned: int
    frames_saved: int
    frames_rejected_blur: int
    output_dir: Path


class VideoExtractionError(Exception):
    pass


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 1.0,
    blur_threshold: float | None = None,
    jpeg_quality: int = 95,
    progress_callback=None,
) -> ExtractionResult:
    """
    Extract frames from a video at a fixed time interval.

    Args:
        video_path: path to the source video file.
        output_dir: directory to write extracted frames into (created if missing).
        interval_seconds: seconds between extracted frames. 0.5 = 2 frames/sec.
        blur_threshold: if set, frames with Laplacian variance below this are
            skipped. Typical usable range is 15-100 depending on source
            resolution/noise; None disables filtering.
        jpeg_quality: 0-100, passed to cv2.imwrite.
        progress_callback: optional callable(current_frame, total_frames) for
            UI progress reporting (e.g. emit a Qt signal from here).

    Returns:
        ExtractionResult with counts and the output path.

    Raises:
        VideoExtractionError: if the video can't be opened or has no frames.
    """
    video_path = Path(video_path)
    out_dir = Path(output_dir)

    if not video_path.is_file():
        raise VideoExtractionError(f"Video file not found: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoExtractionError(f"Could not open video (unsupported codec?): {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0 or total_frames <= 0:
        cap.release()
        raise VideoExtractionError(
            f"Video reports invalid fps/frame count ({video_fps}/{total_frames}); "
            "file may be corrupt or use an unsupported container."
        )

    # Frames to advance between saves. Always >= 1, regardless of how the
    # user's requested interval compares to source fps.
    step = max(round(video_fps * interval_seconds), 1)

    log.info(
        f"Source: {video_fps:.2f} fps, {total_frames} frames "
        f"({total_frames / video_fps:.1f}s) — saving every {step} frames "
        f"(~{step / video_fps:.2f}s interval)"
    )

    frame_id = 0
    saved = 0
    rejected_blur = 0
    digits = len(str(total_frames))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            keep = True
            if blur_threshold is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness < blur_threshold:
                    keep = False
                    rejected_blur += 1

            if keep:
                filename = out_dir / f"frame_{frame_id:0{digits}d}.jpg"
                cv2.imwrite(
                    str(filename), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                saved += 1

        frame_id += 1

        if progress_callback and frame_id % 30 == 0:
            progress_callback(frame_id, total_frames)

    cap.release()

    if progress_callback:
        progress_callback(total_frames, total_frames)

    if saved == 0:
        log.warning(
            "No frames were saved. If using --blur-threshold, try lowering it "
            "or omit it to check the extraction interval is working."
        )

    log.info(
        f"Done: {saved} frames saved, {rejected_blur} rejected for blur, "
        f"{frame_id} frames scanned -> {out_dir}"
    )

    return ExtractionResult(
        total_frames_scanned=frame_id,
        frames_saved=saved,
        frames_rejected_blur=rejected_blur,
        output_dir=out_dir,
    )


def _cli_progress(current: int, total: int) -> None:
    pct = int((current / total) * 100) if total else 0
    sys.stdout.write(f"\r  extracting... {pct:3d}% ({current}/{total})")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video at a fixed interval for photogrammetry input."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--output", required=True, help="Directory to write extracted frames into.")
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Seconds between extracted frames (default: 1.0). Use e.g. 0.5 for 2 fps.",
    )
    parser.add_argument(
        "--blur-threshold", type=float, default=None,
        help="Reject frames with Laplacian variance below this value. Omit to disable filtering.",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95,
        help="JPEG quality 0-100 (default: 95).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-frame progress output.")

    args = parser.parse_args()

    try:
        extract_frames(
            video_path=args.video,
            output_dir=args.output,
            interval_seconds=args.interval,
            blur_threshold=args.blur_threshold,
            jpeg_quality=args.jpeg_quality,
            progress_callback=None if args.quiet else _cli_progress,
        )
    except VideoExtractionError as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()