import hashlib
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def decode_hold(path, targets):
    """Select the latest source PTS not later than each requested target time."""
    selected, pts, indices = [], [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate or 0)
        if fps <= 0:
            raise ValueError(f"Missing source FPS: {path}")
        info = {
            "path": str(Path(path).resolve()),
            "width": stream.width,
            "height": stream.height,
            "fps": fps,
            "declared_frames": stream.frames,
            "time_base": str(stream.time_base),
            "sha256": sha256(path),
        }
        previous, previous_index, last_pts = None, None, None

        def append_previous():
            if previous is None:
                raise ValueError(f"Video starts after requested time: {path}")
            at = float(previous.time)
            error = float(targets[len(pts)]) - at
            if error < -1e-7 or error > max(0.1, 1.6 / fps):
                raise ValueError(f"Source PTS gap at {at}: {path}")
            selected.append(previous.to_ndarray(format="rgb24"))
            pts.append(at)
            indices.append(previous_index)

        for index, frame in enumerate(container.decode(stream)):
            if frame.time is None:
                raise ValueError(f"Missing frame PTS: {path}")
            at = float(frame.time)
            if last_pts is not None and at <= last_pts:
                raise ValueError(f"Nonmonotonic source PTS: {path}")
            last_pts = at
            while len(pts) < len(targets) and targets[len(pts)] < at - 1e-7:
                append_previous()
            if len(pts) == len(targets):
                break
            previous, previous_index = frame, index
        while len(pts) < len(targets):
            append_previous()
    info.update(
        source_pts_s=pts,
        source_frame_indices=indices,
        max_hold_error_s=float(np.max(targets - np.asarray(pts))),
        sampling="latest_source_PTS_at_or_before_target_time",
    )
    return selected, info


def write_video(path, frames, fps):
    temporary = path.with_name(path.stem + ".tmp.mp4")
    with av.open(str(temporary), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(str(fps)))
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        # Preserve chroma resolution in the original INFERNO control.
        stream.pix_fmt = "yuv444p"
        stream.options = {"crf": "14", "preset": "medium", "threads": "1"}
        for index, pixels in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 1) / Fraction(str(fps))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    temporary.replace(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        decoded = [float(frame.time) for frame in container.decode(stream)]
        expected = np.arange(len(frames), dtype=np.float64) / fps
        if len(decoded) != len(frames) or not np.allclose(decoded, expected, atol=1e-6):
            raise ValueError(f"Encoded video timeline mismatch: {path}")
    return {"sha256": sha256(path), "frames": len(frames), "fps": fps}


def probe_video(path):
    """Decode every source PTS and small thumbnails without retaining full frames."""
    pts, thumbnail_times, thumbnails = [], [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate or 0)
        if fps <= 0:
            raise ValueError(f"Missing frame rate: {path}")
        width, height = stream.width, stream.height
        time_base = str(stream.time_base)
        for frame in container.decode(stream):
            if frame.time is None:
                raise ValueError(f"Missing frame PTS: {path}")
            at = float(frame.time)
            if pts and at <= pts[-1]:
                raise ValueError(f"Nonmonotonic source PTS: {path}")
            pts.append(at)
            if not thumbnail_times or at - thumbnail_times[-1] >= 0.24:
                thumbnail_times.append(at)
                pixels = frame.to_ndarray(format="gray")
                thumbnails.append(cv2.resize(pixels, (64, 48), interpolation=cv2.INTER_AREA))
    if len(pts) < 2:
        raise ValueError(f"Video has fewer than two frames: {path}")
    if max(np.diff(pts)) > max(0.1, 1.6 / fps):
        raise ValueError(f"Source has frame timestamp gaps: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "width": width,
        "height": height,
        "fps": fps,
        "time_base": time_base,
        "decoded_frames": len(pts),
        "first_pts_s": pts[0],
        "last_pts_s": pts[-1],
        "pts_s": pts,
    }, (np.asarray(thumbnail_times), np.stack(thumbnails))

