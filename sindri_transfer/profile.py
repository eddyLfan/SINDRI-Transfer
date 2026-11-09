import hashlib
import json
import math
from pathlib import Path

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "configs/inference.json"

def load_profile(path=DEFAULT_PROFILE):
    profile = json.loads(Path(path).read_text())
    if profile.get("schema") != "transfer_generation_profile_v1":
        raise ValueError("Unknown generation profile schema")
    if profile["geometry"] != "stretch_full_frame" or profile["interpolation"] != (
        "opencv_inter_linear"
    ):
        raise ValueError("Only fixed full-frame bilinear resize is supported")
    if profile["temporal_sampling"] != "latest_source_PTS_at_or_before_target_time":
        raise ValueError("Unsupported temporal sampling policy")
    if profile["scheduler"] != "FlowMatchEulerDiscreteScheduler" or profile["shift"] != 5:
        raise ValueError("This profile requires the native Flow scheduler with shift 5")
    if profile["dtype"] != "bfloat16":
        raise ValueError("This profile requires BF16")
    for key, expected in {
        "depth_representation": "relative_disparity_u8_white_near",
        "tail_policy": "repeat_last_depth_frame_then_trim_output",
        "history_policy": "native_video_mask_prefix_first_latent_hard_rest_soft",
        "reference_policy": "fixed_external_per_view_reference_all_windows",
    }.items():
        if profile[key] != expected:
            raise ValueError(f"Unsupported {key}")
    if (
        profile["inference_steps"] < 1
        or not math.isfinite(profile["guidance_scale"])
        or (profile["guidance_scale"] < 1)
    ):
        raise ValueError("Invalid sampling steps/CFG")
    if any(
        profile[k]
        for k in (
            "random_crop",
            "random_flip",
            "letterbox",
            "teacache",
            "cfg_skip",
            "sequence_parallel",
        )
    ):
        raise ValueError("Augmentation and approximate inference switches must stay disabled")
    if min(profile["width"], profile["height"]) < 32 or any(
        profile[k] % 32 for k in ("width", "height")
    ):
        raise ValueError("Profile geometry must be positive multiples of 32")
    if profile["fps"] <= 0 or profile["num_frames"] < 5 or (profile["num_frames"] - 1) % 4:
        raise ValueError("Invalid profile FPS or frame count")
    overlap = profile["overlap_frames"]
    if overlap < 1 or overlap >= profile["num_frames"] or (overlap - 1) % 4:
        raise ValueError("Overlap must be 4n+1 and shorter than a window")
    if profile["stride_frames"] != profile["num_frames"] - overlap:
        raise ValueError("Stride must equal window length minus overlap")
    return profile


def profile_identity(profile):
    encoded = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "config": profile}


def resize_frame(frame, height, width):
    import cv2
    import numpy as np

    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def window_plan(total_frames, profile):
    """Half-open global ranges; retain old overlap, discard the new predicted prefix."""
    if not isinstance(total_frames, int) or total_frames < 1:
        raise ValueError("total_frames must be a positive integer")
    length, overlap = profile["num_frames"], profile["overlap_frames"]
    result, start = [], 0
    while True:
        end = min(start + length, total_frames)
        result.append(
            {
                "index": len(result),
                "start": start,
                "end": end,
                "history_frames": overlap if start else 0,
                "emit_start": start + overlap if start else 0,
                "emit_end": end,
                "pad_frames": start + length - end,
            }
        )
        if end == total_frames:
            return result
        start += profile["stride_frames"]

