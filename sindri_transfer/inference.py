"""Joint three-view sliding inference from depth videos and external references.

No target RGB or training cache is required. --check-only validates inputs and
writes the deterministic window plan without loading model weights.
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from .profile import DEFAULT_PROFILE, load_profile, profile_identity, resize_frame, window_plan
from PIL import Image
from .media import probe_video
from .media import decode_hold, sha256, write_video
from .sliding_window import add_history, enforce_native_start
from .runtime import PROJECT_ROOT, import_runtime, validate_assets, write_summary
from .sampling import (
    VIEWS,
    encode_texts,
    load_vae,
    make_scheduler,
    model_prediction,
)

from .checkpoint import base_provenance, load_denoiser


def inspect_request(path, profile):
    request = json.loads(path.read_text())
    if request.get("schema") != "transfer_request_v1":
        raise ValueError("Expected transfer_request_v1 request")
    if request.get("depth_representation") != profile["depth_representation"]:
        raise ValueError("Controls must be pre-rendered relative disparity, white=near")
    if not request.get("prompt", "").strip():
        raise ValueError("A shared nonempty caption is required")
    total = request["num_frames"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError("num_frames must be a positive integer")
    plan = window_plan(total, profile)
    start_s = float(request.get("start_s", 0))
    if not math.isfinite(start_s) or start_s < 0:
        raise ValueError("start_s must be finite and nonnegative")
    assets, probes = {}, {}
    for name in ("depth_videos", "reference_images"):
        if set(request[name]) != set(VIEWS):
            raise ValueError(f"{name} must have all three views")
        for view, locator in request[name].items():
            asset = Path(locator).expanduser()
            if not asset.is_absolute():
                asset = path.resolve().parent / asset
            asset = asset.resolve()
            if not asset.is_file():
                raise ValueError(f"Input file does not exist: {asset}")
            request[name][view] = str(asset)
            assets[str(asset)] = sha256(asset)
            if name == "depth_videos":
                probe, _ = probe_video(asset)
                if start_s < probe["first_pts_s"] - 1e-7 or (
                    start_s + (total - 1) / profile["fps"] > probe["last_pts_s"] + 1e-7
                ):
                    raise ValueError(f"Requested real control timeline exceeds {asset}")
                probes[view] = probe
            else:
                with Image.open(asset) as image:
                    image.verify()
    return request, {
        "request": request,
        "request_sha256": sha256(path),
        "assets_sha256": assets,
        "generation_profile": profile_identity(profile),
        "windows": plan,
        "control_probes": probes,
        "start_s": start_s,
    }


def control_pixels(request, window, profile, torch):
    """Decode/resize identically to fixed-profile training, padding only the final tail."""
    indices = np.arange(window["start"], window["end"])
    times = float(request.get("start_s", 0)) + indices / profile["fps"]
    selections, videos = {}, []
    for view in VIEWS:
        frames, selections[view] = decode_hold(Path(request["depth_videos"][view]), times)
        if any(np.ptp(frame.astype(np.int16), axis=-1).max() > 2 for frame in frames):
            raise ValueError("Expected gray disparity replicated to RGB; convert colormaps first")
        frames = [resize_frame(f, profile["height"], profile["width"]) for f in frames]
        frames.extend([frames[-1]] * window["pad_frames"])
        videos.append(torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255)
    pts = np.stack([selections[v]["source_pts_s"] for v in VIEWS])
    tolerance = max(1 / selections[v]["fps"] for v in VIEWS) + 1e-6
    if np.ptp(pts, axis=0).max() > tolerance:
        raise ValueError("Cross-view control PTS alignment exceeds a source frame")
    return videos, selections


def stitch_outputs(output, windows, fps, num_frames):
    """Stream through PyAV without an external ffmpeg executable or full-video RAM."""
    from fractions import Fraction

    import av

    videos = {}
    for view in VIEWS:
        destination = output / f"{view}.mp4"
        temporary = output / f"{view}.pending.mp4"
        if destination.exists() or temporary.exists():
            raise FileExistsError("Stitched output exists; choose a new directory")
        count, geometry = 0, None
        with av.open(str(temporary), mode="w") as writer:
            stream = writer.add_stream("libx264", rate=Fraction(str(fps)))
            stream.pix_fmt = "yuv444p"
            stream.options = {"crf": "14", "preset": "medium", "threads": "1"}
            for window in windows:
                segment = output / f"window_{window['index']:04d}" / f"{view}.mp4"
                segment_count = 0
                with av.open(str(segment)) as reader:
                    for frame in reader.decode(video=0):
                        if geometry is None:
                            geometry = (frame.width, frame.height)
                            stream.width, stream.height = geometry
                        if (frame.width, frame.height) != geometry:
                            raise ValueError("Segment geometry changed during stitching")
                        pixels = frame.to_ndarray(format="rgb24")
                        encoded = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                        encoded.pts = count
                        encoded.time_base = Fraction(1, 1) / Fraction(str(fps))
                        for packet in stream.encode(encoded):
                            writer.mux(packet)
                        count += 1
                        segment_count += 1
                if segment_count != window["emit_end"] - window["emit_start"]:
                    raise ValueError("Segment count differs from window plan")
            for packet in stream.encode():
                writer.mux(packet)
        with av.open(str(temporary)) as reader:
            times = [float(frame.time) for frame in reader.decode(video=0)]
        if (
            count != num_frames
            or len(times) != num_frames
            or not np.allclose(times, np.arange(num_frames) / fps, atol=1e-6)
        ):
            raise ValueError("Stitched output count/timeline differs from request")
        temporary.replace(destination)
        videos[view] = str(destination)
    return videos


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True,
                        help="JSON request; asset paths are relative to this file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--model-config", type=Path,
                        default=PROJECT_ROOT / "configs/Transfer.json")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/Transfer.pt")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "models/alibaba-pai/Wan2.2-Fun-5B-Control"
    )
    parser.add_argument(
        "--upstream-dir", type=Path, default=PROJECT_ROOT / "third_party/VideoX-Fun"
    )
    cli = parser.parse_args(argv)
    profile = load_profile(cli.profile)
    request, report = inspect_request(cli.request, profile)
    output = cli.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    report.update(
        status="inputs_checked",
        checkpoint=None if cli.checkpoint is None else str(cli.checkpoint.resolve()),
        quality_validated=False,
    )
    write_summary(output / "plan.json", report)
    if cli.check_only:
        print(json.dumps({"status": "inputs_checked", "windows": len(report["windows"])}))
        return
    if cli.checkpoint is not None and not cli.checkpoint.is_file():
        raise FileNotFoundError(cli.checkpoint)
    runtime = import_runtime(cli.upstream_dir.resolve())
    torch = runtime["torch"]
    if not cli.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for generation; use --check-only on CPU")
    args = SimpleNamespace(
        **{k: v for k, v in vars(cli).items() if k not in ("profile",)},
        height=profile["height"],
        width=profile["width"],
        num_frames=profile["num_frames"],
        fps=profile["fps"],
        condition_scope="per_view",
        gradient_checkpointing=False,
        generation_profile=profile_identity(profile),
    )
    args.model_dir, args.upstream_dir = args.model_dir.resolve(), args.upstream_dir.resolve()
    config, transformer_dir, files, _ = validate_assets(args, runtime)
    if config["scheduler_kwargs"]["shift"] != profile["shift"]:
        raise ValueError("Upstream scheduler no longer matches fixed profile")
    provenance = base_provenance(args)
    if provenance["upstream_commit"] != profile["upstream_commit"]:
        raise ValueError("Upstream revision differs from fixed profile")
    reports = []
    torch.manual_seed(profile["seed"])
    torch.cuda.manual_seed(profile["seed"])
    texts = encode_texts(
        args, runtime, config, files, [request["prompt"], "", profile["negative_prompt"]], reports
    )
    vae = load_vae(args, runtime, config, files, reports)
    model = load_denoiser(args, runtime, config, transformer_dir, provenance, reports).eval()
    ref_latents = []
    with torch.no_grad():
        for view in VIEWS:
            with Image.open(request["reference_images"][view]) as image:
                pixels = resize_frame(np.asarray(image.convert("RGB")), args.height, args.width)
            pixels = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None, :, None].float() / 255
            ref_latents.append(
                vae.encode((pixels * 2 - 1).to(args.device, torch.bfloat16))[0].mode()[0, :, 0]
            )
        full_ref = torch.stack(ref_latents)
        report.update(
            status="running",
            base_provenance=provenance,
            checkpoint_loads=reports,
            checkpoint_sha256=None if not cli.checkpoint else sha256(cli.checkpoint),
            completed_windows=[],
        )
        history = None
        for window in report["windows"]:
            pixels, selections = control_pixels(request, window, profile, torch)
            depth = torch.cat(
                [
                    vae.encode((p[None] * 2 - 1).to(args.device, torch.bfloat16))[0].mode()
                    for p in pixels
                ]
            )
            del pixels
            payload = {
                "y": torch.cat(
                    (
                        depth,
                        torch.zeros(
                            3, 52, *depth.shape[2:], device=args.device, dtype=torch.bfloat16
                        ),
                    ),
                    dim=1,
                ),
                "full_ref": full_ref,
                "context": texts[request["prompt"]],
                "empty_context": texts[""],
                "negative_context": texts[profile["negative_prompt"]],
            }
            payload = add_history(payload, history, vae, torch, args.device, args.num_frames)
            scheduler = make_scheduler(runtime, config)
            scheduler.set_timesteps(profile["inference_steps"], device=args.device)
            seed = profile["seed"] + window["index"]
            generator = torch.Generator(device=args.device).manual_seed(seed)
            latents = torch.randn(
                depth.shape, generator=generator, device=args.device, dtype=torch.bfloat16
            )
            latents *= getattr(scheduler, "init_noise_sigma", 1)
            latents = enforce_native_start(latents, payload)
            for step, timestep in enumerate(scheduler.timesteps):
                prediction = model_prediction(
                    model,
                    payload,
                    latents,
                    timestep,
                    torch,
                    cfg=profile["guidance_scale"],
                )
                latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
                latents = enforce_native_start(latents, payload)
                print(json.dumps({"window": window["index"], "step": step + 1}), flush=True)
            # Store only emitted frames and a float history ring. Never reload compressed
            # MP4 as conditioning, which would introduce accumulating codec errors.
            next_history, paths = [], {}
            window_dir = output / f"window_{window['index']:04d}"
            window_dir.mkdir()
            for i, view in enumerate(VIEWS):
                video = (
                    (vae.decode(latents[i : i + 1]).sample.float() / 2 + 0.5).clamp(0, 1).cpu()[0]
                )
                if tuple(video.shape) != (3, args.num_frames, args.height, args.width):
                    raise ValueError("Unexpected decoded output geometry")
                if not torch.isfinite(video).all():
                    raise ValueError("Nonfinite generated video")
                next_history.append(video[:, -profile["overlap_frames"] :].clone())
                emitted = video[:, window["history_frames"] : window["end"] - window["start"]]
                frames = (emitted.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
                path = window_dir / f"{view}.mp4"
                write_video(path, frames, args.fps)
                paths[view] = str(path)
                del video, emitted, frames
            history = torch.stack(next_history)
            report["completed_windows"].append(
                {**window, "seed": seed, "videos": paths, "control_selections": selections}
            )
            write_summary(output / "summary.json", report)
        report["videos"] = stitch_outputs(
            output, report["windows"], args.fps, request["num_frames"]
        )
        if any(sha256(path) != digest for path, digest in report["assets_sha256"].items()):
            raise ValueError("Input assets changed during generation")
        report["status"] = "generated_quality_unvalidated"
        write_summary(output / "summary.json", report)


if __name__ == "__main__":
    main()
