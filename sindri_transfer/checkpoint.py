"""Strict loading for the official Transfer 20K inference checkpoint."""

import dataclasses
import json
import subprocess
from pathlib import Path

from .media import sha256
from .runtime import PROJECT_ROOT, strict_loads

INFERENCE_FORMAT = "transfer_inference_v1"


def base_provenance(args):
    # Use the release's known base identity, never trust a user-supplied hash alone.
    manifest = json.loads((PROJECT_ROOT / "configs/base_model.json").read_text())
    for relative, expected in manifest["files_sha256"].items():
        path = args.model_dir / relative
        if sha256(path) != expected:
            raise ValueError(f"Base model hash mismatch: {path}")
    commit = subprocess.check_output(
        ["git", "-C", str(args.upstream_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "transformer_sha256": manifest["files_sha256"]["diffusion_pytorch_model.safetensors"],
        "upstream_commit": commit,
    }


def read_checkpoint(path):
    import torch

    release = json.loads((PROJECT_ROOT / "configs/release.json").read_text())
    if sha256(path) != release["sha256"]:
        raise ValueError("Checkpoint differs from the official Transfer 20K release")
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if value.get("format") != INFERENCE_FORMAT:
        raise ValueError("Expected the official Transfer inference checkpoint")
    for key in ("model_config", "identity", "learned", "step", "stage"):
        if key not in value:
            raise ValueError(f"Checkpoint is missing {key}")
    if value["step"] != 20000 or value["stage"] != "joint":
        raise ValueError("Only the official Transfer 20K checkpoint is supported")
    return value


def load_denoiser(args, runtime, config, transformer_dir, provenance, reports):
    from .architecture import TransferConfig, TransferModel

    torch = runtime["torch"]
    model_config = TransferConfig(**json.loads(Path(args.model_config).read_text()))
    value = read_checkpoint(args.checkpoint)
    if value is not None:
        if value["model_config"] != dataclasses.asdict(model_config):
            raise ValueError("Bidirectional model configuration mismatch")
        if value["identity"]["base_provenance"] != provenance:
            raise ValueError("Native base checkpoint provenance mismatch")
        if value["identity"]["generation_profile"] != args.generation_profile:
            raise ValueError("Inference profile differs from checkpoint")
    with strict_loads(torch, reports):
        native = runtime["Wan2_2Transformer3DModel"].from_pretrained(
            str(transformer_dir),
            transformer_additional_kwargs=config["transformer_additional_kwargs"],
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
    native.eval().requires_grad_(False).to(args.device)
    native.gradient_checkpointing = False
    model = TransferModel(native, model_config)
    model.adapters.to(args.device, torch.float32)
    model.loras.to(args.device, torch.float32)
    if value is not None:
        model.load_learned_state(value["learned"])
        reports.append({"label": "bidirectional learned weights", "step": value["step"],
                        "stage": value["stage"], "format": value["format"]})
    return model.eval()

