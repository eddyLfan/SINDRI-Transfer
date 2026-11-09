import gc
import importlib
import json
import os
import struct
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def import_runtime(upstream_dir):
    if not (upstream_dir / "videox_fun").is_dir():
        raise FileNotFoundError(f"Official upstream package not found: {upstream_dir}")
    sys.path.insert(0, str(upstream_dir))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("VIDEOX_ATTENTION_TYPE", "SDPA")
    import torch
    from accelerate import init_empty_weights
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from videox_fun.models import (
        AutoencoderKLWan3_8,
        Wan2_2Transformer3DModel,
        WanT5EncoderModel,
    )
    from videox_fun.pipeline import Wan2_2FunControlPipeline
    from videox_fun.utils import (
        filter_kwargs,
        get_image_latent,
        get_image_to_video_latent,
        get_video_to_video_latent,
        save_videos_grid,
    )
    from videox_fun.utils.gpu_memory_mode import apply_gpu_memory_mode

    return {
        "torch": torch,
        "init_empty_weights": init_empty_weights,
        "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
        "OmegaConf": OmegaConf,
        "AutoTokenizer": AutoTokenizer,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
        "Wan2_2Transformer3DModel": Wan2_2Transformer3DModel,
        "WanT5EncoderModel": WanT5EncoderModel,
        "Wan2_2FunControlPipeline": Wan2_2FunControlPipeline,
        "filter_kwargs": filter_kwargs,
        "get_image_latent": get_image_latent,
        "get_image_to_video_latent": get_image_to_video_latent,
        "get_video_to_video_latent": get_video_to_video_latent,
        "save_videos_grid": save_videos_grid,
        "apply_gpu_memory_mode": apply_gpu_memory_mode,
    }


def checkpoint_header_shapes(path):
    """Read only a safetensors header, including a truncation check."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Incomplete safetensors header: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= 100 * 1024**2:
            raise ValueError(f"Invalid safetensors header length: {path}")
        header = json.loads(stream.read(header_size))
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if any(value["dtype"] not in {"BF16", "F32"} for value in tensors.values()):
        raise ValueError(f"Expected unquantized BF16/F32 transformer weights: {path}")
    payload_size = max((value["data_offsets"][1] for value in tensors.values()), default=0)
    if path.stat().st_size != 8 + header_size + payload_size:
        raise ValueError(f"Checkpoint payload is incomplete or has extra data: {path}")
    return {key: tuple(value["shape"]) for key, value in tensors.items()}


def compare_shapes(expected, actual, label):
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    mismatched = {
        key: {"model": list(expected[key]), "checkpoint": list(actual[key])}
        for key in expected.keys() & actual.keys()
        if expected[key] != actual[key]
    }
    if missing or unexpected or mismatched:
        details = {"missing": missing, "unexpected": unexpected, "mismatched": mismatched}
        raise ValueError(
            f"Strict checkpoint validation failed for {label}: "
            + json.dumps(details, ensure_ascii=False)
        )
    return {"label": label, "keys": len(expected), "missing": 0, "unexpected": 0, "mismatched": 0}


def validate_assets(args, runtime):
    omega = runtime["OmegaConf"]
    config_path = args.upstream_dir / "config/wan2.2/wan_civitai_5b.yaml"
    config = omega.to_container(omega.load(config_path), resolve=True)
    transformer_kwargs = config["transformer_additional_kwargs"]
    transformer_dir = args.model_dir / transformer_kwargs.get(
        "transformer_low_noise_model_subpath", "transformer"
    )
    model_config = json.loads((transformer_dir / "config.json").read_text())
    model_class = runtime["Wan2_2Transformer3DModel"]
    if "has_image_input" in model_config:
        model_config = model_class._convert_from_wan_model_config(model_config)
    if model_config.get("in_dim", model_config.get("in_channels")) != 148:
        raise ValueError("This smoke wrapper requires the official 148-channel Fun-5B-Control")
    if model_config.get("out_dim") != 48 or not model_config.get("add_ref_conv"):
        raise ValueError("Expected 48 latent channels and the official add_ref_conv checkpoint")
    if config["vae_kwargs"].get("vae_type") != "AutoencoderKLWan3_8":
        raise ValueError("Expected the Wan2.2 48-channel, spatial-16 VAE")
    if transformer_kwargs.get("transformer_combination_type", "single") != "single":
        raise ValueError("This wrapper only supports the single-transformer 5B model")

    files = {
        "vae": args.model_dir / config["vae_kwargs"]["vae_subpath"],
        "text_encoder": args.model_dir / config["text_encoder_kwargs"]["text_encoder_subpath"],
        "tokenizer": args.model_dir / config["text_encoder_kwargs"]["tokenizer_subpath"],
    }
    for label, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing local {label}: {path}")
    from diffusers.utils import WEIGHTS_NAME

    if (transformer_dir / WEIGHTS_NAME).exists():
        raise ValueError(
            "Strict header checking requires safetensors, not a preferred .bin checkpoint"
        )
    preferred = transformer_dir / WEIGHTS_NAME.replace(".bin", ".safetensors")
    weight_files = (
        [preferred] if preferred.is_file() else sorted(transformer_dir.glob("*.safetensors"))
    )
    if not weight_files:
        raise FileNotFoundError(f"No local transformer safetensors found: {transformer_dir}")
    shapes = {}
    for path in weight_files:
        shard_shapes = checkpoint_header_shapes(path)
        if shapes.keys() & shard_shapes.keys():
            raise ValueError(f"Duplicate checkpoint keys in {path}")
        shapes.update(shard_shapes)
    additional = dict(transformer_kwargs)
    for old_key, new_key in additional.pop("dict_mapping", {}).items():
        additional[new_key] = model_config[old_key]
    with runtime["init_empty_weights"]():
        model = model_class.from_config(model_config, **additional)
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    report = compare_shapes(expected, shapes, "transformer safetensors headers")
    report["files"] = [{"path": str(path), "bytes": path.stat().st_size} for path in weight_files]
    del model
    gc.collect()
    return config, transformer_dir, files, report


@contextmanager
def strict_loads(torch, reports):
    """Reject upstream's permissive loading, without editing upstream files."""
    original = torch.nn.Module.load_state_dict
    patches = []

    def checked_state_dict(model, state_dict, *args, **kwargs):
        label = type(model).__name__
        report = compare_shapes(
            {key: tuple(value.shape) for key, value in model.state_dict().items()},
            {key: tuple(value.shape) for key, value in state_dict.items()},
            label,
        )
        result = original(model, state_dict, *args, **kwargs)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"Incomplete {label} load: {result}")
        reports.append(report)
        return result

    def wrap_meta_loader(original_meta):
        def checked_meta(model, state_dict, *args, **kwargs):
            report = compare_shapes(
                {key: tuple(value.shape) for key, value in model.state_dict().items()},
                {key: tuple(value.shape) for key, value in state_dict.items()},
                type(model).__name__ + " (meta loader)",
            )
            result = original_meta(model, state_dict, *args, **kwargs)
            reports.append(report)
            return result

        return checked_meta

    torch.nn.Module.load_state_dict = checked_state_dict
    try:
        for module_name in (
            "diffusers.models.model_loading_utils",
            "diffusers.models.modeling_utils",
        ):
            module = importlib.import_module(module_name)
            if hasattr(module, "load_model_dict_into_meta"):
                original_meta = module.load_model_dict_into_meta
                patches.append((module, original_meta))
                module.load_model_dict_into_meta = wrap_meta_loader(original_meta)
        yield
    finally:
        torch.nn.Module.load_state_dict = original
        for module, original_meta in patches:
            module.load_model_dict_into_meta = original_meta

