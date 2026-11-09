import gc
import json
from pathlib import Path
from contextlib import nullcontext

from .runtime import strict_loads

VIEWS = ("head", "left_wrist", "right_wrist")

def load_vae(args, runtime, config, files, reports):
    torch = runtime["torch"]
    with strict_loads(torch, reports):
        vae = runtime["AutoencoderKLWan3_8"].from_pretrained(
            str(files["vae"]), additional_kwargs=config["vae_kwargs"]
        )
    return vae.eval().requires_grad_(False).to(args.device, torch.bfloat16)


def encode_texts(args, runtime, config, files, prompts, reports):
    torch = runtime["torch"]
    with strict_loads(torch, reports):
        encoder = runtime["WanT5EncoderModel"].from_pretrained(
            str(files["text_encoder"]),
            additional_kwargs=config["text_encoder_kwargs"],
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
    encoder.eval().requires_grad_(False).to(args.device)
    tokenizer = runtime["AutoTokenizer"].from_pretrained(
        str(files["tokenizer"]), local_files_only=True
    )
    results = {}
    with torch.no_grad():
        for prompt in dict.fromkeys(prompts):
            inputs = tokenizer(
                [prompt],
                padding="max_length",
                max_length=512,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            length = int(inputs.attention_mask.gt(0).sum())
            embedding = encoder(
                inputs.input_ids.to(args.device),
                attention_mask=inputs.attention_mask.to(args.device),
            )[0][0, :length]
            results[prompt] = embedding.detach().cpu().contiguous()
    encoder.to("cpu")
    del encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return results


def make_scheduler(runtime, config):
    cls = runtime["FlowMatchEulerDiscreteScheduler"]
    return cls(**runtime["filter_kwargs"](cls, config["scheduler_kwargs"]))


def model_prediction(
    model,
    payload,
    noisy,
    timestep,
    torch,
    use_empty=False,
    cfg=1.0,
    adapter_enabled=True,
):
    device = noisy.device
    y = payload["y"].to(device)
    ref = payload["full_ref"].to(device)
    context = payload["empty_context" if use_empty else "context"].to(device)
    seq_len = noisy.shape[2] * (noisy.shape[3] // 2) * (noisy.shape[4] // 2)
    if cfg > 1:
        # Each CFG half consists of a complete ordered 3-view episode, preventing
        # negative/positive view groups from interacting in the adapters.
        negative = payload["negative_context"].to(device)
        x = torch.cat((noisy, noisy))
        y, ref = torch.cat((y, y)), torch.cat((ref, ref))
        contexts = [negative] * 3 + [context] * 3
    else:
        x, contexts = noisy, [context] * 3
    t = timestep.expand(x.shape[0], seq_len)
    if "noise_mask" in payload:
        # Native 5B continuation uses t=0 only for the fixed first latent plane.
        mask = payload["noise_mask"].to(device)
        if tuple(mask.shape) != (3, 1, *noisy.shape[2:]):
            raise ValueError("Continuation noise mask does not match latent geometry")
        token_mask = mask[:, 0, :, ::2, ::2].flatten(1)
        if cfg > 1:
            token_mask = torch.cat((token_mask, token_mask))
        t = t * token_mask
    precision = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    extra = {}
    if "view_present" in payload:
        present = payload["view_present"]
        extra["view_present"] = torch.cat((present, present)) if cfg > 1 else present
    with precision:
        output = model(
            x=x,
            t=t,
            context=contexts,
            seq_len=seq_len,
            y=y,
            full_ref=ref,
            **extra,
            adapter_enabled=adapter_enabled,
        )
    if cfg > 1:
        uncond, cond = output.chunk(2)
        output = uncond + cfg * (cond - uncond)
    return output

