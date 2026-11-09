"""Native Fun-5B prefix conditioning; no new model parameters or attention changes."""


def add_history(payload, history, vae, torch, device, num_frames):
    """history: CPU float [view, RGB, overlap, H, W], decoded before video encoding.

    Match the native mask/video branch: all known prefix frames enter the 52
    conditioning channels, but ONLY latent frame zero is hard constrained.
    The standalone appearance reference in full_ref remains unchanged.
    """
    if history is None:
        return payload
    if history.ndim != 5 or history.shape[:2] != (3, 3):
        raise ValueError("History must contain a complete RGB view triplet")
    count, height, width = history.shape[2:]
    if not 1 <= count < num_frames or (count - 1) % 4:
        raise ValueError("Native history length must be 4n+1 and shorter than the window")
    if not torch.isfinite(history).all() or history.min() < 0 or history.max() > 1:
        raise ValueError("History must be finite decoded RGB in [0,1]")
    y = payload["y"].clone().to(device)
    latent_shape = (3, 48, (num_frames - 1) // 4 + 1, height // 16, width // 16)
    if tuple(y.shape) != (3, 100, *latent_shape[2:]):
        raise ValueError("History and control geometry disagree")
    masked_latents = []
    for view in range(3):
        # Unknown pixels are zero AFTER normalization, as in the native pipeline.
        pixels = torch.zeros(1, 3, num_frames, height, width, dtype=torch.float32)
        pixels[:, :, :count] = history[view : view + 1].float() * 2 - 1
        masked_latents.append(vae.encode(pixels.to(device, torch.bfloat16))[0].mode())
    masked_latents = torch.cat(masked_latents)
    if tuple(masked_latents.shape) != latent_shape:
        raise ValueError("Unexpected native masked-video VAE shape")
    # For a 4n+1 prefix all four packed mask channels agree. Native resize_mask
    # keeps the already matched latent time axis and spatially constant planes.
    y[:, 48:52] = 0
    y[:, 48:52, : (count - 1) // 4 + 1] = 1
    y[:, 52:] = masked_latents
    noise_mask = torch.ones_like(masked_latents[:, :1])
    noise_mask[:, :, 0] = 0
    return {**payload, "y": y, "masked_latents": masked_latents, "noise_mask": noise_mask}


def enforce_native_start(latents, payload):
    if "noise_mask" not in payload:
        return latents
    mask = payload["noise_mask"].to(latents.device)
    return (1 - mask) * payload["masked_latents"].to(latents.device) + mask * latents
