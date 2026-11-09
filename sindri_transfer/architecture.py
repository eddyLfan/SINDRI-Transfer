"""Transfer: four bidirectional cross-view blocks and shared attention LoRA."""

from __future__ import annotations

import contextvars
import dataclasses
import math
import types
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from .native import NativeWanWrapper, NativeConfig


@dataclasses.dataclass(frozen=True)
class TransferConfig:
    architecture: str = "Transfer"
    block_indices: tuple[int, ...] = (5, 11, 17, 23)
    hidden_dim: int = 3072
    adapter_dim: int = 1024
    num_heads: int = 16
    ffn_dim: int = 4096
    time_dim: int = 128
    condition_scope: str = "per_view"
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("q", "k", "v", "o")
    time_modulation: str = "sinusoidal_mlp_1_plus_tanh_gates"

    def __post_init__(self):
        object.__setattr__(self, "block_indices", tuple(self.block_indices))
        object.__setattr__(self, "lora_targets", tuple(self.lora_targets))
        NativeConfig(
            self.block_indices,
            self.hidden_dim,
            self.adapter_dim,
            self.num_heads,
            self.condition_scope,
        )
        if self.architecture != "Transfer" or self.condition_scope != "per_view":
            raise ValueError("The bidirectional architecture requires per-view conditions")
        if min(self.ffn_dim, self.time_dim, self.lora_rank) < 1 or self.time_dim % 2:
            raise ValueError("Invalid FFN/time/LoRA dimensions")
        if self.lora_targets != ("q", "k", "v", "o") or self.lora_dropout != 0:
            raise ValueError("V2 fixes LoRA to self-attention q/k/v/o with dropout=0")
        if not math.isfinite(self.lora_alpha) or self.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        if self.time_modulation != "sinusoidal_mlp_1_plus_tanh_gates":
            raise ValueError("Unsupported time modulation")


class HeadRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + 1e-6
        )
        return (normalized * self.weight.float()).to(value.dtype)


class BidirectionalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.adapter_dim
        self.input_norm = nn.LayerNorm(config.hidden_dim, elementwise_affine=False)
        self.down = nn.Linear(config.hidden_dim, dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.q_norm = HeadRMSNorm(dim // config.num_heads)
        self.k_norm = HeadRMSNorm(dim // config.num_heads)
        self.role_embedding = nn.Embedding(3, dim)
        nn.init.normal_(self.role_embedding.weight, std=0.02)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, config.ffn_dim), nn.GELU(), nn.Linear(config.ffn_dim, dim)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_dim, config.time_dim),
            nn.SiLU(),
            nn.Linear(config.time_dim, 2 * dim),
        )
        self.up = nn.Linear(dim, config.hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.last_residual_ratio = None
        self.collect_diagnostics = False

    def heads(self, value):
        time, spatial, dim = value.shape
        return value.reshape(
            time, spatial, self.config.num_heads, dim // self.config.num_heads
        ).transpose(1, 2)

    def gates(self, time):
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(self.config.time_dim // 2, device=time.device, dtype=torch.float32)
            / (self.config.time_dim // 2)
        )
        phase = time.float()[..., None] * frequencies
        encoding = torch.cat((phase.cos(), phase.sin()), dim=-1).to(self.down.weight.dtype)
        return (1 + self.time_mlp(encoding).tanh()).reshape(-1, 2, self.config.adapter_dim)

    def forward(self, hidden, grid_sizes, seq_lens, timestep, view_present=None):
        batch, _, channels = hidden.shape
        if batch % 3 or batch == 0 or channels != self.config.hidden_dim:
            raise ValueError("Expected complete ordered three-view groups")
        present = [True] * batch if view_present is None else view_present.tolist()
        if len(present) != batch or any(type(value) is not bool for value in present):
            raise ValueError("view_present must be a boolean per view")
        if any(sum(present[i : i + 3]) < 2 for i in range(0, batch, 3)):
            raise ValueError("Cross-view training needs at least two real views")
        grids, lengths = grid_sizes.tolist(), seq_lens.tolist()
        if len(grids) != batch or len(lengths) != batch or timestep.shape[0] != batch:
            raise ValueError("Layout/time batch differs from hidden batch")
        videos, times, slices = [], [], []
        for index, (time_with_ref, height, width) in enumerate(grids):
            spatial = height * width
            count = (time_with_ref - 1) * spatial
            if min(time_with_ref - 1, height, width) < 1 or lengths[index] != count + spatial:
                raise ValueError("Expected one reference plane followed by real video tokens")
            if lengths[index] > hidden.shape[1]:
                raise ValueError("Real video tokens exceed sequence length")
            video = hidden[index, spatial : spatial + count]
            videos.append(
                self.down(self.input_norm(video.float()).to(self.down.weight.dtype)).reshape(
                    time_with_ref - 1, spatial, -1
                )
            )
            raw_time = timestep[index]
            if raw_time.ndim == 0:
                times.append(raw_time.expand(time_with_ref - 1))
            else:
                if raw_time.numel() < count:
                    raise ValueError("Time map is shorter than the video token sequence")
                times.append(raw_time[:count].reshape(time_with_ref - 1, spatial).float().mean(1))
            slices.append((spatial, spatial + count))
        output = hidden.clone()
        ratios = []
        for first in range(0, batch, 3):
            episode = videos[first : first + 3]
            if len({video.shape[0] for video in episode}) != 1:
                raise ValueError("View timelines must agree within an episode")
            norms = [self.attn_norm(video) for video in episode]
            keys = []
            for role, value in enumerate(norms):
                key = self.k(value)
                keys.append(
                    self.k_norm(self.heads(key + self.role_embedding.weight[role].to(key.dtype)))
                )
            values = [self.heads(self.v(value)) for value in norms]
            for role in range(3):
                if not present[first + role]:
                    continue
                other = [
                    source for source in range(3) if source != role and present[first + source]
                ]
                query = self.q(norms[role])
                query = self.q_norm(
                    self.heads(query + self.role_embedding.weight[role].to(query.dtype))
                )
                attended = (
                    F.scaled_dot_product_attention(
                        query,
                        torch.cat([keys[source] for source in other], dim=2),
                        torch.cat([values[source] for source in other], dim=2),
                        dropout_p=0.0,
                    )
                    .transpose(1, 2)
                    .reshape_as(episode[role])
                )
                gate = self.gates(times[first + role]).to(attended.dtype)
                updated = episode[role] + gate[:, 0, None] * self.o(attended)
                updated = updated + gate[:, 1, None] * self.ffn(self.ffn_norm(updated))
                residual = self.up(updated).flatten(0, 1).to(hidden.dtype)
                start, end = slices[first + role]
                base = hidden[first + role, start:end]
                output[first + role, start:end] = base + residual
                if self.collect_diagnostics:
                    ratios.append(
                        residual.detach().float().square().mean().sqrt()
                        / base.detach().float().square().mean().sqrt().clamp_min(1e-8)
                    )
        if ratios:
            self.last_residual_ratio = torch.stack(ratios).detach().cpu().tolist()
        return output


class LoRAResidual(nn.Module):
    def __init__(self, linear, config):
        super().__init__()
        self.down = nn.Linear(linear.in_features, config.lora_rank, bias=False)
        self.up = nn.Linear(config.lora_rank, linear.out_features, bias=False)
        self.scale = config.lora_alpha / config.lora_rank
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        return self.up(self.down(value.to(self.down.weight.dtype))) * self.scale


class TransferModel(NativeWanWrapper):
    def __init__(self, native, config=None):
        config = config or TransferConfig()
        # Reuse native validation and the verified CFG-decorator bypass. Replace
        # legacy adapters before any forward; no legacy weights enter this model.
        super().__init__(
            native,
            NativeConfig(
                config.block_indices,
                config.hidden_dim,
                config.adapter_dim,
                config.num_heads,
                "per_view",
            ),
        )
        self.multiview_config = config
        self.adapters = nn.ModuleDict(
            {str(i): BidirectionalBlock(config) for i in config.block_indices}
        )
        self.loras = nn.ModuleDict()
        self._call = contextvars.ContextVar(f"bidir_call_{id(self)}", default=None)
        self._lora_enabled = contextvars.ContextVar(f"bidir_lora_{id(self)}", default=False)
        self._records = {}
        for index, block in enumerate(native.blocks):
            for target in config.lora_targets:
                layer = getattr(block.self_attn, target)
                if not isinstance(layer, nn.Linear):
                    raise ValueError("LoRA targets must be native self-attention Linear layers")
                key = f"{index}_{target}"
                self.loras[key] = LoRAResidual(layer, config)
                self._hook_handles.append(layer.register_forward_hook(self._lora_hook(key)))
            block.forward = types.MethodType(self._block_forward(block.forward), block)

    def _record(self, grid):
        key = id(grid)
        old = self._records.get(key)
        if old is not None and old[0]() is grid:
            return old[1]
        state = self._call.get()
        if state is None:
            return None
        owner = weakref.ref(self)

        def remove(reference):
            model = owner()
            if model is not None and model._records.get(key, (None,))[0] is reference:
                model._records.pop(key)

        self._records[key] = (weakref.ref(grid, remove), state)
        return state

    def _block_forward(self, original):
        def forward(module, *args, **kwargs):
            del module
            grid = kwargs.get("grid_sizes", args[3] if len(args) > 3 else None)
            state = self._record(grid) if grid is not None else None
            token = self._lora_enabled.set(state is not None and state[1])
            try:
                return original(*args, **kwargs)
            finally:
                # Also reset on non-reentrant checkpoint early-stop exceptions.
                self._lora_enabled.reset(token)

        return forward

    def _lora_hook(self, key):
        def hook(module, args, output):
            del module
            if self._lora_enabled.get():
                return output + self.loras[key](args[0]).to(output.dtype)
            return output

        return hook

    def _make_hook(self, index):
        def hook(module, args, kwargs, output):
            del module
            grid = kwargs.get("grid_sizes", args[3] if len(args) > 3 else None)
            lengths = kwargs.get("seq_lens", args[2] if len(args) > 2 else None)
            state = self._record(grid)
            if state is not None and state[0]:
                return self.adapters[str(index)](output, grid, lengths, state[2], state[3])
            return output

        return hook

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        *,
        adapter_enabled=True,
        lora_enabled=True,
        view_present=None,
        **kwargs,
    ):
        token = self._call.set((adapter_enabled, lora_enabled, t, view_present))
        try:
            return super().forward(
                x, t, context, seq_len, adapter_enabled=adapter_enabled, **kwargs
            )
        finally:
            self._call.reset(token)

    def forward_native(self, *args, **kwargs):
        t = kwargs.get("t", args[1] if len(args) > 1 else None)
        kwargs.pop("view_present", None)
        token = self._call.set((False, False, t, None))
        try:
            return super().forward_native(*args, **kwargs)
        finally:
            self._call.reset(token)




    def load_learned_state(self, state):
        if set(state) != {"adapters", "loras"}:
            raise ValueError("Checkpoint must contain both modules and LoRA")
        for name in ("adapters", "loras"):
            expected = getattr(self, name).state_dict()
            if state[name].keys() != expected.keys() or any(
                value.shape != expected[key].shape or not torch.isfinite(value).all()
                for key, value in state[name].items()
            ):
                raise ValueError(f"Invalid {name} checkpoint tensors")
        self.adapters.load_state_dict(state["adapters"], strict=True)
        self.loras.load_state_dict(state["loras"], strict=True)
