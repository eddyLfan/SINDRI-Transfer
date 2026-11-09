"""Native Wan wrapper used by Transfer; no training/checkpoint writer."""

from __future__ import annotations

import contextvars
import dataclasses
import importlib
import inspect
import types
import weakref
from typing import Any

import torch
from torch import Tensor, nn


def _native_forward_body(native: nn.Module):
    """Bypass only the pinned native CFG batch-splitting decorator.

    Its no-grad branch halves *every* batch, even when CFG skipping is disabled.
    That interprets a head/left/right triple as a CFG batch and breaks the view
    grouping. The underlying function is the unchanged native network math.
    Undecorated/fake models are called normally, including their module hooks.
    """
    bound = native.forward
    decorated = getattr(bound, "__func__", None)
    if decorated is None or decorated.__module__ != "videox_fun.utils.cfg_optimization":
        return native.__call__
    expected_name = "cfg_skip.<locals>.decorator.<locals>.wrapper"
    if decorated.__qualname__ != expected_name:
        raise ValueError("unrecognized native CFG decorator; refusing to bypass it")
    if decorated.__code__.co_freevars != ("func",) or not decorated.__closure__:
        raise ValueError("native CFG decorator closure does not contain the expected func")
    body = decorated.__closure__[0].cell_contents
    if not isinstance(body, types.FunctionType) or body.__name__ != "forward":
        raise ValueError("native CFG decorator does not wrap a forward function")
    if tuple(inspect.signature(body).parameters)[:5] != ("self", "x", "t", "context", "seq_len"):
        raise ValueError("native CFG decorator wraps an unexpected forward signature")
    # Code identity verifies this really is the loaded official cfg_skip factory,
    # rather than an arbitrary wrapper with a similarly named closure variable.
    module = importlib.import_module("videox_fun.utils.cfg_optimization")
    expected = module.cfg_skip()(body)
    if decorated.__code__ is not expected.__code__:
        raise ValueError("native CFG decorator differs from the verified cfg_skip factory")
    return types.MethodType(body, native)


@dataclasses.dataclass(frozen=True)
class NativeConfig:
    """Adapter placement and conditioning identity for strict checkpoint loads.

    ``head_only`` preserves the original head-reference protocol and is the
    default when loading checkpoints written before ``condition_scope`` existed.
    ``per_view`` accepts native references for all three views. This changes the
    conditioning contract, not the native transformer or the adapter operations.
    """

    block_indices: tuple[int, ...] = (5, 11, 17, 23)
    hidden_dim: int = 3072
    adapter_dim: int = 512
    num_heads: int = 8
    condition_scope: str = "head_only"

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_indices", tuple(self.block_indices))
        if not self.block_indices or tuple(sorted(set(self.block_indices))) != self.block_indices:
            raise ValueError("block_indices must be nonempty, unique and increasing")
        if self.block_indices[0] < 0:
            raise ValueError("block_indices must be nonnegative")
        if min(self.hidden_dim, self.adapter_dim, self.num_heads) <= 0:
            raise ValueError("hidden_dim, adapter_dim and num_heads must be positive")
        if self.adapter_dim % self.num_heads:
            raise ValueError("adapter_dim must be divisible by num_heads")
        if self.condition_scope not in ("head_only", "per_view"):
            raise ValueError("condition_scope must be 'head_only' or 'per_view'")




class NativeWanWrapper(nn.Module):
    """Frozen native transformer with trainable residual hooks on four blocks.

    No upstream module or parameter is replaced. Forward-hook layouts are bound
    to each native grid tensor, rather than a mutable "current batch" field.
    Consequently, a checkpoint recomputation uses its own original layout even
    after another forward (including an adapter-disabled baseline) has run.
    Direct calls to ``native`` outside this wrapper remain native calls.
    """

    def __init__(self, native: nn.Module, config: NativeConfig | None = None) -> None:
        super().__init__()
        self.native = native
        self._native_forward = _native_forward_body(native)
        self.multiview_config = config or NativeConfig()
        config = self.multiview_config
        if not hasattr(native, "blocks") or config.block_indices[-1] >= len(native.blocks):
            raise ValueError("adapter block index exceeds the native transformer depth")
        if getattr(native, "dim", config.hidden_dim) != config.hidden_dim:
            raise ValueError("adapter hidden_dim does not match the native transformer")
        native.requires_grad_(False)
        native.eval()
        self.adapters = nn.ModuleDict()
        self._invocation: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
            f"wan_multiview_{id(self)}", default=None
        )
        self._layouts: dict[int, tuple[weakref.ReferenceType[Tensor], bool]] = {}
        self._hook_handles = [
            native.blocks[index].register_forward_hook(self._make_hook(index), with_kwargs=True)
            for index in config.block_indices
        ]

    @property
    def config(self) -> Any:
        """Expose the original native configuration for pipeline compatibility."""
        return self.native.config

    @property
    def dtype(self) -> torch.dtype:
        return next(self.native.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.native.parameters()).device



    def train(self, mode: bool = True) -> NativeWanWrapper:
        super().train(mode)
        # Freezing weights does not remove the differentiable operations after
        # adapters. Keep the native model deterministic, but do not no_grad it.
        self.native.eval()
        return self


    def _check_native_caches_disabled(self) -> None:
        if getattr(self.native, "teacache", None) is not None:
            raise ValueError("TeaCache must be disabled for whole-batch native forward")
        if getattr(self.native, "cfg_skip_ratio", None) is not None:
            raise ValueError("CFG skipping must be disabled for whole-batch native forward")

    def forward_native(self, *args, **kwargs):
        """Unchanged native body over the whole batch, with adapters bypassed.

        Only the cfg_skip decorator's incorrect batch halving is bypassed; no
        network operation is changed. TeaCache/CFG skipping must stay disabled,
        so the native body's cond_flag has no effect on its network operations.
        """
        self._check_native_caches_disabled()
        token = self._invocation.set(None)
        try:
            return self._native_forward(*args, **kwargs)
        finally:
            self._invocation.reset(token)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context,
        seq_len: int,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref: Tensor | None = None,
        subject_ref=None,
        cond_flag: bool = True,
        *,
        adapter_enabled: bool = True,
    ):
        if adapter_enabled:
            if x.ndim != 5 or not x.shape[0] or x.shape[0] % 3:
                raise ValueError("x must be folded as [episodes * 3, C, T, H, W]")
            if full_ref is None or full_ref.ndim != 4 or full_ref.shape[0] != x.shape[0]:
                raise ValueError("full_ref must be [episodes * 3, C, H, W]")
            if full_ref.shape[-2:] != x.shape[-2:]:
                raise ValueError("reference and video latent spatial dimensions must match")
            if self.multiview_config.condition_scope == "head_only":
                if torch.count_nonzero(full_ref[1::3]) or torch.count_nonzero(full_ref[2::3]):
                    raise ValueError("only head may receive a reference; wrist refs must be zero")
            if getattr(self.native, "ref_conv", None) is None:
                raise ValueError("native Fun reference convolution is required")
            if subject_ref is not None or y_camera is not None or clip_fea is not None:
                raise ValueError("subject, camera and CLIP conditioning are outside this adapter")
            if getattr(self.native, "sp_world_size", 1) != 1:
                raise ValueError("sequence parallelism is not supported by these residual hooks")
            if t.shape[0] != x.shape[0] or len(context) != x.shape[0]:
                raise ValueError("t and text context must match the folded view batch")
            grouped_t = t.reshape(-1, 3, *t.shape[1:])
            if not torch.equal(grouped_t[:, 0], grouped_t[:, 1]) or not torch.equal(
                grouped_t[:, 0], grouped_t[:, 2]
            ):
                raise ValueError("all three views of an episode must use the same diffusion time")
        self._check_native_caches_disabled()
        token = self._invocation.set(adapter_enabled)
        try:
            return self._native_forward(
                x=x,
                t=t,
                context=context,
                seq_len=seq_len,
                clip_fea=clip_fea,
                y=y,
                y_camera=y_camera,
                full_ref=full_ref,
                subject_ref=subject_ref,
                cond_flag=cond_flag,
            )
        finally:
            self._invocation.reset(token)




