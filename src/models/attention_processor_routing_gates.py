"""
Split-Injection Attention Processors with Ordinal Delta Steering.

Each UNet cross-attention layer receives a 3-segment token sequence::

    encoder_hidden_states = [Target_AOE(N) | E_clean(N) | Delta_AOE(N)]

and routes them through three pathways:

* **Anatomy pathway** — pretrained ``attn.to_k`` / ``attn.to_v``
  processes ``E_clean`` (disease-purified IP-Adapter tokens).
* **Disease pathway** — learned ``to_k_dis`` / ``to_v_dis``
  processes ``Target_AOE`` (desired severity embedding).
* **Delta pathway** — reuses ``to_k_dis`` / ``to_v_dis``
  processes ``Delta_AOE = proj(E[target]) - proj(E[source])``.
  Carries a directional disease-change signal: positive for
  progression, negative for regression, zero when source == target.
  Since ``to_k_dis`` / ``to_v_dis`` have ``bias=False``, zero delta
  produces exactly zero output — safe during training.

**Severity-conditioned dynamic gates** control how much the anatomy
and disease pathways contribute.  Each gate has a learnable base logit
(per block) plus a **per-sample shift** predicted from the Target_AOE
tokens via a small ``gate_modulator`` MLP::

    severity = mean-pool(Target_AOE)          # (B, D)
    shift    = gate_modulator(severity)       # (B, 1)
    anat_gate = σ(base_logit − shift)         # high severity → lower anatomy
    dis_gate  = σ(base_logit + shift)         # high severity → higher disease

The modulator is **zero-initialised** so at training start the gates
are identical to the static baseline, then gradually learn to adapt.

Combination formula::

    z = anat_gate · z_anat + dis_gate · z_dis + delta_scale · z_delta

Gate initialisation follows the block's frequency-hierarchy role:

* ``anatomy`` blocks (low-res: down_blocks.2, mid, up_blocks.1):
  anatomy gate ≈ 0.9, disease gate ≈ 0.1
* ``disease`` blocks (high-res: down_blocks.0-1, up_blocks.2-3):
  anatomy gate ≈ 0.1, disease gate ≈ 0.9

The model can shift these soft gates during training.

**Clinical motivation**: In UC, mucosal vasculature disappears with
increasing severity (Mayo 2–3).  Static gates preserve vascular
pattern regardless of target severity; dynamic gates learn to
suppress anatomical detail when generating high-severity images.

Reference design: DEADiff (architectural disentanglement),
Inv-Adapter (split injection by resolution),
CASteer / NASA (attention-level delta steering).
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0


def _logit(p: float) -> float:
    """Inverse sigmoid: returns x such that sigmoid(x) = p."""
    p = max(min(p, 0.9999), 0.0001)
    return math.log(p / (1.0 - p))


class SplitInjectionAttentionProcessor(nn.Module):
    """Triple-pathway cross-attention with learnable per-block routing.

    Parameters
    ----------
    hidden_size : int
        UNet hidden dimension for this block.
    cross_attention_dim : int | None
        Dimension of ``encoder_hidden_states``.
    num_image_tokens : int
        Number of anatomy (E_clean) tokens per sample.
    num_aoe_tokens : int
        Number of disease (Target_AOE) tokens per sample.
    num_delta_tokens : int
        Number of delta steering (Delta_AOE) tokens per sample.
    block_type : ``"anatomy"`` | ``"disease"`` | ``"both"``
        Determines the initial gate values.
    anat_gate_init : float | None
        Initial sigmoid value for anatomy gate (overrides block_type default).
    dis_gate_init : float | None
        Initial sigmoid value for disease gate (overrides block_type default).
    delta_scale : float
        Non-learnable scale for the delta steering pathway.
        Set to 0.0 during training (delta tokens are zero anyway);
        set to a positive value at inference to enable steering.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        num_image_tokens: int = 16,
        num_aoe_tokens: int = 16,
        num_delta_tokens: int = 16,
        block_type: Literal["anatomy", "disease", "both"] = "both",
        anat_gate_init: Optional[float] = None,
        dis_gate_init: Optional[float] = None,
        delta_scale: float = 0.0,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_image_tokens = num_image_tokens
        self.num_aoe_tokens = num_aoe_tokens
        self.num_delta_tokens = num_delta_tokens
        self.block_type = block_type
        self.delta_scale = delta_scale

        _defaults: dict[str, tuple[float, float]] = {
            "anatomy": (0.9, 0.1),
            "disease": (0.1, 0.9),
            "both": (0.5, 0.5),
        }
        a_init = (
            anat_gate_init if anat_gate_init is not None else _defaults[block_type][0]
        )
        d_init = (
            dis_gate_init if dis_gate_init is not None else _defaults[block_type][1]
        )

        self.anat_gate_logit = nn.Parameter(torch.tensor(_logit(a_init)))
        self.dis_gate_logit = nn.Parameter(torch.tensor(_logit(d_init)))

        self.to_k_dis = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )
        self.to_v_dis = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )

        _mod_dim = cross_attention_dim or hidden_size
        self.gate_modulator = nn.Linear(_mod_dim, 1)
        nn.init.zeros_(self.gate_modulator.weight)
        nn.init.zeros_(self.gate_modulator.bias)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size, sequence_length, _ = encoder_hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        N_aoe = self.num_aoe_tokens
        N_img = self.num_image_tokens
        N_delta = self.num_delta_tokens

        dis_tokens = encoder_hidden_states[:, :N_aoe, :]
        anat_tokens = encoder_hidden_states[:, N_aoe : N_aoe + N_img, :]
        delta_tokens = encoder_hidden_states[:, -N_delta:, :]

        k_anat = attn.to_k(anat_tokens)
        v_anat = attn.to_v(anat_tokens)

        k_dis = self.to_k_dis(dis_tokens)
        v_dis = self.to_v_dis(dis_tokens)

        inner_dim = k_anat.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k_anat = k_anat.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v_anat = v_anat.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k_dis = k_dis.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v_dis = v_dis.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        attn_weights_anat = torch.matmul(query, k_anat.transpose(-2, -1)) / math.sqrt(
            head_dim
        )
        attn_probs_anat = F.softmax(attn_weights_anat, dim=-1)
        z_anat = torch.matmul(attn_probs_anat, v_anat)

        attn_weights_dis = torch.matmul(query, k_dis.transpose(-2, -1)) / math.sqrt(
            head_dim
        )
        attn_probs_dis = F.softmax(attn_weights_dis, dim=-1)
        z_dis = torch.matmul(attn_probs_dis, v_dis)

        severity_signal = dis_tokens.mean(dim=1)
        shift = self.gate_modulator(severity_signal)
        shift = shift.view(batch_size, 1, 1, 1)

        anat_gate = torch.sigmoid(self.anat_gate_logit - shift)
        dis_gate = torch.sigmoid(self.dis_gate_logit + shift)

        if self.delta_scale != 0.0:
            k_delta = self.to_k_dis(delta_tokens)
            v_delta = self.to_v_dis(delta_tokens)
            k_delta = k_delta.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            v_delta = v_delta.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            attn_weights_delta = torch.matmul(
                query, k_delta.transpose(-2, -1)
            ) / math.sqrt(head_dim)
            attn_probs_delta = F.softmax(attn_weights_delta, dim=-1)
            z_delta = torch.matmul(attn_probs_delta, v_delta)

            z_final = anat_gate * z_anat + dis_gate * z_dis + self.delta_scale * z_delta
        else:
            z_final = anat_gate * z_anat + dis_gate * z_dis

        hidden_states = z_final.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def get_block_type(block_name: str) -> str:
    """Assign ``anatomy`` or ``disease`` role based on UNet block position.

    Low-resolution blocks (coarse, global structure) → ``anatomy``
    High-resolution blocks (fine, texture/colour)    → ``disease``

    SD 1.x UNet layout (256×256 input → 32×32 latent)::

        down_blocks.0  320  32×32  cross-attn  → disease  (high-res)
        down_blocks.1  640  16×16  cross-attn  → disease  (medium-res)
        down_blocks.2  1280  8×8   cross-attn  → anatomy  (low-res)
        mid_block      1280  8×8   cross-attn  → anatomy  (bottleneck)
        up_blocks.1    1280  8×8   cross-attn  → anatomy  (low-res)
        up_blocks.2    640  16×16  cross-attn  → disease  (medium-res)
        up_blocks.3    320  32×32  cross-attn  → disease  (high-res)
    """
    if "mid_block" in block_name:
        return "anatomy"

    if "down_blocks" in block_name:
        try:
            idx = int(block_name.split("down_blocks.")[1].split(".")[0])
        except (IndexError, ValueError):
            return "both"
        return "anatomy" if idx >= 2 else "disease"

    if "up_blocks" in block_name:
        try:
            idx = int(block_name.split("up_blocks.")[1].split(".")[0])
        except (IndexError, ValueError):
            return "both"
        return "anatomy" if idx <= 1 else "disease"

    return "both"


def set_split_injection_processors(
    unet,
    num_image_tokens: int = 16,
    num_aoe_tokens: int = 16,
    num_delta_tokens: int = 16,
    use_frequency_strategy: bool = True,
    delta_scale: float = 0.0,
) -> dict:
    """Replace cross-attention processors with :class:`SplitInjectionAttentionProcessor`.

    Self-attention layers (``attn1``) keep the default ``AttnProcessor2_0``.
    Cross-attention layers (``attn2``) get the triple-pathway processor with
    learnable gates initialised by block type and a non-learnable ``delta_scale``.

    After installation the disease K/V weights are **warm-started** from the
    pretrained text K/V to prevent early collapse.
    """
    attn_procs: dict = {}

    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = unet.config.block_out_channels[0]

        if cross_attention_dim is None:
            # Self-attention keep default
            attn_procs[name] = AttnProcessor2_0()
        else:
            block_type: str = get_block_type(name) if use_frequency_strategy else "both"

            processor = SplitInjectionAttentionProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                num_image_tokens=num_image_tokens,
                num_aoe_tokens=num_aoe_tokens,
                num_delta_tokens=num_delta_tokens,
                block_type=block_type,  # type: ignore[arg-type]
                delta_scale=delta_scale,
            )
            attn_procs[name] = processor

    unet.set_attn_processor(attn_procs)

    for _name, module in unet.named_modules():
        if hasattr(module, "processor") and isinstance(
            module.processor, SplitInjectionAttentionProcessor
        ):
            with torch.no_grad():
                module.processor.to_k_dis.weight.copy_(module.to_k.weight)
                module.processor.to_v_dis.weight.copy_(module.to_v.weight)

    return attn_procs
