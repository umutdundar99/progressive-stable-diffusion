from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0


class OrdinalIPAttnProcessor2_0(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        num_image_tokens: int = 16,
        num_aoe_tokens: int = 16,
        frequency_mode: Literal["both", "aoe_dominant", "image_dominant"] = "both",
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_image_tokens = num_image_tokens
        self.num_aoe_tokens = num_aoe_tokens
        self.frequency_mode = frequency_mode

        if frequency_mode == "aoe_dominant":
            self.scale_aoe = 1
            self.scale_ip = 1
        elif frequency_mode == "image_dominant":
            self.scale_aoe = 1
            self.scale_ip = 1
        else:
            self.scale_aoe = 1.0
            self.scale_ip = 1.0

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
        else:
            if attn.norm_cross:
                raise NotImplementedError(
                    "Cross-attention with separate encoder hidden states is not implemented in OrdinalIPAttnProcessor2_0."
                )

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
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1)

        if self.frequency_mode != "both":
            current_tokens = attn_probs.shape[-1]
            scale_vector = torch.ones(
                (1, 1, 1, current_tokens),
                device=attn_probs.device,
                dtype=attn_probs.dtype,
            )

            if current_tokens >= (self.num_aoe_tokens + self.num_image_tokens):
                scale_vector[..., : self.num_aoe_tokens] *= self.scale_aoe
                scale_vector[..., -self.num_image_tokens :] *= self.scale_ip

            attn_probs = attn_probs * scale_vector
            attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True)

        hidden_states = torch.matmul(attn_probs, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

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


def get_frequency_mode_for_block(block_name: str) -> str:
    if "mid_block" in block_name:
        return "aoe_dominant"

    elif "down_blocks" in block_name:
        try:
            block_idx = int(block_name.split("down_blocks.")[1].split(".")[0])
        except (IndexError, ValueError):
            return "both"

        if block_idx <= 1:
            return "image_dominant"
        else:
            return "aoe_dominant"

    elif "up_blocks" in block_name:
        try:
            block_idx = int(block_name.split("up_blocks.")[1].split(".")[0])
        except (IndexError, ValueError):
            return "both"

        if block_idx <= 1:
            return "aoe_dominant"
        else:
            return "image_dominant"

    return "both"


def set_ordinal_ip_attention_processors(
    unet,
    num_image_tokens: int = 16,
    num_aoe_tokens: int = 16,
    use_frequency_strategy: bool = True,
) -> dict:
    attn_procs = {}

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
            attn_procs[name] = AttnProcessor2_0()
        else:
            if use_frequency_strategy:
                frequency_mode = get_frequency_mode_for_block(name)
            else:
                frequency_mode = "both"

            processor = OrdinalIPAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                num_image_tokens=num_image_tokens,
                num_aoe_tokens=num_aoe_tokens,
                frequency_mode=frequency_mode,
            )

            attn_procs[name] = processor

    unet.set_attn_processor(attn_procs)

    return attn_procs
