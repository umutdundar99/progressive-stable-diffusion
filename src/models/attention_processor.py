"""
Custom Attention Processors for Ordinal Disease Progression with Image Conditioning.

This module implements attention processors that combine:
1. AOE (Additive Ordinal Embedding) for disease severity conditioning
2. Image features for patient-specific anatomical conditioning

Key difference from IP-Adapter:
- IP-Adapter: UNet frozen, only IP attention trained
- Our approach: UNet trainable, both attention pathways trainable

Frequency-based conditioning strategy:
- High-resolution layers: Image features dominant (anatomical details)
- Low-resolution layers: AOE dominant (global disease patterns)
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0


class OrdinalIPAttnProcessor2_0(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        num_tokens: int = 4,
        frequency_mode: Literal["both", "aoe_dominant", "image_dominant"] = "both",
        aoe_scale: float = 1.0,
        image_scale: float = 1.0,
        frequency_dominant_scale: float = 1.5,
        frequency_non_dominant_scale: float = 0.5,
    ) -> None:
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("OrdinalIPAttnProcessor2_0 requires PyTorch 2.0+.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.frequency_mode = frequency_mode

        if frequency_mode == "aoe_dominant":
            self.aoe_scale = aoe_scale * frequency_dominant_scale
            self.image_scale = image_scale * frequency_non_dominant_scale
        elif frequency_mode == "image_dominant":
            self.aoe_scale = aoe_scale * frequency_non_dominant_scale
            self.image_scale = image_scale * frequency_dominant_scale
        else:
            self.aoe_scale = aoe_scale
            self.image_scale = image_scale

        self.to_k_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )
        self.to_v_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )

        self.save_attention_maps = True
        self.last_aoe_contribution = (
            None  
        )
        self.last_ip_contribution = (
            None 
        )

    def get_last_attention_maps(self):
        """Return contribution maps: L2 norm of attention output per spatial location."""
        return {"aoe": self.last_aoe_contribution, "ip": self.last_ip_contribution}

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

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

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

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            ip_hidden_states = None
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            aoe_hidden_states = encoder_hidden_states[:, :end_pos, :]
            ip_hidden_states = encoder_hidden_states[:, end_pos:, :]

            encoder_hidden_states = aoe_hidden_states

            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if self.save_attention_maps:
            # Pre-softmax attention logits: (batch, heads, spatial, 1)
            attn_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
                head_dim
            )
            if attention_mask is not None:
                attn_logits = attn_logits + attention_mask
            attn_weights = F.softmax(attn_logits, dim=-1)
            hidden_states = torch.matmul(attn_weights, value)
            self.last_aoe_contribution = (
                attn_logits.squeeze(-1).mean(dim=1).detach().cpu()
            )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(
                1, 2
            )

            if self.save_attention_maps:
                ip_attn_weights = torch.matmul(
                    query, ip_key.transpose(-2, -1)
                ) / math.sqrt(head_dim)
                ip_attn_weights = F.softmax(ip_attn_weights, dim=-1)
                ip_attn_out = torch.matmul(ip_attn_weights, ip_value)
                self.last_ip_contribution = (
                    ip_attn_out.norm(dim=-1).mean(dim=1).detach().cpu()
                )
                ip_hidden_states = ip_attn_out
            else:
                ip_hidden_states = F.scaled_dot_product_attention(
                    query,
                    ip_key,
                    ip_value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            ip_hidden_states = ip_hidden_states.to(query.dtype)

            hidden_states = (
                self.aoe_scale * hidden_states
                + self.image_scale * self.scale * ip_hidden_states
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
    num_tokens: int = 16,
    scale: float = 1.0,
    use_frequency_strategy: bool = True,
    frequency_dominant_scale: float = 1.5,
    frequency_non_dominant_scale: float = 0.5,
) -> Dict[str, nn.Module]:
    attn_procs = {}
    unet_sd = unet.state_dict()

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
            layer_name = name.split(".processor")[0]

            if use_frequency_strategy:
                frequency_mode = get_frequency_mode_for_block(name)
            else:
                frequency_mode = "both"

            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"].clone(),
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"].clone(),
            }

            processor = OrdinalIPAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=scale,
                num_tokens=num_tokens,
                frequency_mode=frequency_mode,
                frequency_dominant_scale=frequency_dominant_scale,
                frequency_non_dominant_scale=frequency_non_dominant_scale,
            )
            processor.load_state_dict(weights, strict=False)
            attn_procs[name] = processor

    unet.set_attn_processor(attn_procs)
    return attn_procs
