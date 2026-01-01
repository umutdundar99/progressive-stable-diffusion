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

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalIPAttnProcessor(nn.Module):
    """
    Attention processor combining AOE (ordinal class) and Image conditioning.

    This processor handles both conditioning signals in a decoupled manner,
    allowing the model to learn when to use anatomical structure vs disease features.

    Unlike IP-Adapter, we train ALL components:
    - Original UNet attention (to_q, to_k, to_v) - for AOE/class conditioning
    - New image attention (to_k_ip, to_v_ip) - for anatomical conditioning

    Args:
        hidden_size: The hidden dimension of the attention layer
        cross_attention_dim: The dimension of the conditioning embeddings
        scale: Default scale for image conditioning
        num_tokens: Number of image tokens (from ImageProjection)
        frequency_mode: Which conditioning to emphasize
            - "both": Equal weight to AOE and Image
            - "aoe_dominant": AOE has higher weight (for low-res layers)
            - "image_dominant": Image has higher weight (for high-res layers)
        aoe_scale: Scale factor for AOE attention output
        image_scale: Scale factor for Image attention output
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        num_tokens: int = 4,
        frequency_mode: Literal["both", "aoe_dominant", "image_dominant"] = "both",
        aoe_scale: float = 1.0,
        image_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.frequency_mode = frequency_mode

        # Set scales based on frequency mode
        if frequency_mode == "aoe_dominant":
            self.aoe_scale = aoe_scale * 1.5
            self.image_scale = image_scale * 0.5
        elif frequency_mode == "image_dominant":
            self.aoe_scale = aoe_scale * 0.5
            self.image_scale = image_scale * 1.5
        else:  # both
            self.aoe_scale = aoe_scale
            self.image_scale = image_scale

        # Image conditioning layers (TRAINABLE)
        # These project image embeddings to Key and Value for cross-attention
        self.to_k_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )
        self.to_v_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )

        # Optional: Learnable scale parameters
        self.learnable_aoe_scale = nn.Parameter(torch.tensor(self.aoe_scale))
        self.learnable_image_scale = nn.Parameter(torch.tensor(self.image_scale))
        self.use_learnable_scale = False  # Set True to enable learning

    def __call__(
        self,
        attn,  # The original attention module (contains to_q, to_k, to_v)
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with dual conditioning.

        Args:
            attn: Original attention module with to_q, to_k, to_v
            hidden_states: UNet features (B, H*W, C)
            encoder_hidden_states: Concatenated [AOE_embed, Image_embed]
                Shape: (B, aoe_tokens + image_tokens, cross_attention_dim)
            attention_mask: Optional attention mask
            temb: Optional timestep embedding

        Returns:
            Updated hidden states
        """
        residual = hidden_states

        # Handle spatial normalization if present
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        # Reshape 4D input to 3D
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

        # Prepare attention mask
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        # Group normalization if present
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            # Self-attention case
            encoder_hidden_states = hidden_states
            ip_hidden_states = None
        else:
            # Cross-attention case: Split the concatenated embeddings
            # encoder_hidden_states shape: (B, aoe_tokens + image_tokens, D)
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens

            # AOE embeddings (ordinal class conditioning)
            aoe_hidden_states = encoder_hidden_states[:, :end_pos, :]

            # Image embeddings (anatomical conditioning)
            ip_hidden_states = encoder_hidden_states[:, end_pos:, :]

            # Update encoder_hidden_states to only contain AOE
            encoder_hidden_states = aoe_hidden_states

            # Apply cross-normalization if configured
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute AOE attention
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if ip_hidden_states is not None:
            # Project image embeddings to Key and Value
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            # Reshape for multi-head attention
            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)

            # Compute Image attention (same query, different key/value)
            ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
            ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
            ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

            # Store attention map for visualization
            self.attn_map = ip_attention_probs.detach()

            if self.use_learnable_scale:
                aoe_scale = torch.sigmoid(self.learnable_aoe_scale)
                image_scale = torch.sigmoid(self.learnable_image_scale)
            else:
                aoe_scale = self.aoe_scale
                image_scale = self.image_scale

            # Weighted combination
            hidden_states = (
                aoe_scale * hidden_states + image_scale * self.scale * ip_hidden_states
            )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Reshape back to 4D if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        # Residual connection
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class OrdinalIPAttnProcessor2_0(nn.Module):
    """
    PyTorch 2.0 optimized version using scaled_dot_product_attention.

    Same functionality as OrdinalIPAttnProcessor but with:
    - Flash Attention support (faster, memory efficient)
    - Better GPU utilization
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        scale: float = 1.0,
        num_tokens: int = 4,
        frequency_mode: Literal["both", "aoe_dominant", "image_dominant"] = "both",
        aoe_scale: float = 1.0,
        image_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "OrdinalIPAttnProcessor2_0 requires PyTorch 2.0+. "
                "Please upgrade PyTorch."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.frequency_mode = frequency_mode

        # Set scales based on frequency mode
        if frequency_mode == "aoe_dominant":
            self.aoe_scale = aoe_scale * 1.5
            self.image_scale = image_scale * 0.5
        elif frequency_mode == "image_dominant":
            self.aoe_scale = aoe_scale * 0.5
            self.image_scale = image_scale * 1.5
        else:
            self.aoe_scale = aoe_scale
            self.image_scale = image_scale

        # Image conditioning layers
        self.to_k_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )
        self.to_v_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )

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

        # Split AOE and Image embeddings
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

        # AOE Attention
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Flash Attention for AOE
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # Image Attention
        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(
                1, 2
            )

            # Flash Attention for Image
            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            ip_hidden_states = ip_hidden_states.to(query.dtype)

            # Fusion with frequency-based scaling
            hidden_states = (
                self.aoe_scale * hidden_states
                + self.image_scale * self.scale * ip_hidden_states
            )

        # Output projection
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
    """
    Determine the frequency mode based on UNet block location.

    High-resolution blocks (early down, late up) -> Image dominant (anatomy)
    Low-resolution blocks (late down, mid, early up) -> AOE dominant (disease)

    Args:
        block_name: Name of the attention block (e.g., "down_blocks.0.attentions.0")

    Returns:
        frequency_mode: "aoe_dominant", "image_dominant", or "both"
    """

    if "mid_block" in block_name:
        # Mid block = lowest resolution = global disease patterns
        return "aoe_dominant"

    elif "down_blocks" in block_name:
        # Extract block index: down_blocks.X.attentions.Y
        try:
            block_idx = int(block_name.split("down_blocks.")[1].split(".")[0])
        except (IndexError, ValueError):
            return "both"

        # down_blocks.0, down_blocks.1 = high resolution = anatomy
        # down_blocks.2, down_blocks.3 = low resolution = disease
        if block_idx <= 1:
            return "image_dominant"
        else:
            return "aoe_dominant"

    elif "up_blocks" in block_name:
        # Extract block index: up_blocks.X.attentions.Y
        try:
            block_idx = int(block_name.split("up_blocks.")[1].split(".")[0])
        except (IndexError, ValueError):
            return "both"

        # up_blocks.0, up_blocks.1 = low resolution = disease
        # up_blocks.2, up_blocks.3 = high resolution = anatomy
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
) -> dict:
    """
    Replace UNet attention processors with OrdinalIPAttnProcessor.

    Args:
        unet: The UNet model (from diffusers)
        num_tokens: Number of image tokens from ImageProjection
        scale: Global scale for image conditioning
        use_frequency_strategy: Whether to use frequency-based attention weighting

    Returns:
        Dictionary of processor names to processors
    """
    attn_procs = {}
    unet_sd = unet.state_dict()

    for name in unet.attn_processors.keys():
        # Check if this is cross-attention (attn2) or self-attention (attn1)
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )

        # Determine hidden size based on block location
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            # Fallback
            hidden_size = unet.config.block_out_channels[0]

        # Self-attention: Use standard processor
        if cross_attention_dim is None:
            from diffusers.models.attention_processor import AttnProcessor2_0

            attn_procs[name] = AttnProcessor2_0()
        else:
            # Cross-attention: Use our custom processor
            layer_name = name.split(".processor")[0]

            # Get frequency mode for this layer
            if use_frequency_strategy:
                frequency_mode = get_frequency_mode_for_block(name)
            else:
                frequency_mode = "both"

            # Initialize with pretrained weights (warm start)
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"].clone(),
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"].clone(),
            }

            # Create processor
            processor = OrdinalIPAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=scale,
                num_tokens=num_tokens,
                frequency_mode=frequency_mode,
            )
            processor.load_state_dict(weights, strict=False)

            attn_procs[name] = processor

            print(f"  {name}: {frequency_mode} (hidden={hidden_size})")

    # Set all processors at once
    unet.set_attn_processor(attn_procs)

    return attn_procs
