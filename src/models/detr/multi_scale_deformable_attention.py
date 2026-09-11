"""Multi-scale deformable attention implementation."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiScaleDeformableAttention(nn.Module):
    """Sample and aggregate sparse locations from multiple feature levels."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
    ) -> None:
        """Initialize projections for offsets, weights, values, and output.

        Args:
            hidden_dim: Model/channel dimension. Must be divisible by
                ``num_heads``.
            num_heads: Number of independent attention heads.
            num_levels: Number of feature-map levels represented in
                ``input_flatten``.
            num_points: Number of sampling points per head and level.
        """
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if num_levels <= 0:
            raise ValueError("num_levels must be positive")
        if num_points <= 0:
            raise ValueError("num_points must be positive")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = hidden_dim // num_heads

        self.sampling_offsets = nn.Linear(
            hidden_dim, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            hidden_dim, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize projections and radial sampling-offset biases."""
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        # Initialize sampling_offsets bias with angle-based grid
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        input_flatten: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        input_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-scale deformable attention with bilinear sampling.

        Args:
            query: Query tensor with shape [B, Lq, hidden_dim].
            input_flatten: Flattened source feature tensor with shape [B, S, hidden_dim].
            reference_points: Reference locations for each level with shape
                [B, Lq, num_levels, 2].
            spatial_shapes: Level-wise feature map sizes with shape [num_levels, 2].
            level_start_index: First index for each level in the flattened sequence.
            input_padding_mask: Mask for padded source tokens with shape [B, S].
                ``True`` entries are zeroed before sampling.

        Returns:
            Attention output with shape [B, Lq, hidden_dim].
        """
        if query.dim() != 3 or query.shape[2] != self.hidden_dim:
            raise ValueError(f"query must be [B, Lq, {self.hidden_dim}]")
        if input_flatten.dim() != 3 or input_flatten.shape[2] != self.hidden_dim:
            raise ValueError(f"input_flatten must be [B, S, {self.hidden_dim}]")
        if query.shape[0] != input_flatten.shape[0]:
            raise ValueError("query and input_flatten must have the same batch size")
        if (
            reference_points.dim() != 4
            or reference_points.shape[2] != self.num_levels
            or reference_points.shape[3] != 2
        ):
            raise ValueError(f"reference_points must be [B, Lq, {self.num_levels}, 2]")
        if reference_points.shape[0] != query.shape[0]:
            raise ValueError("reference_points and query must have the same batch size")
        if reference_points.shape[1] != query.shape[1]:
            raise ValueError(
                "reference_points and query must have the same number of queries"
            )
        if spatial_shapes.shape != (self.num_levels, 2):
            raise ValueError(f"spatial_shapes must be [{self.num_levels}, 2]")
        if level_start_index.shape != (self.num_levels,):
            raise ValueError(f"level_start_index must be [{self.num_levels}]")
        if input_padding_mask.shape != input_flatten.shape[:2]:
            raise ValueError("input_padding_mask must have shape [B, S]")
        if input_flatten.shape[1] != spatial_shapes.prod(dim=1).sum().item():
            raise ValueError("input_flatten sequence length must match spatial_shapes")
        # In encoder self-attention, query and input_flatten usually describe the
        # same source tokens. They have different roles, though: the query predicts
        # where to sample, while input_flatten supplies the values to be sampled.
        # A caller will commonly pass query = src + positional_encoding and
        # input_flatten = src. The positional signal is therefore available to the
        # offset/weight predictors, while the values remain the image features.
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        batch_size, num_queries, _ = query.shape

        # One query produces a separate 2D offset for every head, feature level,
        # and sampling point. The final 2 stores (x, y), so the flat linear output
        # is reorganized into the shape used by the attention equations.
        offsets = self.sampling_offsets(query).view(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        # The linear layer emits one scalar per head/level/point. Softmax is done
        # over all points from all levels for each head and query, then the flat
        # axis is split back into [level, point] for the sampling loop below.
        weights = self.attention_weights(query).view(
            batch_size, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        weights = F.softmax(weights, -1)  # Global softmax
        # reshape for per-level use
        weights = weights.view(
            batch_size, num_queries, self.num_heads, self.num_levels, self.num_points
        )

        # Add singleton axes so broadcasting lines up the tensors as:
        # [batch, query, head, level, point, xy]. The reference point is shared
        # by all heads and points; each head then learns its own local pattern of
        # offsets around that anchor.
        reference_points = reference_points[:, :, None, :, None, :]
        # spatial_shapes is [H, W], but coordinates are [X, Y] = [W, H]
        # So swap them: [H, W] → [W, H]
        offset_normalizer = torch.stack(
            [
                spatial_shapes[..., 1],  # W (for X coordinate)
                spatial_shapes[..., 0],  # H (for Y coordinate)
            ],
            dim=-1,
        )
        offset_normalizer = offset_normalizer[None, None, None, :, None, :]
        sampling_locations = reference_points + offsets / offset_normalizer

        if (
            sampling_locations.shape[0] != query.shape[0]
            or sampling_locations.shape[1] != query.shape[1]
        ):
            raise ValueError(
                "sampling_locations and query must have the same batch size and number of queries"
            )
        if sampling_locations.shape[2] != self.num_heads:
            raise ValueError(
                "sampling_locations and query must have the same number of heads"
            )
        if sampling_locations.shape[4] != self.num_points:
            raise ValueError(
                "sampling_locations and query must have the same number of sampling points"
            )

        # Each feature level was flattened earlier. This loop restores one level
        # at a time to an image-like [batch, channel, height, width] tensor because
        # grid_sample operates on spatial feature maps, not token sequences.
        level_outputs: list[torch.Tensor] = []
        for lvl, (height, width) in enumerate(spatial_shapes):
            start_idx = level_start_index[lvl]
            end_idx = start_idx + height * width
            # Select exactly this level's tokens: [B, H*W, C].
            feat = value[:, start_idx:end_idx, :]
            # Undo flattening: [B, H*W, C] -> [B, H, W, C].
            feat = feat.reshape(batch_size, height, width, self.hidden_dim)
            # PyTorch convolution/grid_sample layout is channel-first:
            # [B, H, W, C] -> [B, C, H, W].
            feat = feat.permute(0, 3, 1, 2)

            # Split the model dimension into attention heads:
            # [B, C, H, W] -> [B, heads, head_dim, H, W].
            feat = feat.reshape(
                batch_size, self.num_heads, self.head_dim, height, width
            )
            # Keep this level's locations and merge batch, query, and head so one
            # grid row corresponds to one (image, query, head) combination:
            # [B, queries, heads, points, xy] -> [B*queries*heads, points, xy].
            loc = sampling_locations[:, :, :, lvl, :, :]
            loc = loc.reshape(
                batch_size * num_queries * self.num_heads, self.num_points, 2
            )

            grid_x = 2.0 * loc[..., 0] - 1.0
            grid_y = 2.0 * loc[..., 1] - 1.0
            # grid_sample expects normalized coordinates in [-1, 1], not [0, 1].
            # The singleton dimension is the output height; points become its
            # output width, giving [B*queries*heads, 1, points, 2].
            grid = torch.stack((grid_x, grid_y), dim=-1)[:, None, :, :]

            # Repeat each head feature map for every query. Permute to [B, query,
            # head, ...] so this order matches the flattened sampling locations.
            feat = feat.unsqueeze(2).expand(
                batch_size, self.num_heads, num_queries, self.head_dim, height, width
            )
            feat = feat.permute(0, 2, 1, 3, 4, 5)
            feat = feat.reshape(
                batch_size * num_queries * self.num_heads, self.head_dim, height, width
            )

            sampled_feat = F.grid_sample(
                feat,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            # grid_sample returns [combined, head_dim, 1, points]. Remove the
            # artificial height and put points before channels:
            # [combined, D, 1, points] -> [combined, points, D].
            sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)
            # Restore semantic axes for weighted aggregation:
            # [B*queries*heads, points, D] -> [B, queries, heads, points, D].
            sampled_feat = sampled_feat.reshape(
                batch_size, num_queries, self.num_heads, self.num_points, self.head_dim
            )

            level_weights = weights[:, :, :, lvl, :]
            # Broadcast each scalar point weight across its head dimension, then
            # sum the sampled points to get one vector per head.
            weighted = sampled_feat * level_weights.unsqueeze(-1)
            out_level = weighted.sum(dim=3)
            level_outputs.append(out_level)

        # Every level now has [B, queries, heads, head_dim]. Sum levels, then
        # merge heads back into the model dimension [heads, head_dim] -> C.
        output = torch.stack(level_outputs, dim=0).sum(dim=0)
        output = output.reshape(batch_size, num_queries, self.hidden_dim)
        output = self.output_proj(output)
        return output
