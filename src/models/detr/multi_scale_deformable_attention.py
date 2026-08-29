"""Multi-scale deformable attention implementation."""

import torch
import torch.nn.functional as F
from torch import nn


class MultiScaleDeformableAttention(nn.Module):
    """Multi-scale deformable attention over a list of feature maps."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
    ) -> None:
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

    def forward(
        self,
        query: torch.Tensor,
        input_flatten: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        input_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-scale deformable attention.

        Args:
            query: Query tensor with shape [B, Lq, hidden_dim].
            input_flatten: Flattened source feature tensor with shape [B, S, hidden_dim].
            reference_points: Reference locations for each level with shape
                [B, Lq, num_levels, 2].
            spatial_shapes: Level-wise feature map sizes with shape [num_levels, 2].
            level_start_index: First index for each level in the flattened sequence.
            input_padding_mask: Mask for padded source tokens with shape [B, S].

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

        batch_size, num_queries, _ = query.shape

        offsets = self.sampling_offsets(query).view(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        weights = self.attention_weights(query).view(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        reference_points = reference_points[:, :, None, :, None, :]
        normalizer = spatial_shapes.flip(dims=[1])[None, None, None, :, None, :]
        sampling_locations = reference_points + offsets / normalizer

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

        level_outputs: list[torch.Tensor] = []
        for lvl, (height, width) in enumerate(spatial_shapes):
            start_idx = level_start_index[lvl]
            end_idx = start_idx + height * width
            feat = input_flatten[:, start_idx:end_idx, :]
            feat = feat.reshape(batch_size, height, width, self.hidden_dim)
            feat = feat.permute(0, 3, 1, 2)

            feat = feat.reshape(
                batch_size, self.num_heads, self.head_dim, height, width
            )
            feat = feat.permute(0, 1, 2, 3, 4).reshape(
                batch_size * self.num_heads, self.head_dim, height, width
            )

            loc = sampling_locations[:, :, :, lvl, :, :]
            loc = loc.reshape(
                batch_size * num_queries * self.num_heads, self.num_points, 2
            )

            grid_x = 2.0 * loc[..., 0] - 1.0
            grid_y = 2.0 * loc[..., 1] - 1.0
            grid = torch.stack((grid_x, grid_y), dim=-1)[:, None, :, :]

            feat = feat.unsqueeze(1).expand(
                batch_size * self.num_heads, num_queries, self.head_dim, height, width
            )
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
            sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)
            sampled_feat = sampled_feat.reshape(
                batch_size, num_queries, self.num_heads, self.num_points, self.head_dim
            )

            level_weights = weights[:, :, :, lvl, :].softmax(dim=-1)
            weighted = sampled_feat * level_weights.unsqueeze(-1)
            out_level = weighted.sum(dim=3)
            level_outputs.append(out_level)

        output = torch.stack(level_outputs, dim=0).sum(dim=0)
        output = output.reshape(batch_size, num_queries, self.hidden_dim)
        return output
