import torch
from torch import nn

from .multi_scale_deformable_attention import MultiScaleDeformableAttention


class DeformableTransformerEncoderLayer(nn.Module):
    """A single layer of the Deformable Transformer encoder.

    Args:
        hidden_dim: The number of expected features in the input.
        num_heads: The number of heads in the multihead attention models.
        num_levels: The number of feature levels.
        num_points: The number of sampling points per attention head.
        dropout: The dropout value.
        activation: The activation function of intermediate layer, relu or gelu.

    Returns:
        output: Tensor of shape [B, S, hidden_dim] after passing through the encoder layer.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        ffn_dim: int = 1024,
        activation: str = "relu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.ffn_dim = ffn_dim
        self.multi_scale_deformable_attention = MultiScaleDeformableAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.dropout_attention = nn.Dropout(dropout)
        self.norm_attention = nn.LayerNorm(hidden_dim)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def _get_reference_points(self, spatial_shapes: torch.Tensor) -> torch.Tensor:
        """Generate reference points for the Deformable Transformer encoder.

        Args:
            spatial_shapes: Tensor containing the shape of each feature map level [num_levels, 2].

        Returns:
            reference_points: Tensor containing the reference points for all feature map levels [S, 2], where S is the total number of spatial locations across all levels.
        """
        all_level_points = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            rows = torch.arange(0, H, dtype=torch.float32, device=spatial_shapes.device)
            cols = torch.arange(0, W, dtype=torch.float32, device=spatial_shapes.device)
            x = (cols + 0.5) / W
            y = (rows + 0.5) / H
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")  # [H, W]
            grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
            all_level_points.append(grid.reshape(H * W, 2))  # [H*W, 2]
        reference_points = torch.cat(all_level_points, dim=0)  # [sum(H*W), 2]
        return reference_points

    def forward(
        self,
        input_flatten: torch.Tensor,
        positional_encodings: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        input_padding_mask: torch.Tensor | None = None,
    ):
        """Forward pass of the Deformable Transformer encoder layer.

        Args:
            input_flatten: Flattened source feature tensor [B, S, C].
            positional_encodings: Tensor containing the positional encodings [B, S, C].
            spatial_shapes: Tensor containing the shape of each feature map level [num_levels, 2].
            level_start_index: Tensor containing the start index of each feature map level [num_levels].
            input_padding_mask: Tensor containing the padding mask [B, S] or None.
        """
        if input_flatten.dim() != 3:
            raise ValueError("input_flatten must have shape [B, S, C]")
        if input_flatten.shape != positional_encodings.shape:
            raise ValueError(
                f"Mismatch between input_flatten {input_flatten.shape} and positional_encodings {positional_encodings.shape}"
            )
        if spatial_shapes.shape != (self.num_levels, 2):
            raise ValueError(f"spatial_shapes must have shape [{self.num_levels}, 2]")
        if (
            input_padding_mask is not None
            and input_flatten.shape[0] != input_padding_mask.shape[0]
        ):
            raise ValueError(
                f"Mismatch between input_flatten batch size {input_flatten.shape[0]} and input_padding_mask batch size {input_padding_mask.shape[0]}"
            )
        # reference points
        reference_points = self._get_reference_points(spatial_shapes)  # [S, 2]
        if reference_points.shape[0] != input_flatten.shape[1]:
            raise ValueError(
                f"Mismatch between reference points ({reference_points.shape[0]}) and input flatten ({input_flatten.shape[1]})"
            )
        reference_points = reference_points[None, :, None, :].expand(
            input_flatten.shape[0], -1, self.num_levels, -1
        )  # [B, S, num_levels, 2])
        # padding mask
        if input_padding_mask is not None:
            input_padding_mask = input_padding_mask.flatten(1)  # [B, S]
            if input_padding_mask.shape != input_flatten.shape[:2]:
                raise ValueError(
                    "input_padding_mask must have shape [B, S] or flatten to [B, S]"
                )
        else:
            input_padding_mask = torch.zeros(
                input_flatten.shape[0],
                input_flatten.shape[1],
                dtype=torch.bool,
                device=input_flatten.device,
            )  # [B, S]
        # get positional encodings
        query = input_flatten + positional_encodings  # [B, S, C]
        # multi-scale deformable attention
        attention_output = self.multi_scale_deformable_attention(
            query=query,
            input_flatten=input_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            input_padding_mask=input_padding_mask,
        )  # [B, S, hidden_dim]
        # skip connection + normalize
        src = self.norm_attention(
            input_flatten + self.dropout_attention(attention_output)
        )  # [B, S, hidden_dim]
        # feed forward network
        ffn_output = self.linear2(
            self.activation(self.linear1(src))
        )  # [B, S, hidden_dim]
        # skip connection + normalize
        output = self.norm_ffn(src + self.dropout_ffn(ffn_output))  # [B, S, hidden_dim]
        return output


class DeformableTransformerEncoder(nn.Module):
    """Deformable Transformer Encoder consisting of multiple encoder layers.

    Args:
        hidden_dim: The number of expected features in the input.
        num_heads: The number of heads in the multihead attention models.
        num_levels: The number of feature levels.
        num_points: The number of sampling points per attention head.
        dropout: The dropout value.
        activation: The activation function of intermediate layer, relu or gelu.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        ffn_dim: int = 1024,
        activation: str = "relu",
        num_layers: int = 6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.dropout = dropout
        self.ffn_dim = ffn_dim
        self.activation = activation
        self.layers = nn.ModuleList(
            [
                DeformableTransformerEncoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_levels=num_levels,
                    num_points=num_points,
                    dropout=dropout,
                    ffn_dim=ffn_dim,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        input_flatten: torch.Tensor,
        positional_encodings: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        input_padding_mask: torch.Tensor | None = None,
    ):
        """Forward pass of the Deformable Transformer encoder.

        Args:
            input_flatten: Flattened source feature tensor [B, S, C].
            positional_encodings: Flattened positional encodings [B, S, C].
            spatial_shapes: Tensor containing the shape of each feature map level [num_levels, 2].
            level_start_index: Tensor containing the start index of each feature map level [num_levels].
            input_padding_mask: Tensor containing the padding mask [B, S] or None.
        """
        for layer in self.layers:
            input_flatten = layer(
                input_flatten=input_flatten,
                positional_encodings=positional_encodings,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                input_padding_mask=input_padding_mask,
            )
        return input_flatten
