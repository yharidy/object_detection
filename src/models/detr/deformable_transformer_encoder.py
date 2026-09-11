import torch
from torch import nn

from .multi_scale_deformable_attention import MultiScaleDeformableAttention


class DeformableTransformerEncoderLayer(nn.Module):
    """One deformable self-attention plus feed-forward encoder layer.

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
    ):
        """Initialize attention, normalization, dropout, and FFN modules.

        Args:
            hidden_dim: Transformer feature dimension.
            num_heads: Number of deformable-attention heads.
            num_levels: Number of multi-scale feature levels.
            num_points: Number of samples per head and level.
            dropout: Dropout probability used after attention and in the FFN.
            ffn_dim: Hidden dimension of the feed-forward network.
            activation: FFN activation, either ``"relu"`` or ``"gelu"``.
        """
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
        """Generate normalized pixel-center references for every source token.

        Args:
            spatial_shapes: Integer tensor containing ``[H_i, W_i]`` for each
                feature level, with shape ``[num_levels, 2]``.

        Returns:
            Tensor with shape ``[S, 2]`` containing normalized ``(x, y)``
            coordinates, where ``S = sum(H_i * W_i)``.
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
        """Apply deformable self-attention followed by the FFN.

        Args:
            input_flatten: Flattened source feature tensor [B, S, C].
            positional_encodings: Tensor containing the positional encodings [B, S, C].
            spatial_shapes: Tensor of level shapes ``[num_levels, 2]``.
            level_start_index: Starting token index of each level, shape
                ``[num_levels]``.
            input_padding_mask: Optional boolean mask ``[B, S]``.

        Returns:
            Encoded source features with shape ``[B, S, hidden_dim]``.
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
    """Stack multiple deformable transformer encoder layers.

    Args:
        hidden_dim: The number of expected features in the input.
        num_heads: The number of heads in the multihead attention models.
        num_levels: The number of feature levels.
        num_points: The number of sampling points per attention head.
        dropout: The dropout value.
        activation: Activation function used by each layer, ``"relu"`` or
            ``"gelu"``.
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
        """Construct the encoder layer stack."""
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
        """Run every encoder layer over the flattened multi-scale features.

        Args:
            input_flatten: Flattened source feature tensor [B, S, C].
            positional_encodings: Flattened positional encodings [B, S, C].
            spatial_shapes: Tensor of level shapes ``[num_levels, 2]``.
            level_start_index: Tensor of level start indices ``[num_levels]``.
            input_padding_mask: Optional boolean mask ``[B, S]``.

        Returns:
            Encoder memory with shape ``[B, S, hidden_dim]``.
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
