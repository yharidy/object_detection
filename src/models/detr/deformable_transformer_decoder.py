import torch
from torch import nn

from .multi_scale_deformable_attention import MultiScaleDeformableAttention


class DeformableTransformerDecoderLayer(nn.Module):
    """Run query self-attention, deformable cross-attention, and an FFN.

    Args:
        hidden_dim: The dimension of the hidden representations.
        num_heads: The number of attention heads.
        num_levels: The number of feature map levels.
        num_points: The number of sampling points per attention head.
        ffn_dim: The dimension of the feed forward network.
        dropout: The dropout rate.
        activation: The activation function to use in the feed forward network.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        """Initialize the decoder attention and feed-forward sublayers.

        Args:
            hidden_dim: Transformer feature dimension.
            num_heads: Number of self-attention and deformable-attention heads.
            num_levels: Number of encoder feature levels.
            num_points: Number of deformable samples per head and level.
            ffn_dim: Hidden dimension of the feed-forward network.
            dropout: Dropout probability used in residual branches.
            activation: FFN activation, either ``"relu"`` or ``"gelu"``.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.dropout = dropout
        self.ffn_dim = ffn_dim
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")
        # self attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # cross attention layer
        self.cross_attention = MultiScaleDeformableAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        # feed forward network
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout_self_attention = nn.Dropout(dropout)
        self.dropout_cross_attention = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_self_attention = nn.LayerNorm(hidden_dim)
        self.norm_cross_attention = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        encoder_memory: torch.Tensor,
        queries: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        input_padding_mask: torch.Tensor | None = None,
    ):
        """Apply one decoder layer to object queries and encoder memory.

        Args:
            encoder_memory: The memory from the encoder of shape [B, S, C].
            queries: The query tensor of shape [B, n_queries, C].
            reference_points: The reference points for the queries of shape [B, n_queries, n_levels, 2].
            spatial_shapes: The spatial shapes of the feature maps of shape [num_levels, 2].
            level_start_index: The start index of each feature map level of shape [num_levels].
            input_padding_mask: Optional boolean mask for encoder memory with
                shape ``[B, S]``.

        Returns:
            output: Tensor of shape [B, n_queries, C] after passing through the decoder layer.
        """
        if input_padding_mask is None:
            input_padding_mask = torch.zeros(
                encoder_memory.shape[0],
                encoder_memory.shape[1],
                dtype=torch.bool,
                device=encoder_memory.device,
            )

        # Self-attention on queries
        self_attention_output = self.self_attention(queries, queries, queries)[0]
        self_attention_output = self.norm_self_attention(
            queries + self.dropout_self_attention(self_attention_output)
        )

        # Cross-attention with encoder memory
        cross_attention_output = self.cross_attention(
            query=self_attention_output,
            input_flatten=encoder_memory,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            input_padding_mask=input_padding_mask,
        )
        cross_attention_output = self.norm_cross_attention(
            self_attention_output + self.dropout_cross_attention(cross_attention_output)
        )

        # FFN
        ffn_output = self.linear1(cross_attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.linear2(ffn_output)
        output = self.norm_ffn(cross_attention_output + self.dropout_ffn(ffn_output))
        return output


class DeformableTransformerDecoder(nn.Module):
    """Decode learned object queries into query features, boxes, and logits.

    Args:
        n_queries: The number of queries.
        query_dim: The dimension of the query embeddings.
        num_layers: The number of decoder layers and box-refinement heads.
        num_heads: Number of attention heads.
        num_levels: Number of encoder feature levels.
        dropout: Dropout probability in decoder layers.
        num_points: Number of deformable samples per head and level.
        ffn_dim: Hidden dimension of each decoder FFN.
        num_classes: Number of foreground classes. One additional output
            class is reserved for no-object/background.
        activation: Decoder FFN activation, either ``"relu"`` or ``"gelu"``.
    """

    def __init__(
        self,
        n_queries: int,
        query_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        num_levels: int = 4,
        dropout: float = 0.1,
        num_points: int = 4,
        ffn_dim: int = 2048,
        num_classes: int = 91,
        activation: str = "relu",
    ):
        """Initialize queries, reference-point projection, and prediction heads."""
        super().__init__()
        self.n_queries = n_queries
        self.query_dim = query_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.query_embed = nn.Embedding(n_queries, query_dim)
        self.reference_points = nn.Linear(query_dim, 2)
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.num_points = num_points
        self.num_classes = num_classes
        # decoder layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_layers.append(
                DeformableTransformerDecoderLayer(
                    hidden_dim=query_dim,
                    num_heads=num_heads,
                    num_levels=num_levels,
                    num_points=num_points,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    activation=activation,
                )
            )
        # box refinement heads
        self.box_refinements = nn.ModuleList()
        for _ in range(num_layers):
            self.box_refinements.append(nn.Linear(query_dim, 4))

        # classification head
        self.classification_head = nn.Linear(
            query_dim, self.num_classes + 1
        )  # +1 for the "no object" class

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Map probabilities in ``[0, 1]`` to logits with clamping."""
        x = x.clamp(min=eps, max=1 - eps)
        return torch.log(x / (1 - x))

    def forward(
        self,
        encoder_output: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        input_padding_mask: torch.Tensor | None = None,
    ):
        """Iteratively decode queries and refine their reference points.

        Args:
            encoder_output: The output from the encoder of shape [B, S, C].
            spatial_shapes: The spatial shapes of the feature maps of shape [num_levels, 2].
            level_start_index: The start index of each feature map level of shape [num_levels].
            input_padding_mask: The padding mask for the encoder output of shape [B, S] or None.

        Returns:
            A tuple ``(queries, boxes, class_logits)``. Queries have shape
            ``[B, num_queries, query_dim]``; boxes have normalized ``cxcywh``
            coordinates with shape ``[B, num_queries, 4]``; and class logits
            have shape ``[B, num_queries, num_classes + 1]``.
        """
        if input_padding_mask is None:
            input_padding_mask = torch.zeros(
                encoder_output.size(0),
                encoder_output.size(1),
                dtype=torch.bool,
                device=encoder_output.device,
            )
        queries = self.query_embed.weight.unsqueeze(0).expand(
            encoder_output.size(0), -1, -1
        )  # [batch_size, n_queries, query_dim]
        reference_points = (
            self.reference_points(queries)
            .sigmoid()
            .unsqueeze(2)
            .expand(-1, -1, self.num_levels, -1)
        )  # [B, n_queries, n_levels, 2]

        for decoder, box_refinement in zip(self.decoder_layers, self.box_refinements):
            queries = decoder(
                encoder_memory=encoder_output,
                queries=queries,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                input_padding_mask=input_padding_mask,
            )
            # Box refinement: only x and y are updated between layers
            box_offsets = box_refinement(
                queries
            )  # [B, n_queries, 4], d_x, d_y, d_w, d_h
            box_offsets = box_offsets.unsqueeze(2).expand(
                -1, -1, self.num_levels, -1
            )  # [B, n_queries, n_levels, 4]
            reference_points = (
                self._inverse_sigmoid(reference_points) + box_offsets[..., :2]
            ).sigmoid()  # [B, n_queries, n_levels, 2]

        boxes = torch.cat((reference_points, box_offsets[..., 2:].sigmoid()), dim=-1)[
            :, :, 0, :
        ]  # [B, n_queries, 4]
        class_logits = self.classification_head(queries)  # [B, n_queries, num_classes]

        return queries, boxes, class_logits
