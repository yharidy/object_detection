# Deformable DETR

This folder contains a focused, modular implementation of Deformable DETR for a portfolio project and learning codebase. The goal is to capture the main ideas of the architecture in a way that is readable, testable, and easy to extend.

Deformable DETR is a transformer-based object detector that improves on the original DETR in two important ways:

- it reduces the cost of encoder-decoder attention by sampling only a small number of locations per query
- it works well with multi-scale feature maps, which is important for detecting objects at different sizes

The architecture keeps the high-level DETR pattern:

- a CNN backbone extracts image features
- a transformer encoder processes those features
- a transformer decoder produces object queries
- a detection head predicts boxes and classes

But Deformable DETR uses a more efficient attention mechanism called multi-scale deformable attention.

## High-level architecture

```text
Image
  |
  v
Backbone (ResNet-like CNN)
  |
  +--> Feature map level 1
  +--> Feature map level 2
  +--> Feature map level 3
  +--> Feature map level 4
  |
  v
Flatten + positional encoding
  |
  v
Transformer encoder
  |
  v
Decoder queries + reference points
  |
  v
Multi-scale deformable attention
  |
  v
Detection heads
  |
  +--> class logits
  +--> box predictions
```

## Why Deformable DETR?

Standard DETR uses dense attention over all image tokens. That is expensive, especially with high-resolution feature maps. Deformable DETR instead learns a small set of sampling locations around each reference point and only attends to those points.

This makes three big differences:

1. Lower compute cost
2. Better handling of multi-scale feature maps
3. Faster convergence than vanilla DETR

## Core idea: deformable attention

Instead of asking a query to attend to every spatial location in a feature map, we do this:

- choose a reference point for each query
- predict offsets around that point
- sample a small number of nearby positions
- compute attention weights over those sampled points
- aggregate the sampled features

This is the key idea behind the implementation in this folder.

```text
Query
  |
  v
Reference point + learned offsets
  |
  +--> sample nearby positions from each feature level
  |
  v
Attention weights
  |
  v
Weighted sum of sampled features
```

In practice, this means that each query is not looking at the entire feature map. It is looking at a sparse set of informative locations.

## Multi-scale feature maps

Objects in images appear at many scales. A single feature map is often not enough to detect both large and small objects well.

Deformable DETR handles this by building attention over multiple feature levels, each with different spatial resolution.

Example:

```text
Level 1:  H/8  x  W/8   (fine detail)
Level 2:  H/16 x  W/16  (medium detail)
Level 3:  H/32 x  W/32  (coarse context)
Level 4:  H/64 x  W/64  (global context)
```

The model can then sample from whichever level is most appropriate for a given object reference point.

A feature map at a low resolution contains high-level semantics, while a high-resolution map contains fine-grained spatial detail. Deformable attention combines both.

## Positional encoding

Transformers are permutation-sensitive, so they need positional information. In this implementation, that is done with a sinusoidal 2D positional embedding.

The positional encoding is computed for each spatial location in the feature map and added to the feature representation. This helps the model understand where a feature comes from in the image.

```text
Feature map [B, C, H, W]
   +
Positional embedding [B, C, H, W]
   =
Encoded feature map [B, C, H, W]
```

This is implemented in the module `position_embedding.py`.

## Flattened multi-scale features

Before deformable attention, the multi-scale feature maps are flattened into a sequence form that the attention layer can consume.

Each feature level contributes a set of tokens:

- level 1: H1 * W1 tokens
- level 2: H2 * W2 tokens
- level 3: H3 * W3 tokens
- level 4: H4 * W4 tokens

These are concatenated into one long flattened tensor, and the code also keeps:

- spatial shapes per level
- the start index for each level in the flattened sequence
- padding masks

This is what lets the attention layer know which spatial region belongs to each level.

```text
Level 1 tokens  | Level 2 tokens | Level 3 tokens | Level 4 tokens
     \n          \n                 \n                       \n        [B, S, C]
```

This is handled by `flatten_multi_scale_features` in `feature_utils.py`.

## The attention math

For each query, Deformable DETR predicts:

- a set of sampling offsets for each feature level
- a set of attention weights for each sampled point

Given a feature level $l$, query $q$, and sample point $k$, the model computes a sampling location as:

$$
 p_{qlk} = r_{ql} + \Delta p_{qlk}
$$

where:

- $r_{ql}$ is the reference point
- $\Delta p_{qlk}$ is the learned offset

Then it samples nearby features from that location and aggregates them using learned weights:

$$
\text{out}_{q} = \sum_{l=1}^{L} \sum_{k=1}^{K} A_{qlk} \cdot \text{Sample}(F_l, p_{qlk})
$$

where:

- $L$ is the number of feature levels
- $K$ is the number of sampled points per level
- $A_{qlk}$ is the attention weight for that sample

This is the core reason the method is efficient: it never evaluates all positions, only a few sampled sites.

## The encoder-decoder perspective

The full Deformable DETR pipeline follows the DETR pattern:

```text
CNN backbone
   -> multi-scale features
   -> encoder
   -> decoder
   -> prediction heads
```

### Encoder

The encoder processes flattened multi-scale image features using self-attention, but in practice the deformable attention block is more efficient than full dense attention. The encoder learns contextual relationships between image regions.

### Decoder

The decoder uses object queries. For each query, it learns where to look in the image and which sampled points matter most. The attention is conditioned on the query content and the reference point.

### Detection heads

After decoder output, the model predicts:

- class logits for each object query
- bounding box parameters, usually center/size or box corners

These are then converted into final detections.

## What is implemented in this folder

This folder focuses on the foundational pieces that are most educational and easiest to reason about:

- `position_embedding.py`
  - 2D sinusoidal positional embeddings
- `backbone.py`
  - a ResNet-style multi-scale feature extractor
- `feature_utils.py`
  - flattening and metadata generation for multi-scale features
- `multi_scale_deformable_attention.py`
  - deformable attention over multiple levels

Those pieces are exactly the parts that make the architecture different from a standard transformer and are the most useful conceptual building blocks for learning the design.

## File map

```text
src/models/detr/
├── __init__.py
├── README.md
├── backbone.py
├── feature_utils.py
├── multi_scale_deformable_attention.py
├── position_embedding.py
└── ...
```

The package is intentionally split by responsibility so it is easier to understand and test each concept in isolation.

## Why this structure is helpful

A good portfolio project should not only work, but also be readable. Splitting the code by concept keeps the architecture understandable:

- positional encoding is isolated
- backbone logic is isolated
- flattening utilities are isolated
- attention is isolated
- tests can target each piece independently

This makes debugging much easier and helps explain the model in interviews and project walkthroughs.

## Practical training flow

A typical training pass looks like this:

1. forward image through the CNN backbone
2. produce multi-scale feature maps
3. add positional encodings
4. flatten each level into a token sequence
5. pass flattened features into deformable attention
6. decode object queries
7. predict boxes and classes
8. compute detection loss

That is the full training lifecycle the architecture is designed around.

## Important implementation notes

A few details matter in Deformable DETR and are often the source of confusion:

- feature maps from different levels are not all the same size
- each query samples a small number of offsets per level
- attention weights are normalized over the sampled points
- the feature maps must be flattened and reconstructed carefully so that each level is sampled in the right spatial region
- the query dimension and head dimension must stay consistent across the sampling process

This implementation reflects those constraints directly.

## Summary

Deformable DETR is a more efficient and practical version of DETR for object detection. It keeps the transformer-based end-to-end detection philosophy while replacing dense global attention with sparse, learned sampling around reference points.

The key idea is simple but powerful:

- detect where each object query should look
- sample only a few informative points around that location
- aggregate them with attention weights
- do this across multiple feature scales

That is what makes the model both more efficient and more useful for real detection problems.

## Suggested next reading

For a deeper understanding, the most useful next topics are:

- DETR vs. Deformable DETR
- multi-head attention and head splitting
- reference points and sampling offsets
- bipartite matching loss in DETR
- deformable attention from the original paper

These are the next conceptual building blocks if you want to turn this codebase into a stronger portfolio-level object detection project.
