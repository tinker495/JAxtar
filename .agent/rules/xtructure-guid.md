---
trigger: model_decision
description: Xtructure usage invariants (Xtructurable, xnp, JIT safety) in JAxtar
---

# Xtructure Usage Guidelines

## Data Structure Definitions
- JAX pytrees **must** inherit directly from `Xtructurable` (or use `@base_dataclass` for top-level containers) with `validate`/`bitpack` options in the class header. Share behavior through plain Python mixins or nested fields instead of subclassing an already-transformed dataclass; do not use standard `@dataclass` or `chex.dataclass`.
- Use `FieldDescriptor` annotations so JIT and Pyrefly share one layout schema (tensor `shape` excludes batch dimensions):

```python
from typing import Annotated

import jax
import jax.numpy as jnp
from puxle import Puzzle
from xtructure import FieldDescriptor, Xtructurable


class SearchNode(Xtructurable):
    state: Annotated[Puzzle.State, FieldDescriptor.scalar(dtype=Puzzle.State)]
    cost: Annotated[jax.Array, FieldDescriptor.scalar(dtype=jnp.float32)]
    history: Annotated[jax.Array, FieldDescriptor.tensor(dtype=jnp.int32, shape=(10,))]
```

## Operations and Indexing Invariants
- **Use `xnp` for structure ops**: Never replace `xnp.take`, `xnp.stack`, `xnp.concatenate`, `xnp.reshape`, or `xnp.unique_mask` with manual `jax.tree_util.tree_map`, which bypasses packed/bitpacked Xtructure layout metadata.
- **Batch-1 state extraction**: Use `xnp.take(states, idx[jnp.newaxis], axis=0)` rather than `states[idx]` when extracting a single solved state so the batch-1 axis is preserved for `Xtructurable` leaves.
- **Top-k pruning**: Mask invalid slots to `+inf` before `jax.lax.top_k(-safe_scores, k=width)`, then derive `keep_mask = jnp.isfinite(safe_scores[top_idx]) & valid_mask[top_idx]` so padded slots stay masked when fewer than `width` valid candidates exist.
- **ID* `action_history` bounds and padding**: Clamp writes with `safe_depth = jnp.minimum(flat_depth_flat, max_path_len - 1)` before `.at[..., safe_depth].set(...)` to prevent out-of-bounds updates, keep `action_history` arrays in `ACTION_DTYPE`, and pad unused slots with `ACTION_PAD = jnp.iinfo(ACTION_DTYPE).max`.

Reference: `https://tinker495.github.io/Xtructure/`
