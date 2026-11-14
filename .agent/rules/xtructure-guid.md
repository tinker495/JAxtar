---
trigger: model_decision
description: Detailed guidelines for using the Xtructure library (xnp, Xtructurable)
---

# Xtructure Usage Guidelines

Ensure all JAX-compatible data structures and operations in this project adhere to the following rules.

## 1. Data Structure Definitions

### Mandatory: Inherit from `Xtructurable`
All structured data classes intended for use with JAX (Pytrees) **MUST** inherit directly from `Xtructurable` (or use `@base_dataclass` for top-level containers). Put `validate` and `bitpack` options in the class header. Share behavior through plain Python mixins or compose nested fields instead of inheriting from an already transformed data class.
Do not use standard Python `@dataclass` or `chex.dataclass` for these structures.

### Field Definitions
Use `FieldDescriptor` factory methods to strictly define fields. This is critical for JIT compatibility.

*   **Tensor Fields**: Use `FieldDescriptor.tensor(dtype=..., shape=...)`
    *   `dtype`: Must be a valid JAX dtype (e.g., `jnp.int32`, `jnp.float32`).
    *   `shape`: Tuple defining the *intrinsic* shape (excluding batch dims). Use `()` for scalars wrapped in arrays.
*   **Scalar Fields**: Use `FieldDescriptor.scalar(dtype=...)`
*   **Nested Structures**: A field's `dtype` can be another `Xtructurable` subclass.

**Example:**
```python
from xtructure import FieldDescriptor, Xtructurable


class SearchNode(Xtructurable):
    state: FieldDescriptor.scalar(dtype=Puzzle.State)  # Nested structure
    cost: FieldDescriptor.scalar(dtype=jnp.float32)
    history: FieldDescriptor.tensor(dtype=jnp.int32, shape=(10,))
```

## 2. Operations with `xnp` (Xtructure NumPy)

### Mandatory: Use `xnp` for Structure Manipulation
Any operation involving `Xtructurable` objects (splitting, stacking, reshaping, gathering) **MUST** use the `xnp` module. Do **NOT** use `jax.tree_util.tree_map` manually for standard array manipulations exposed by `xnp`.

### Key Operations Mapping
| Operation | Standard NumPy/JAX | Xtructure `xnp` |
| :--- | :--- | :--- |
| **Indexing/Slicing** | `arr[idx]` / `take` | `xnp.take(struct, idx, axis)` |
| **Gather (along axis)** | `take_along_axis` | `xnp.take_along_axis(struct, idx, axis)` |
| **Stacking** | `stack` | `xnp.stack([s1, s2], axis)` |
| **Concatenation** | `concatenate` | `xnp.concatenate([s1, s2], axis)` |
| **Reshaping** | `reshape` | `xnp.reshape(struct, new_shape)` |
| **Broadcasting** | `expand_dims` | `xnp.expand_dims(struct, axis)` |

## 3. High-Performance Patterns

### Deduplication
Use `xnp.unique_mask` to filter duplicate states within a batch. This is JIT-compatible and highly optimized.

```python
# Create a mask where True = keep this element (it is the first/best unique instance)
# key: cost array (min cost is kept)
# filled: valid mask (ignore invalid slots)
unique_mask = xnp.unique_mask(batch_states, key=batch_costs, filled=batch_valid_mask)
```

### Filtering & Pruning
To filter a batch (e.g., pruning nodes in beam search):
1.  Calculate scores/costs and mask invalid candidates to `jnp.inf` before calling `jax.lax.top_k`.
2.  Select the top indices via `jax.lax.top_k(-safe_scores, k=beam_width)` and build a `keep_mask` so padded slots stay masked when fewer than `beam_width` valid candidates exist.
3.  Use `xnp.take` to extract the subset and apply `keep_mask` to the selected fields.

```python
# 1. Mask invalid slots to +inf before top-k selection
safe_scores = jnp.where(valid_mask, scores, jnp.inf)
_, top_indices = jax.lax.top_k(-safe_scores, k=beam_width)

# 2. Extract structure subset
next_beam = xnp.take(current_beam_batch, top_indices, axis=0)

# 3. Apply keep_mask when valid candidates < beam_width
selected_scores = safe_scores[top_indices]
keep_mask = jnp.isfinite(selected_scores)
next_valid = jnp.logical_and(keep_mask, valid_mask[top_indices])
next_scores = jnp.where(next_valid, selected_scores, jnp.inf)
```

## 4. Documentation References
For detailed specs, refer to:
- Official Web Docs: `https://tinker495.github.io/Xtructure/`
- Parsed Specs (Internal): `xtructure_web_docs.md`
- Internal Report: `xtructure_internal_report.md`

## 5. ID-Stars Alignment Notes (id_astar.py / id_qstar.py)

### Intentional Algorithmic Differences

| Aspect | ID-A* | ID-Q* | Rationale |
| :--- | :--- | :--- | :--- |
| **f-value source** | Computed on **child states** (expensive heuristic) | Derived from **parent Q-values** (cheap) | Q* leverages parent Q; A* must evaluate children |
| **Early f > bound pruning** | After heuristic eval (in `_expand_step`) | Before unique_mask (in `inner_body`, lines 468-483) | Q* can prune early; A* needs h first |
| **Heuristic caching** | Uses `ht_dist` to cache h | No Q-value caching | Heuristic eval is expensive; Q is cheap per-action |
| **next_bound update location** | In `_expand_step` (lines 638-643) | In `inner_body` before `_expand_step` (lines 473-481) | Both update from pruned f-values; timing differs due to f-value availability |

### Shared Patterns (MUST align)

- **State indexing**: Always use `xnp.take(states, idx[jnp.newaxis], axis=0)` not `states[idx]`
- **Solved-state extraction**: Use `xnp.take(...)` pattern consistently
- **Deduplication**: `xnp.unique_mask` → `hashtable.parallel_insert` → `_apply_non_backtracking`
- **action_history depth clamping**: Use `safe_depth = jnp.minimum(flat_depth_flat, max_path_len - 1)` before `.at[].set()` to prevent out-of-bounds writes
- **ACTION_PAD for ID* action_history**: Use `jnp.iinfo(ACTION_DTYPE).max` and keep action_history arrays in `ACTION_DTYPE`

## 6. Path Reconstruction Reference (stars/ / beamsearch/)

### stars/common.py
- Uses **parent pointer reconstruction**: `Parent(hashidx, action)` per node.
- `_reconstruct_path_arrays()` backtracks via `parent_indices` (no action_history propagation).
- Reliability comes from updating parents only on valid/optimal insertions.

### beamsearch/search_base.py
- Uses **trace_parent / trace_action** arrays and `_reconstruct_trace_indices()`.
- Updates traces only when candidates are accepted into the beam.
- No history propagation across invalid rows.

### Implication for ID*
- `IDSearchResult` (`JAxtar/id_stars/search_base.py`) already uses a parent-pointer arena (`trace_parent` and `trace_action`) and `reconstruct_solution_actions()` to reconstruct the solution path (splicing the initial frontier's `action_history` prefix once via `splice_frontier_trace()`).
- Because ID* already uses `trace_parent` and `trace_action` for path reconstruction, no additional implementation changes are required.
