"""
Visualization constants for consistent plotting across the codebase.

This module defines standard figure sizes, color palettes, and styling parameters
to ensure visual consistency across all plotting functions.
"""

import seaborn as sns

# ============================================================================
# Figure Sizes
# ============================================================================

# Standard figure sizes for different plot types
DEFAULT_FIGSIZE = (10, 6)  # Basic plots (distributions, boxplots)
WIDE_FIGSIZE = (16, 6)  # Wide horizontal layouts (benchmark comparisons)
LARGE_FIGSIZE = (12, 12)  # Large square plots (heuristic accuracy)
COMPARISON_FIGSIZE = (12, 8)  # Comparison scatter plots
TREE_FIGSIZE = (14, 10)  # Search tree visualizations
TALL_FIGSIZE = (12, 18)  # Tall vertical stacks (expansion distributions)

# ============================================================================
# Color Palettes
# ============================================================================

# Primary colors for consistent theming
COLOR_PRIMARY = "#4C72B0"  # Blue - primary data
COLOR_SUCCESS = "#2ca02c"  # Green - success/optimal
COLOR_ERROR = "#d62728"  # Red - error/suboptimal
COLOR_WARNING = "#DD8452"  # Orange - warning/mean
COLOR_MEDIAN = "#55A868"  # Green - median values
COLOR_NEUTRAL = "#555555"  # Gray - neutral/reference

# Seaborn palettes
PALETTE_TAB10 = sns.color_palette("tab10", 10)  # Standard categorical palette
PALETTE_VIRIDIS = "viridis"  # Sequential colormap for heatmaps

# Named color mappings for semantic use
COLORS = {
    "primary": COLOR_PRIMARY,
    "success": COLOR_SUCCESS,
    "error": COLOR_ERROR,
    "warning": COLOR_WARNING,
    "median": COLOR_MEDIAN,
    "neutral": COLOR_NEUTRAL,
}

# ============================================================================
# Line and Marker Styles
# ============================================================================

# Grid styling
GRID_LINESTYLE = "--"
GRID_ALPHA = 0.5
GRID_LINEWIDTH = 0.7

# Line styling
DEFAULT_LINEWIDTH = 1.0
THICK_LINEWIDTH = 2.0
THIN_LINEWIDTH = 0.5

# Scatter plot styling
SCATTER_ALPHA_LOW = 0.1  # Very transparent (background points)
SCATTER_ALPHA_MED = 0.3  # Medium transparency (standard scatter)
SCATTER_ALPHA_HIGH = 0.7  # High opacity (highlighted points)

# Marker sizes
MARKER_SIZE_SMALL = 1  # Tiny markers for dense plots
MARKER_SIZE_DEFAULT = 5  # Standard scatter points
MARKER_SIZE_MEDIUM = 20  # Highlighted points
MARKER_SIZE_LARGE = 100  # Special markers (start/goal)
MARKER_SIZE_XLARGE = 120  # Cluster centers

# ============================================================================
# Plot-Specific Defaults
# ============================================================================

# Expansion distribution plots
EXPANSION_SCATTER_MAX_POINTS = 5000
EXPANSION_COLORS = {
    "cost": "blue",
    "heuristic": "green",
    "key": "red",
}

# Search tree visualization
TREE_MAX_POINTS = 500000
TREE_EDGE_COLOR = "gray"
TREE_EDGE_ALPHA = 0.3
TREE_EDGE_LINEWIDTH = 1.0
TREE_PATH_COLOR = "red"
TREE_PATH_LINEWIDTH = 2.0
TREE_START_COLOR = "lime"
TREE_GOAL_COLOR = "gold"

# Comparison plots
COMPARISON_SCATTER_MAX_POINTS = 2000
COMPARISON_ELLIPSE_ALPHA = 0.5
COMPARISON_ELLIPSE_LINEWIDTH = 2.0

# Heuristic accuracy plots
HEURISTIC_DIAGONAL_COLOR = "green"
HEURISTIC_DIAGONAL_LINESTYLE = "--"
HEURISTIC_DIAGONAL_ALPHA = 0.75

# Benchmark comparison
BENCHMARK_MATCH_COLOR = COLOR_SUCCESS
BENCHMARK_MISMATCH_COLOR = COLOR_ERROR
BENCHMARK_DIAGONAL_COLOR = COLOR_NEUTRAL
BENCHMARK_DIAGONAL_LINESTYLE = "--"
BENCHMARK_DIAGONAL_LINEWIDTH = 1.0
