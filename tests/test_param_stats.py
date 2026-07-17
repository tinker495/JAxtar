import numpy as np

from helpers.param_stats import jax_param_stats

KERNEL = (100, 200)  # 20_000 params
BIAS = (200,)  # 200 params


def _base_tree(dtype):
    return {
        "params": {
            "dense": {
                "kernel": np.zeros(KERNEL, dtype=dtype),
                "bias": np.zeros(BIAS, dtype=dtype),
            }
        }
    }


def test_plain_tree_counts_real_bytes():
    stats = jax_param_stats(_base_tree(np.float32))
    assert stats["total_params"] == "20.2K"
    assert stats["total_bytes"] == "80.8K"
    assert "(" not in stats["total_size"]


def test_unconverted_tree_with_aqt_cfg_estimates_kernel_bytes():
    stats = jax_param_stats(_base_tree(np.float32), aqt_cfg="int8")
    # kernel at 1B + bias at real 4B, original in parentheses
    assert stats["total_params"] == "20.2K"
    assert stats["total_bytes"] == "20.8K (80.8K)"


def test_frozen_aqt_tree_not_double_counted():
    tree = _base_tree(np.float16)  # 2B/elem, same itemsize as bfloat16
    tree["aqt"] = {
        "dense": {
            "frozen_kernel": np.zeros(KERNEL, dtype=np.int8),
            "scale": np.zeros((1, 200), dtype=np.float32),
        }
    }
    stats = jax_param_stats(tree, aqt_cfg="int8")
    # logical params: 20.2K, NOT 40.4K (int8 copies are not extra parameters)
    assert stats["total_params"] == "20.2K"
    # serving = bias 200*2B + qvalue 20_000*1B + scale 200*4B = 21.2K
    # original = 20_200 * 2B = 40.4K
    assert stats["total_bytes"] == "21.2K (40.4K)"


def test_frozen_aqt_tree_detected_without_aqt_cfg():
    tree = _base_tree(np.float32)
    tree["aqt"] = {"dense": {"frozen_kernel": np.zeros(KERNEL, dtype=np.int8)}}
    stats = jax_param_stats(tree)
    assert stats["total_params"] == "20.2K"
    # serving = bias 200*4B + qvalue 20_000*1B = 20.8K, original 80.8K
    assert stats["total_bytes"] == "20.8K (80.8K)"
