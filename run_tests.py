#!/usr/bin/env python3
"""
run_tests.py — Automated Test Suite Runner for Gist Memory
"""

import sys
import time
from pathlib import Path

# Add src to Python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from tests.test_equivalence import (
    test_parallel_vs_streaming_equivalence,
    test_chunked_ssd_vs_full_sequence,
)
from tests.test_multihead import (
    test_multihead_forward_and_shapes,
    test_multihead_gradient_flow,
    test_multihead_streaming_equivalence,
)
from tests.test_decay import (
    test_multiscale_decay_initialization,
    test_data_dependent_decay,
)
from tests.test_state import (
    test_gist_state_diagnostics,
    test_gist_state_serialization,
)
from tests.test_hf_adapter import (
    test_zero_init_preservation,
    test_gist_cache_propagation,
)
from tests.test_fixes import (
    test_data_dependent_decay_exact_equivalence,
    test_attention_mask_left_padding_invariance,
    test_gist_cache_reorder_beams,
    test_state_multi_source_metadata_fusion,
)

TEST_CASES = [
    ("Numerical Equivalence: Parallel vs Streaming (Square Kernel)", lambda: test_parallel_vs_streaming_equivalence(kernel="square")),
    ("Numerical Equivalence: Parallel vs Streaming (ReLU2 Kernel)", lambda: test_parallel_vs_streaming_equivalence(kernel="relu2")),
    ("Numerical Equivalence: Parallel vs Streaming (ELU1 Kernel)", lambda: test_parallel_vs_streaming_equivalence(kernel="elu1")),
    ("Chunked SSD vs Full Sequence Equivalence", test_chunked_ssd_vs_full_sequence),
    ("MultiHead: Forward & Shape Validation", test_multihead_forward_and_shapes),
    ("MultiHead: Gradient Flow & Parameter Updates", test_multihead_gradient_flow),
    ("MultiHead: Streaming Step Equivalence", test_multihead_streaming_equivalence),
    ("Decay: MultiScale Progression & Positivity", test_multiscale_decay_initialization),
    ("Decay: Data-Dependent Contextual Decay", test_data_dependent_decay),
    ("Decay: Data-Dependent Exact Parallel vs Streaming Equivalence", test_data_dependent_decay_exact_equivalence),
    ("Mask: Attention Mask & Left-Padding Invariance", test_attention_mask_left_padding_invariance),
    ("State: Diagnostics, Energy & SVD Capacity", test_gist_state_diagnostics),
    ("State: Disk Save & Load Roundtrip", test_gist_state_serialization),
    ("State: Chained Multi-Source Metadata Retention", test_state_multi_source_metadata_fusion),
    ("Adapter: Strict Zero-Init Baseline Preservation", test_zero_init_preservation),
    ("Adapter: GistCache Step-by-Step Propagation", test_gist_cache_propagation),
    ("Adapter: GistCache Beam Search Reordering", test_gist_cache_reorder_beams),
]


def main():
    print("=" * 70)
    print("  GIST MEMORY AUTOMATED TEST SUITE")
    print("=" * 70)

    passed = 0
    failed = 0
    start_total = time.time()

    for name, test_fn in TEST_CASES:
        print(f"\n[RUNNING] {name}...")
        t0 = time.time()
        try:
            test_fn()
            elapsed = time.time() - t0
            print(f"[PASS] {name} ({elapsed*1000:.1f}ms)")
            passed += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[FAIL] {name} ({elapsed*1000:.1f}ms): {e}")
            failed += 1

    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"  TEST RESULTS: {passed} PASSED | {failed} FAILED | {total_time:.2f}s total")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed.")

if __name__ == "__main__":
    main()
