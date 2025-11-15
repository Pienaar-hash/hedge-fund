"""
Compatibility stub for legacy orderbook entry gating.
Real orderbook filters were removed in v5.9; returning True keeps callers happy.
"""


def evaluate_entry_gate(*args, **kwargs):
    """
    Compatibility stub. Always allow entries.
    """
    return True
