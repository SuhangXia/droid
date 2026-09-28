import time


def time_ms():
    return time.time_ns() // 1_000_000


def monotonic_ns():
    """Host monotonic timestamp for cross-stream alignment.

    ``time_ms`` remains unchanged for dataset compatibility. New integrations
    should persist this value alongside device and legacy wall-clock fields.
    """

    return time.monotonic_ns()
