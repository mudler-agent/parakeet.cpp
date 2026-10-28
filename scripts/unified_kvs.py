"""GGUF metadata for unified checkpoints (NeMo att_context_style chunked_limited_with_rc).

Standard library and numpy only, so a unit test can import it without NeMo.
The converter calls write_unified_kvs(); see docs/conversion.md for the keys.
"""

UNIFIED_ATT_STYLE = "chunked_limited_with_rc"

# [L, C, R] buffered-streaming windows in 80 ms encoder frames. The checkpoint's
# config does not carry these: its att_context_size is [-1,-1] and it only lists
# the training windows (att_chunk_context_size). These values are the presets of
# the model card (nvidia/nemotron-labs-unified-asr-0.6b), through NeMo's
# int(seconds * 100 / 8) truncation (5.6 s -> 70, 1.04 s -> 13). Keep them
# in step with the card; a checkpoint that carries its own list should use it.
UNIFIED_PRESETS = [(70, 13, 13), (70, 7, 7), (70, 2, 5), (70, 1, 3), (70, 1, 2), (70, 1, 1)]
UNIFIED_DEFAULT_PRESET = 1  # (70, 7, 7), the card's example command

_TRAIN_NAMES = ("train_left", "train_chunk", "train_right")


def _as_int_list(part, name):
    """An int or a sequence of ints, as a list of ints. Anything else is a clear ValueError."""
    if isinstance(part, int) and not isinstance(part, bool):
        return [part]
    try:
        items = list(part)
    except TypeError:
        raise ValueError(f"att_chunk_context_size {name}: expected an int or a list of ints, "
                         f"got {type(part).__name__}") from None
    return [_as_int(x, name, items) for x in items]


def _as_int(x, name, items):
    """An integer value; a float only when it is whole (1.5 is an error, not 1)."""
    if isinstance(x, bool):
        pass
    elif isinstance(x, int):
        return int(x)
    elif isinstance(x, float) and x.is_integer():
        return int(x)
    elif hasattr(x, "__index__"):  # numpy integers
        return int(x.__index__())
    raise ValueError(f"att_chunk_context_size {name}: non-integer entries {items!r}")


def train_windows(att_chunk_context_size):
    """[(name, [ints]), ...] for the training windows, or [] when the config has none.

    Raises ValueError when the value is not [left, chunk, right] with integer parts.
    """
    if att_chunk_context_size is None:
        return []
    try:
        parts = list(att_chunk_context_size)
    except TypeError:
        raise ValueError("att_chunk_context_size: expected [left, chunk, right], got "
                         f"{type(att_chunk_context_size).__name__}") from None
    if len(parts) != 3:
        raise ValueError(f"att_chunk_context_size: expected 3 parts [left, chunk, right], got {len(parts)}")
    return [(name, _as_int_list(part, name)) for name, part in zip(_TRAIN_NAMES, parts)]


def write_unified_kvs(w, conv_context_style, att_chunk_context_size):
    """Write the unified-checkpoint KVs on GGUF writer w (only for chunked_limited_with_rc)."""
    w.add_bool("parakeet.unified.present", True)
    # The loader reads this key to pick the unified block over the streaming one.
    w.add_string("parakeet.encoder.att_context_style", UNIFIED_ATT_STYLE)
    w.add_string("parakeet.encoder.conv_context_style", str(conv_context_style))
    # Training window sets (informational).
    for name, values in train_windows(att_chunk_context_size):
        w.add_array(f"parakeet.unified.{name}", values)
    flat = [v for p in UNIFIED_PRESETS for v in p]
    w.add_array("parakeet.unified.presets", flat)
    w.add_uint32("parakeet.unified.default_preset", UNIFIED_DEFAULT_PRESET)
    # NeMo's encoder gets length = floor(samples / hop) frames, one less than the
    # mel tensor holds (center padding). The engine trims the mel to match.
    w.add_bool("parakeet.unified.valid_length", True)
