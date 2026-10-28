#!/usr/bin/env python3
"""CPU-only check of the unified-checkpoint GGUF metadata writer (no NeMo, no checkpoint).

Drives scripts/unified_kvs.py with a recording writer and checks the keys the loader
reads, the handling of the training windows (scalars, lists, bad shapes), and that a
bad att_chunk_context_size fails with a clear ValueError instead of a TypeError.
Exit 0 on success.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import unified_kvs as uk  # noqa: E402


class Rec:
    def __init__(self):
        self.calls = []

    def add_bool(self, k, v):
        self.calls.append((k, "bool", v))

    def add_string(self, k, v):
        self.calls.append((k, "string", v))

    def add_array(self, k, v):
        self.calls.append((k, "array", list(v)))

    def add_uint32(self, k, v):
        self.calls.append((k, "uint32", v))

    def get(self, k):
        return [c for c in self.calls if c[0] == k]


failures = []


def check(cond, what):
    if not cond:
        failures.append(what)
        print("FAIL:", what)


def expect_value_error(att, what):
    try:
        uk.train_windows(att)
    except ValueError as e:
        check("att_chunk_context_size" in str(e), f"{what}: message names the key ({e})")
        return
    except Exception as e:  # a bare TypeError etc. is the bug we fixed
        check(False, f"{what}: raised {type(e).__name__}, not ValueError")
        return
    check(False, f"{what}: no error raised")


# 1. The checkpoint's shape: [[70], [1,2,7,13], [0,1,2,3,4,7,13]].
w = Rec()
uk.write_unified_kvs(w, "dcc", [[70], [1, 2, 7, 13], [0, 1, 2, 3, 4, 7, 13]])
check(w.get("parakeet.unified.present") == [("parakeet.unified.present", "bool", True)], "present flag")
check(w.get("parakeet.encoder.att_context_style") ==
      [("parakeet.encoder.att_context_style", "string", "chunked_limited_with_rc")], "style key")
check(w.get("parakeet.encoder.conv_context_style") ==
      [("parakeet.encoder.conv_context_style", "string", "dcc")], "conv style key")
check(w.get("parakeet.unified.train_chunk") ==
      [("parakeet.unified.train_chunk", "array", [1, 2, 7, 13])], "train_chunk values")
check(w.get("parakeet.unified.train_left") == [("parakeet.unified.train_left", "array", [70])], "train_left values")
check(w.get("parakeet.unified.presets") ==
      [("parakeet.unified.presets", "array", [70, 13, 13, 70, 7, 7, 70, 2, 5, 70, 1, 3, 70, 1, 2, 70, 1, 1])],
      "presets are the card's six triples")
check(w.get("parakeet.unified.default_preset") == [("parakeet.unified.default_preset", "uint32", 1)], "default preset")
check(w.get("parakeet.unified.valid_length") == [("parakeet.unified.valid_length", "bool", True)], "valid_length on")
check(not any(k.startswith("parakeet.streaming.") or k.startswith("parakeet.encoder.causal")
              or k == "parakeet.encoder.conv_causal" for k, _, _ in w.calls),
      "no streaming or causal keys for unified checkpoints")

# 2. Scalars in the parts are accepted as one-element lists.
w = Rec()
uk.write_unified_kvs(w, "dcc", [70, 13, 13])
check(w.get("parakeet.unified.train_left") == [("parakeet.unified.train_left", "array", [70])], "scalar parts")

# 3. No training windows: the informational keys are omitted, the rest still written.
w = Rec()
uk.write_unified_kvs(w, "dcc", None)
check(not w.get("parakeet.unified.train_left"), "None writes no train keys")
check(w.get("parakeet.unified.valid_length"), "None still writes valid_length")

# 4. Bad shapes fail with a clear ValueError.
expect_value_error([[70], [1, 2]], "two parts")
expect_value_error([[70], [1, 2], [0], [9]], "four parts")
expect_value_error(70, "a bare scalar")
expect_value_error([[70], "x", [0]], "a string part")
expect_value_error([[70], [1.5], [0]], "a non-integer entry")
expect_value_error([[70], [None], [0]], "a None entry")

if failures:
    print(f"check_unified_kvs: {len(failures)} failure(s)")
    sys.exit(1)
print("check_unified_kvs: OK")
