#!/usr/bin/env python3
"""Dump NeMo's chunked_limited_with_rc attention mask for the unit test.

Calls NeMo's own ConformerEncoder._create_masks (no copy of the rule) for a set
of (left, chunk, right) windows and writes the visibility matrix as text, one
case per block, read by tests/test_rc_mask.cpp:

    case <L> <C> <R> <T> <valid>
    <T rows of T characters: '1' = query row may see key, '0' = masked>

Only the rows of valid query frames (qi < valid) are compared by the test. The
pad mask (keys >= valid) is included in the matrix, as in NeMo.

Needs NeMo main at or after b19b1a79d (the unified checkpoint's NeMo). Pin used
to make the committed fixture: cd5e4fbd36ba098d0c78b4552f6f9592ad34de93.

Usage (in a venv that has nemo_toolkit[asr]):
    python scripts/gen_rc_mask_fixture.py > tests/fixtures/rc_mask_nemo.txt
"""
import sys
from types import SimpleNamespace

import torch
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder

# (left, chunk, right) in 80 ms frames: the unified checkpoint's presets
# (1.12 s, 0.56 s, 0.16 s, 1.04 s, 0.24 s, 0.08 s and 2.0 s / 0.56 s).
PRESETS = [(70, 13, 13), (70, 7, 7), (70, 2, 5), (70, 1, 3), (70, 1, 2), (70, 1, 1), (25, 2, 5)]

cases = []
for L, C, R in PRESETS:
    N = L + C + R
    cases.append((L, C, R, N + 1, N))   # steady window, NeMo's layout (encoder tensor N+1, length N)
    cases.append((L, C, R, N, N))       # steady window, trimmed to N frames
# Warm-up: the left context grows by C per step from zero.
for k in (1, 2, 3):
    cases.append((70, 13, 13, k * 13, k * 13))
cases.append((70, 7, 7, 15, 15))        # first window of the 1.12 s preset (7 + 7 + 1)
cases.append((70, 7, 7, 37, 29))       # last window: partial, chunk longer than C
cases.append((70, 2, 5, 4, 4))
cases.append((70, 1, 1, 4, 3))

stub = SimpleNamespace(self_attention_model="rel_pos", att_context_style="chunked_limited_with_rc")

out = sys.stdout
for L, C, R, T, valid in cases:
    pad_mask, att_mask = ConformerEncoder._create_masks(
        stub,
        att_context_size=[L, C, R],
        padding_length=torch.tensor([valid]),
        max_audio_length=T,
        offset=None,
        device="cpu",
    )
    visible = ~att_mask[0]  # att_mask is True where masked (NeMo inverts it last)
    out.write(f"case {L} {C} {R} {T} {valid}\n")
    for qi in range(T):
        row = "".join("1" if bool(visible[qi, kj]) else "0" for kj in range(T))
        out.write(row + "\n")
