# Part of Mati's field — the phase-field substrate its memory is built on.
#
# ⚠️ THIS IS THE AUTHORITATIVE COPY. These modules began as a shared study
# codebase and were split off so the field could change without anything
# else moving under it. Nothing outside Mati imports them.
#
# ⚠️ KEEP THIS A COMMENT, NOT A DOCSTRING. A docstring here would sit
# before `from __future__`, which is a SyntaxError that `ast.parse`
# does not report.
"""
Reference overlap judgment.

Pure functions: no field imports, no DB imports, no side effects.
Called by chat_qa.py now; callable by Barrier output gate later.

Reference overlap: token-level F1 between generated token ids and
the best-matching reference answer in the QA dataset.  UNK (id=0)
and BOS (id=1) are excluded — they carry no semantic content.

Design constraints:
    - No global state
    - No side effects
    - Callable from any context (chat, barrier, test)
    - Reference corpus is a standalone file, not embedded in the checkpoint
"""

import numpy as np
from collections import Counter


def reference_overlap(generated_ids, reference_ids):
    """
    Clipped multiset token F1 between generated and reference sequences.

    Uses bag-of-words counts (not sets) so that repeating a token 20 times
    does not inflate the score beyond the count in the reference.  Each token
    contributes at most min(count_gen, count_ref) to the intersection — the
    same clipping rule used by BLEU unigram precision.

    UNK (0) and BOS (1) excluded from both counts.

    Returns float in [0, 1].
    """
    gen = Counter(i for i in generated_ids if i > 1)
    ref = Counter(i for i in reference_ids if i > 1)
    if not ref or not gen:
        return 0.0
    # Clipped intersection: each token counted at most min(gen, ref) times
    clipped = sum(min(gen[k], ref[k]) for k in gen if k in ref)
    gen_total = sum(gen.values())
    ref_total = sum(ref.values())
    precision = clipped / gen_total
    recall    = clipped / ref_total
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def save_qa_encoded(path, qa_pairs, tokenizer):
    """
    Encode QA string pairs with tokenizer and save to .npy file.

    Args:
        path      : output file path (without .npy extension)
        qa_pairs  : list of (question_str, answer_str)
        tokenizer : any object with .encode(str) -> list[int]
    """
    encoded = np.array(
        [(tokenizer.encode(q), tokenizer.encode(a)) for q, a in qa_pairs],
        dtype=object,
    )
    np.save(path, encoded, allow_pickle=True)
    print(f"  QA encoded pairs saved → {path}.npy  ({len(qa_pairs)} pairs)")

