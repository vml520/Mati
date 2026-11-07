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
Character-level tokenizer for the field LM.

Vocabulary (fixed):
    ID 0 : <UNK>  — unknown character
    ID 1 : <BOS>  — beginning of sequence
    ID 2+: printable characters (space, a-z, A-Z, 0-9, punctuation)
"""

_CHARS = (
    ' abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '0123456789'
    '.,!?;:\'"-()\n'
)

_SPECIAL  = ['<UNK>', '<BOS>']
_VOCAB    = _SPECIAL + list(_CHARS)
VOCAB_SIZE = len(_VOCAB)

_TO_ID = {ch: i for i, ch in enumerate(_VOCAB)}
_TO_CH = {i: ch for i, ch in enumerate(_VOCAB)}

UNK_ID = 0
BOS_ID = 1


def encode(text):
    """Text string → list of token IDs."""
    return [_TO_ID.get(c, UNK_ID) for c in text]


def decode(ids):
    """List of token IDs → text string."""
    out = []
    for i in ids:
        ch = _TO_CH.get(i, '?')
        if ch not in ('<UNK>', '<BOS>'):
            out.append(ch)
    return ''.join(out)

