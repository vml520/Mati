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
BPE Subset Tokenizer — Phase 4.

Uses GPT-2 BPE (via tiktoken) with vocabulary reduction:
    1. Tokenize training corpus with full GPT-2 tokenizer (50,257 tokens).
    2. Count token frequencies in corpus.
    3. Keep the top min(max_vocab, n_unique) most frequent tokens.
    4. Remap to contiguous IDs 0..V-1.

Special IDs (same contract as char tokenizer):
    0 = UNK  — rare / out-of-vocabulary tokens
    1 = BOS  — beginning of sequence (left-padding)
    2..V-1   — vocabulary tokens, frequency-ordered

Phase comparison:
    Phase 1–3 char tokenizer:  V=77,  ~1 char per token
    Phase 4 BPE tokenizer:     V≈300–2000, ~3–5 chars per token

Same field architecture (encoder, field_lm, modes, trainer) works unchanged —
only vocab_size parameter changes.
"""

import json
from collections import Counter

import tiktoken

UNK_ID = 0
BOS_ID = 1


class BPESubsetTokenizer:
    """
    GPT-2 BPE tokenizer reduced to the V most frequent tokens in a corpus.

    Usage:
        tok = BPESubsetTokenizer(max_vocab=2000)
        tok.fit(corpus_text)                  # build vocab from corpus
        ids  = tok.encode("hello world")      # text → list[int]
        text = tok.decode(ids)                # list[int] → text
        tok.save("vocab.json")                # persist
        tok2 = BPESubsetTokenizer.load("vocab.json")

    Vocabulary layout:
        ID 0  = UNK  (any token not in the kept set)
        ID 1  = BOS  (padding / start-of-sequence)
        ID 2+ = corpus tokens, most frequent first
    """

    def __init__(self, max_vocab: int = 2000):
        self._gpt2      = tiktoken.get_encoding('gpt2')
        self.max_vocab  = max_vocab
        self.vocab_size = 2          # grows to V after fit()
        self._id_map    = {}         # gpt2_id   → remapped_id
        self._rev_map   = {UNK_ID: None, BOS_ID: None}  # remapped_id → gpt2_id
        self._freq_rank = []         # list of (token_str, count) for display

    # ── Build vocabulary ───────────────────────────────────────────────────────

    def fit(self, text: str) -> 'BPESubsetTokenizer':
        """
        Build vocabulary from text corpus.

        Tokenises with full GPT-2 BPE, counts frequencies, keeps
        the max_vocab-2 most frequent tokens (2 slots reserved for UNK/BOS),
        then remaps IDs to 0..V-1 ordered by frequency (most common = lowest ID).
        """
        raw_ids = self._gpt2.encode(text)
        freq    = Counter(raw_ids)

        # Keep top N tokens (most frequent first, ties broken by gpt2 ID)
        n_keep = min(self.max_vocab - 2, len(freq))
        top    = [tok_id for tok_id, _ in freq.most_common(n_keep)]

        self._id_map  = {}
        self._rev_map = {UNK_ID: None, BOS_ID: None}

        for new_id, orig_id in enumerate(top, start=2):
            self._id_map[orig_id]  = new_id
            self._rev_map[new_id]  = orig_id

        self.vocab_size = 2 + n_keep

        # Human-readable frequency table (for display)
        self._freq_rank = [
            (repr(self._gpt2.decode([orig_id])), count)
            for orig_id, count in freq.most_common(n_keep)
        ]
        return self

    # ── Encode / decode ────────────────────────────────────────────────────────

    def encode(self, text: str) -> list:
        """text → list of remapped token IDs (UNK=0 for out-of-vocab tokens)."""
        raw = self._gpt2.encode(text)
        return [self._id_map.get(t, UNK_ID) for t in raw]

    def encode_with_phases(self, text: str, phi) -> tuple:
        """
        Encode text returning (ids, phases).

        For known vocab tokens: phases[i] = phi[subset_id].
        For UNK tokens: phases[i] = circular_mean(phi[subwords]) on S¹.
            The phase is derived from the token's byte decomposition —
            unknown words inherit meaning from their known parts.
        phases[i] may be None if no known subwords are found (truly unknown).
        """
        raw    = self._gpt2.encode(text)
        ids    = []
        phases = []
        for gpt2_id in raw:
            subset_id = self._id_map.get(gpt2_id, UNK_ID)
            ids.append(subset_id)
            if subset_id != UNK_ID:
                phases.append(float(phi[subset_id]))
            else:
                phases.append(self._compositional_phase(gpt2_id, phi))
        return ids, phases

    def _compositional_phase(self, gpt2_id: int, phi) -> float:
        """
        Circular mean of phi for known-vocab subwords of a GPT-2 token.

        Strategy:
            1. Decode token to UTF-8 string.
            2. Re-encode each character with tiktoken → GPT-2 token IDs.
            3. For each subword token in our subset vocab, collect phi[id].
            4. Circular mean on S¹ = arctan2(⟨sin φ⟩, ⟨cos φ⟩).

        Returns None if no known subwords are found.
        """
        import numpy as np
        try:
            token_bytes = self._gpt2.decode_single_token_bytes(gpt2_id)
            token_str   = token_bytes.decode('utf-8', errors='replace')
        except Exception:
            return None

        gathered = []

        # Pass 1: character-level subword lookup
        for char in token_str:
            try:
                for cg_id in self._gpt2.encode(char):
                    if cg_id in self._id_map:
                        gathered.append(float(phi[self._id_map[cg_id]]))
            except Exception:
                continue

        # Pass 2: byte-level fallback if no character subwords found
        if not gathered:
            for byte_val in token_bytes:
                try:
                    for bg_id in self._gpt2.encode(
                            bytes([byte_val]).decode('latin-1')):
                        if bg_id in self._id_map:
                            gathered.append(float(phi[self._id_map[bg_id]]))
                except Exception:
                    continue

        if not gathered:
            return None

        arr = np.array(gathered)
        return float(
            np.arctan2(np.mean(np.sin(arr)), np.mean(np.cos(arr))) % (2 * np.pi)
        )

    def decode(self, ids: list) -> str:
        """list of remapped IDs → text (UNK and BOS are skipped)."""
        orig = []
        for i in ids:
            if i >= 2 and i in self._rev_map:
                orig_id = self._rev_map[i]
                if orig_id is not None:
                    orig.append(orig_id)
        return self._gpt2.decode(orig) if orig else ''

    def token_str(self, remapped_id: int) -> str:
        """Human-readable string for a remapped token ID."""
        if remapped_id == UNK_ID:
            return '<UNK>'
        if remapped_id == BOS_ID:
            return '<BOS>'
        orig = self._rev_map.get(remapped_id)
        if orig is None:
            return f'<{remapped_id}>'
        return repr(self._gpt2.decode([orig]))

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save vocab mapping as JSON."""
        data = {
            'max_vocab':  self.max_vocab,
            'vocab_size': self.vocab_size,
            'id_map':     {str(k): v for k, v in self._id_map.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"  Vocabulary saved → {path}  ({self.vocab_size} tokens)")

    @classmethod
    def load(cls, path: str) -> 'BPESubsetTokenizer':
        """Load tokenizer from saved JSON."""
        with open(path) as f:
            data = json.load(f)
        tok = cls(max_vocab=data['max_vocab'])
        tok._id_map   = {int(k): v for k, v in data['id_map'].items()}
        tok._rev_map  = {UNK_ID: None, BOS_ID: None}
        tok._rev_map.update({v: int(k) for k, v in tok._id_map.items()})
        tok.vocab_size = data['vocab_size']
        print(f"  Vocabulary loaded ← {path}  ({tok.vocab_size} tokens)")
        return tok

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def print_vocab_summary(self, n=20):
        """Print top-n most frequent tokens and vocab stats."""
        print(f"  BPE vocab: {self.vocab_size} tokens  (max={self.max_vocab})")
        if self._freq_rank:
            print(f"  Top {min(n, len(self._freq_rank))} tokens:")
            for i, (tok_str, count) in enumerate(self._freq_rank[:n]):
                new_id = i + 2
                print(f"    [{new_id:4d}] {tok_str:20s}  ×{count}")
