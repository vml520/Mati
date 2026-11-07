"""
Every turn's coherence, whether it was kept or not.

⚠️ WHY THIS HAD TO BE BUILT BEFORE THE QUESTION COULD BE ASKED. An episode is
written only when `coherence >= EPISODE_GATE_MIN` (0.7). A turn that misses the
gate leaves **no trace at all** — no row, no counter, nothing. And coherence is
not stored on the episodes that do pass, so even the passing side is invisible
afterwards.

So "is 0.7 the right threshold" was unanswerable from the store. Not hard to
answer — impossible. There was no record of what had been rejected, or by how
much it missed.

This records one row per turn: when, the coherence, whether it cleared, and a
HASH of the text. The hash rather than the text because the point is a
distribution, not a transcript, and a second plaintext copy of every
conversation is a poor trade for a histogram.

Changes no behaviour. The gate still decides exactly what it decided before;
this only writes down what it decided and on what number.
"""

from __future__ import annotations

import hashlib
import sqlite3

from . import bridge

TABLE = "gate_log"


def ensure(db_path=None):
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH)
    try:
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                    at         TEXT,
                    coherence  REAL,
                    kept       INTEGER,
                    gate       REAL,
                    reply_coh  REAL,
                    consolidated INTEGER,
                    n_words    INTEGER,
                    reply_words INTEGER,
                    reply_sents INTEGER,
                    text_hash  TEXT
                )""")
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _migrate(conn):
    """Columns added after the table existed. NULL for rows written before."""
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({TABLE})")}
    for name in ("reply_words", "reply_sents"):
        if name not in cols:
            conn.execute(f"ALTER TABLE {TABLE} ADD COLUMN {name} INTEGER")


def record(coherence, kept, gate, reply_coh=None, consolidated=None,
           text="", reply="", db_path=None):
    """One row per turn. Never raises — a broken log must not break a turn."""
    try:
        h = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]
        conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=5)
        try:
            _migrate(conn)
            # ⚠️ REPLY LENGTH IS HERE FOR A REASON THAT COULD NOT BE SETTLED
            # WITHOUT IT. The waking block roughly quadruples replies — 12
            # words without it, 44 with — and that reproduced twice. WHICH PART
            # of the block does it did not: an ablation blamed the principles
            # (43 words against 5), and a direct comparison minutes later said
            # the opposite. Three probes, two answers. Real conversation
            # settles it; synthetic probes demonstrably do not.
            import re as _re
            rp = " ".join((reply or "").split())
            n_sent = len(_re.split(r"(?<=[.!?])\s+", rp)) if rp else 0
            conn.execute(
                f"INSERT INTO {TABLE} (at, coherence, kept, gate, reply_coh, "
                f"consolidated, n_words, reply_words, reply_sents, text_hash) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (bridge._now(), float(coherence), int(bool(kept)),
                 float(gate),
                 float(reply_coh) if reply_coh is not None else None,
                 int(bool(consolidated)) if consolidated is not None else None,
                 len((text or "").split()), len(rp.split()), n_sent, h))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass                    # observability must never cost a conversation


def _reply_stats(db_path=None):
    """Reply length, and whether it moves with coherence."""
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH)
    try:
        rows = conn.execute(
            f"SELECT coherence, reply_words, reply_sents FROM {TABLE} "
            f"WHERE reply_words IS NOT NULL AND reply_words > 0").fetchall()
    except sqlite3.Error:
        return {"n": 0}
    finally:
        conn.close()
    if not rows:
        return {"n": 0}
    w = sorted(r[1] for r in rows)
    s_ = sorted(r[2] for r in rows if r[2])
    return {"n": len(rows),
            "median_words": w[len(w) // 2],
            "p90_words": w[min(int(0.9 * len(w)), len(w) - 1)],
            "median_sentences": s_[len(s_) // 2] if s_ else None}


def summary(db_path=None):
    """The distribution, and what other thresholds would have kept."""
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH)
    try:
        rows = conn.execute(
            f"SELECT coherence, kept FROM {TABLE} "
            f"WHERE coherence IS NOT NULL").fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if not rows:
        return {"n": 0}
    vals = sorted(c for c, _k in rows)
    n = len(vals)

    def pct(p):
        return vals[min(int(p * n), n - 1)]

    out = {"n": n, "kept": sum(k for _c, k in rows),
           "min": vals[0], "max": vals[-1],
           "median": pct(0.5), "p10": pct(0.10), "p90": pct(0.90)}
    # What each candidate threshold WOULD have kept. This is the whole point:
    # the current gate's effect is only legible against the alternatives.
    out["reply"] = _reply_stats(db_path)
    out["would_keep"] = {
        f"{t:.2f}": sum(1 for v in vals if v >= t)
        for t in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)}
    return out
