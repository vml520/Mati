"""
Reading a file, as something Mati does slowly.

    python3 -m Mati.study notes.md              # queue it
    python3 -m Mati.study --status
    python3 -m Mati.study --work 20             # study now instead of on idle

WHY NOT "TOKENIZE THE FILE AND IMPRINT IT". Imprinting costs ~1s per token, so
a 5,000-token document is about 80 minutes — and it deforms θ₀ globally rather
than locally, measurably homogenising it. A bulk imprint would flatten the
memory it was supposed to add to. That is the wall this feature kept hitting.

So: uploading is fast and studying is slow. An upload chunks the file, records
ONE episode ("I read notes.md today"), and queues the chunks. The worker drains
the queue while Mati is idle, on a budget. Reading is quick; consolidation
happens in sleep.

TESTING WHAT IT LEARNED — and what the measurement changed
----------------------------------------------------------
The design was: test before (skip what Mati already knows) and test after
(did it stick?), both via `recall(address_vector(chunk))`. Test-after works.
**Test-before does not, and the numbers say so plainly.** Measured against the
live corpus :

    exact stored text        1.000        <- retrieval is exact and stable
    a slice of stored text   0.424–0.688  <- material Mati HAS read
    genuinely new text       0.305–0.754  <- material it has never seen

New text scores *higher on average* than known material. The classes are
inverted and overlapping, so no threshold separates them: at 0.7 a brand-new
sentence about mitochondria is skipped as "already known", at 0.4 nothing is
ever learned. This is the documented limit of the retrieval — trajectory
features discriminate token surface, not meaning, and shared stopwords
dominate — surfacing exactly where the design leaned on it hardest.

What is used instead:

  TEST BEFORE  — a content hash. Exact, free, and correct for the case that
                 actually recurs: the same file, or overlapping files, offered
                 twice. It does not catch paraphrase, and does not pretend to.
  TEST AFTER   — re-address the chunk and confirm it retrieves itself. This is
                 the test that carries the meaning ("I read it but it didn't
                 stick"), and it is the one that works.
  COST BOUND   — a per-run study budget. The original plan bounded cost by how
                 much got skipped; since skipping is unreliable, the budget
                 does it directly and honestly.

The similarity gate is left implemented but OFF (`--use-similarity-gate`), so
the day retrieval becomes semantic it can be measured again rather than
rebuilt.

PROVENANCE — a fourth kind
--------------------------
Mati separates remembered (lived), known (from the language model) and
computed (deterministic). Studied material is none of those, and the
distinction has to be written down at the moment of storage or it is gone: the
`source` column carries `studied:<filename>`, and `bridge.state_block` renders
those entries differently so Mati can say "I read that in the file you gave
me" rather than "I remember us discussing it".
"""

import argparse
import hashlib
import os
import re
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .field import notes as N
from . import vault

from . import admit
from . import bridge
from . import learning
from . import config as C
from . import exclusive

STUDY_PREFIX = "studied:"

# `bridge.remember` addresses and imprints only the first TRIPLET_CAP tokens,
# so anything past that in a chunk is silently never learned. Chunks are built
# to fit — about 26 words at this tokenizer's rate.
MAX_CHUNK_TOKENS = C.TRIPLET_CAP

# Fewer than 2 tokens imprints nothing (`imprint_question` early-returns on an
# empty pair list), so a fragment shorter than this is dropped at chunk time
# rather than queued to fail later.
MIN_CHUNK_TOKENS = 4

_DDL = """
CREATE TABLE IF NOT EXISTS study_queue (
    chunk_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    origin    TEXT NOT NULL,
    body      TEXT NOT NULL,
    sha       TEXT NOT NULL UNIQUE,
    status    TEXT NOT NULL DEFAULT 'queued',
    note_id   INTEGER,
    added     TEXT,
    studied   TEXT,
    attempts  INTEGER NOT NULL DEFAULT 0,
    detail    TEXT
);
CREATE INDEX IF NOT EXISTS study_status ON study_queue(status);
"""


def _conn(db_path=None):
    conn = sqlite3.connect(db_path or bridge.NOTES_DB_PATH)
    # WAL still allows only ONE writer. Without a busy timeout a write that
    # lands while the idle worker is reindexing raises "database is locked"
    # immediately, which surfaced as POST /study closing the connection
    # mid-request. Wait for the other writer instead of failing the request.
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(_DDL)
    conn.commit()
    return conn


def _sha(text):
    return hashlib.sha256(" ".join(text.split()).lower().encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text, tok, max_tokens=MAX_CHUNK_TOKENS):
    """Split into pieces that fit whole inside one imprint.

    Paragraph first, then sentence, then — only if a single sentence is still
    too long — words. Splitting mid-sentence is a last resort because a chunk
    is what Mati will later recall as a unit.
    """
    chunks = []
    for para in _PARA.split(text):
        para = " ".join(para.split())
        if not para:
            continue
        buf = []
        for sentence in _SENT.split(para):
            if not sentence.strip():
                continue
            candidate = (" ".join(buf + [sentence])).strip()
            if len(tok.encode(candidate)) <= max_tokens:
                buf.append(sentence)
                continue
            if buf:
                chunks.append(" ".join(buf).strip())
                buf = []
            if len(tok.encode(sentence)) <= max_tokens:
                buf = [sentence]
            else:
                chunks.extend(_split_words(sentence, tok, max_tokens))
        if buf:
            chunks.append(" ".join(buf).strip())
    return [c for c in chunks if len(tok.encode(c)) >= MIN_CHUNK_TOKENS]


def _split_words(sentence, tok, max_tokens):
    out, buf = [], []
    for word in sentence.split():
        if len(tok.encode(" ".join(buf + [word]))) > max_tokens and buf:
            out.append(" ".join(buf))
            buf = []
        buf.append(word)
    if buf:
        out.append(" ".join(buf))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Upload — fast: chunk, dedupe, queue, record one episode
# ─────────────────────────────────────────────────────────────────────────────

def enqueue(text, origin, tok, db_path=None):
    """Queue a document for study. Returns a summary; does no imprinting."""
    chunks = chunk_text(text, tok)
    conn = _conn(db_path)
    queued = duplicate = 0
    try:
        for body in chunks:
            sha = _sha(body)
            try:
                conn.execute(
                    "INSERT INTO study_queue (origin, body, sha, added) "
                    # ⚠️ SEALED. 551,988 characters of book text sat here in
                    # the clear while `notes.body` was encrypted — same file,
                    # same Touch ID. `sha` stays plain: already one-way, and
                    # the duplicate check needs to match on it.
                    "VALUES (?,?,?,?)",
                    (origin, vault.seal_soft(body), sha, N._now()))
                queued += 1
            except sqlite3.IntegrityError:
                # TEST BEFORE. Exact, and correct for the case that recurs:
                # the same file offered twice, or two files that overlap.
                duplicate += 1
        conn.commit()
    finally:
        conn.close()
    return {"origin": origin, "chunks": len(chunks),
            "queued": queued, "already_had": duplicate}


# Formats `textutil` can turn into text. ⚠️ macOS ships it, so this costs no
# dependency and no install — the same reason `/usr/bin/say` is the voice.
RICH_SUFFIXES = (".rtf", ".rtfd", ".doc", ".docx", ".odt", ".html", ".htm",
                 ".webarchive")


def clean_text(text):
    """Strip what should never reach the field, whatever the source.

    ⚠️ CONTROL AND FORMAT CHARACTERS ARE NOT TEXT. A zero-width joiner or a
    bidi override is invisible on screen and present in the store — and
    `FUTURE_collaborate.md` records why that matters beyond tidiness: hidden
    characters are how instructions get smuggled past a reader who is only
    looking at what is visible.

    ⚠️ NFC, because the same word in two normal forms is two different words
    to a lexical index. `notes.search` stems what it is given; it cannot know
    that a decomposed é is the é it already holds.
    """
    import unicodedata
    text = unicodedata.normalize("NFC", text or "")
    return "".join(
        c for c in text
        if c in "\n\t" or unicodedata.category(c) not in ("Cc", "Cf"))


def read_file(path):
    """Text of a file Mati can study.

    ⚠️ RICH FORMATS ARE CONVERTED, NOT READ. This was `f.read()`, so handing
    it an RTF studied the MARKUP: a 370 KB reference was 32% control words —
    16,829 of them — and `{\rtf1\ansi\cocoartf2822 \fonttbl\f0\fswiss` would
    have gone into the field as prose. It would not have errored. It would
    have queued 2,200 chunks and spent hours imprinting font tables, which is
    the worst shape of failure: confident, slow and wrong.

    ⚠️ `textutil` SHIPS WITH macOS, so this adds no dependency — the same
    argument as `/usr/bin/say` for the voice and `sysctl` for the machine
    check. A reader that needs installing before it can read is not a reader.
    """
    import os
    import subprocess
    if os.path.splitext(path)[1].lower() in RICH_SUFFIXES:
        try:
            out = subprocess.run(
                ["/usr/bin/textutil", "-convert", "txt", "-stdout",
                 "-encoding", "UTF-8", path],
                capture_output=True, timeout=120)
            if out.returncode == 0 and out.stdout.strip():
                return clean_text(out.stdout.decode("utf-8", "replace"))
            # ⚠️ FALL THROUGH RATHER THAN RAISE, but say so — a conversion
            # that failed is better reported as "this looks like markup" by
            # the caller than as a traceback here.
        except Exception:
            pass
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return clean_text(f.read())


def reading_episode(origin, summary):
    """The one episode an upload records — that it read the thing, not what it said.

    What the file *contains* is not something Mati lived; it is studied
    material, and it arrives note by note as the queue drains. Conflating the
    two here is exactly the provenance blur this feature has to avoid.
    """
    return (f"I read {origin} today. "
            f"It came to {summary['chunks']} pieces; "
            f"{summary['queued']} were new to me"
            + (f" and {summary['already_had']} I already had"
               if summary["already_had"] else "")
            + ". I have not studied them yet — that happens when I am idle.")


# ─────────────────────────────────────────────────────────────────────────────
# Study — slow: imprint one chunk, then check whether it took
# ─────────────────────────────────────────────────────────────────────────────

def _already_recallable(lm, tok, body, min_sim):
    """The similarity gate, kept for the day retrieval becomes semantic.

    OFF by default. : known material scores 0.424–0.688 and
    unseen material 0.305–0.754 — overlapping, and inverted on the mean. Any
    threshold either skips new material or learns nothing.
    """
    hits = bridge.recall(bridge.address_vector(lm, tok, body),
                         k=1, lm=lm, tok=tok, min_sim=-1.0)
    return bool(hits) and (1.0 - hits[0][0]) >= min_sim


def study_one(lm, tok, db_path=None, similarity_gate=None, verbose=False):
    """Learn the oldest queued chunk. Returns a result dict, or None if empty."""
    conn = _conn(db_path)
    try:
        row = conn.execute(
            "SELECT chunk_id, origin, body FROM study_queue "
            "WHERE status='queued' ORDER BY chunk_id LIMIT 1").fetchone()
        if row is None:
            return None
        chunk_id, origin, body = row
        body = vault.open_body(body)      # plaintext passes through unchanged

        if similarity_gate is not None and _already_recallable(
                lm, tok, body, similarity_gate):
            conn.execute("UPDATE study_queue SET status='skipped', studied=?, "
                         "detail='similarity gate' WHERE chunk_id=?",
                         (N._now(), chunk_id))
            conn.commit()
            return {"chunk_id": chunk_id, "status": "skipped", "body": body}

        # Checked here as well as in `remember`, so the queue can record WHY
        # a chunk was declined instead of reporting a bare "too_short".
        verdict = admit.check(body, f"{STUDY_PREFIX}{origin}", db_path=db_path)
        if not verdict.ok:
            conn.execute("UPDATE study_queue SET status='refused', studied=?, "
                         "attempts=attempts+1, detail=? WHERE chunk_id=?",
                         (N._now(), f"{verdict.rule}: {verdict.why}", chunk_id))
            conn.commit()
            if verbose:
                print(f"  [study] refused   {verdict.rule}: {verdict.why}",
                      file=sys.stderr)
            return {"chunk_id": chunk_id, "status": "refused",
                    "rule": verdict.rule, "body": body}

        t0 = time.time()
        # ⚠️ WHAT IS STORED IS WHAT MATI NOW KNOWS, not the author's sentence.
        # A studied note used to hold the passage verbatim, and you can quote a
        # passage or paraphrase it but you cannot SPEAK FROM it — which is why
        # studied material never transferred into a reply the way memory did.
        # : 2,670 studied notes, ~67h of imprinting, 202
        # ever recalled (7.6%).
        #
        # The source is kept in `verbatim` beside it, so a claim can always be
        # checked against what was actually written. Fidelity measured at 14/15
        # against an LLM judge with a working control; the one miss was an
        # inference the passage did not make, which is exactly why the source
        # is retained rather than discarded.
        #
        # If it cannot be learned, it stays a quotation. Inventing a sentence
        # to stand for a passage is the failure this is meant to prevent.
        learned = learning.learn_from(body, verbose=verbose)
        if learned:
            note = learning.store(lm, tok, learned, body,
                                  f"{STUDY_PREFIX}{origin}", db_path=db_path)
            res = (note, None) if note else None
            if verbose and note:
                print(f"  [learned] {learned[:70]}", file=sys.stderr)
        else:
            res = bridge.remember(lm, tok, body,
                                  source=f"{STUDY_PREFIX}{origin}",
                                  db_path=db_path)
        if res is None:
            conn.execute("UPDATE study_queue SET status='too_short', studied=?, "
                         "attempts=attempts+1 WHERE chunk_id=?",
                         (N._now(), chunk_id))
            conn.commit()
            return {"chunk_id": chunk_id, "status": "too_short", "body": body}
        note_id, _triplet = res

        # TEST AFTER. Exact text against the field that just absorbed it — the
        # one test the measurement supports. If it does not come back, Mati
        # read it and it did not stick, which is a real answer rather than a
        # failure to report.
        hits = bridge.recall(bridge.address_vector(lm, tok, body),
                             k=1, lm=lm, tok=tok, min_sim=-1.0)
        stuck = bool(hits) and hits[0][1].strip() == body.strip()
        sim = (1.0 - hits[0][0]) if hits else float("nan")

        conn.execute(
            "UPDATE study_queue SET status=?, note_id=?, studied=?, "
            "attempts=attempts+1, detail=? WHERE chunk_id=?",
            ("learned" if stuck else "not_stuck", note_id, N._now(),
             f"top_sim={sim:.3f}", chunk_id))
        conn.commit()
        out = {"chunk_id": chunk_id, "note_id": note_id, "body": body,
               "status": "learned" if stuck else "not_stuck",
               "top_sim": sim, "seconds": time.time() - t0}
        if verbose:
            print(f"  [study] {out['status']:<9} sim={sim:.3f} "
                  f"{out['seconds']:.0f}s  {body[:52]}", file=sys.stderr)
        return out
    finally:
        conn.close()


def study(lm, tok, budget=1, db_path=None, similarity_gate=None,
          should_stop=None, reindex_after=True, verbose=False):
    """Drain up to `budget` chunks, then repair the index.

    REINDEXING IS PART OF STUDYING, not a follow-up. Every imprint moves θ₀,
    and every stored address was measured against the θ₀ of the moment — so
    studying chunk N stales chunk N−1. Measured: two chunks studied back to
    back, and the first no longer retrieved its own exact text (0.301) though
    it had matched perfectly when written. A burst of imprints is the worst
    case for this, which makes study the place it bites hardest.

    So the batch ends by putting every address back in one space. Without it,
    only the last chunk of a batch is reliably recallable — Mati would report
    "learned" for material it had already lost.

    `should_stop()` lets an arriving turn interrupt between chunks.
    """
    done = []
    for _ in range(budget):
        if should_stop is not None and should_stop():
            break
        out = study_one(lm, tok, db_path=db_path,
                        similarity_gate=similarity_gate, verbose=verbose)
        if out is None:
            break
        done.append(out)
    if done and reindex_after:
        n = bridge.reindex(lm, tok, db_path=db_path)
        if verbose:
            print(f"  [study] reindexed {n} against the studied field",
                  file=sys.stderr)
    if done:
        bridge.save_notes_field(lm)
    return done


def status(db_path=None):
    conn = _conn(db_path)
    try:
        counts = dict(conn.execute(
            "SELECT status, count(*) FROM study_queue GROUP BY status").fetchall())
        origins = conn.execute(
            "SELECT origin, count(*), sum(status='queued') FROM study_queue "
            "GROUP BY origin ORDER BY min(chunk_id)").fetchall()
    finally:
        conn.close()
    return {"counts": counts, "pending": counts.get("queued", 0),
            "documents": [{"origin": o, "chunks": n, "pending": p}
                          for o, n, p in origins]}


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Give Mati something to read")
    ap.add_argument("path", nargs="?", help="a text file to queue")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--work", type=int, default=0,
                    help="study N chunks now instead of waiting for idle")
    ap.add_argument("--use-similarity-gate", type=float, default=None,
                    metavar="SIM",
                    help="skip chunks already recallable above SIM. Measured "
                         "not to separate known from new — see module docstring")
    ap.add_argument("--force", action="store_true",
                    help="run even if Mati holds the field (unsafe)")
    args = ap.parse_args()

    if args.status and not (args.path or args.work):
        st = status()
        print(f"pending: {st['pending']}")
        for k, v in sorted(st["counts"].items()):
            print(f"  {k:<12} {v}")
        for d in st["documents"]:
            print(f"  {d['origin']:<40} {d['chunks']:>4} chunks, "
                  f"{d['pending']:>4} pending")
        return 0

    # Queueing needs only the tokenizer; studying needs the field, and Mati
    # must not be holding it.
    if args.work:
        rc = exclusive.guard("studying", force=args.force)
        if rc:
            return rc

    if args.path:
        from .field.tokenizer_bpe import BPESubsetTokenizer
        tok = BPESubsetTokenizer.load(C.VOCAB_PATH)
        summary = enqueue(read_file(args.path), os.path.basename(args.path), tok)
        print(f"{summary['origin']}: {summary['chunks']} chunks, "
              f"{summary['queued']} queued, "
              f"{summary['already_had']} already had")
        print(f"  {reading_episode(summary['origin'], summary)}")

    if args.work:
        lm, tok = bridge.load_field()
        done = study(lm, tok, budget=args.work,
                     similarity_gate=args.use_similarity_gate, verbose=True)
        learned = sum(1 for d in done if d["status"] == "learned")
        print(f"studied {len(done)} chunks — {learned} stuck, "
              f"{len(done)-learned} did not")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
