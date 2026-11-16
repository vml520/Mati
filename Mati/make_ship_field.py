"""
Build the field that ships: baseline θ₀ + the charter, and nothing else.

    python3 -m Mati.make_ship_field            # build into .cache/ship/
    python3 -m Mati.make_ship_field --check    # report, build nothing

⚠️ NOT A PERSON'S FIELD. A notes θ₀ that has been talked to wears that
person's conversations; shipping it would ship their shape. This starts from
`config.CKPT_THETA0` — the trained checkpoint, which a fresh install already
falls back to.

⚠️ AND NOT "GENERAL KNOWLEDGE". Field addressing scores 4/111 = 0.036 against
lexical's 93/111 = 0.838 on identical cues, and θ₀ is smaller than the text it
indexes: the field is the INDEX, not the store. Imprinting an encyclopaedia
would move θ₀ a long way and leave Mati exactly as informed as before.

WHAT THIS IS FOR: otherwise the assistant has no field memory to recall
during a turn. **The field surfaces NOTES**, and a shipped store with nothing
in it means `memories_recalled: 0` on every turn of a fresh install, for as
long as it takes the person to fill it.

The rest is already in place: `bridge.charter_chunks()` reads
`source LIKE 'studied:charter:%'`, charter injection hands the charter over
whole on identity turns, and the provenance label is written —

    "THIS IS MY OWN CHARTER — what I hold myself to. Speak it as mine. I did
     not live it as an event, so never say I remember it happening."

This populates them.

⚠️ STORED FOR RECALL, NEVER FOR ENFORCEMENT. The principles are enforced in
code, not held in the field (`PRINCIPLES.md`): the field has no opinion about
a principle versus its opposite. This makes the charter RECALLABLE. It does
not make Mati principled; `admit.py` and the named rules do that, and they are
what the tests prove.

⚠️ THREE FILES, ONE MATCHED SET, VERSIONED TOGETHER:

    mati_notes.db                     the charter, as studied: notes
    mati_notes_theta0.npz             θ₀ the notes are addressed against
    mati_notes_theta0_address.npz     the space those notes are filed in

Their addresses are computed against THIS θ₀. Ship any one of them without the
others and every stored `feat` becomes a distance between two different
spaces — not a worse match, a meaningless one.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import time

CHARTER = "MATI_MISSION_AND_PRINCIPLES.txt"
# The origin string becomes `studied:charter:…`, which is the shape
# `bridge.charter_chunks`, `library._NOT_A_WORK` and `ollin.py` already expect.
ORIGIN = f"charter:{CHARTER}"


def _paths(root):
    return {
        "db": os.path.join(root, "mati_notes.db"),
        "theta": os.path.join(root, "mati_notes_theta0.npz"),
        "address": os.path.join(root, "mati_notes_theta0_address.npz"),
    }


def check(db_path=None):
    """What is in a built set. Reports, changes nothing."""
    from . import bridge
    db_path = bridge.NOTES_DB_PATH if db_path is None else db_path
    if not os.path.exists(db_path):
        return {"exists": False}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # ⚠️ THE CHUNK INDEX IS STRIPPED FOR THE SUMMARY. Verbatim charter
        # rows carry a per-chunk source (`studied:charter:FILE:7`), so a plain
        # GROUP BY turned a 24-row charter into 24 one-row "sources" and the
        # summary became unreadable. The rows keep their distinct origins;
        # only this report folds them.
        rows = conn.execute(
            "SELECT CASE WHEN source LIKE 'studied:charter:%' "
            "            THEN rtrim(source, '0123456789:') "
            "            ELSE source END AS src, count(*) "
            "FROM notes GROUP BY src").fetchall()
        feats = conn.execute(
            "SELECT count(*) FROM notes WHERE feat IS NOT NULL").fetchone()[0]
    except sqlite3.Error:
        return {"exists": True, "readable": False}
    finally:
        conn.close()
    return {"exists": True, "readable": True, "by_source": dict(rows),
            "addressed": feats}


def write_charter_verbatim(lm, tok, text, origin, db_path, verbose=True):
    """Store the charter AS WRITTEN, addressed but never imprinted.

    ⚠️ `study.enqueue` + `study.study` WOULD DO TWO THINGS WRONG HERE. First,
    `learning.py` stores what a 3B *learned* from a chunk, not the chunk —
    right for a book, wrong for a charter, which the provenance label has Mati
    speak as ITS OWN. A charter studied that way comes out as paraphrase: "I am
    Mati. I am a personal AI that runs entirely on one person's own computer"
    rewritten into the model's words, and the memory and privacy principles
    rewritten into false claims.

    ⚠️ Second, studying IMPRINTS θ₀. A few thousand words of one voice imprinted
    into a cold field push settling coherence below the episode gate on every
    turn, so a fresh install would reject EVERY episode and form no memories at
    all, silently, for ever.

    ⭐ SO THIS WRITES ROWS AND LEAVES θ₀ ALONE. The charter does not reach
    Mati by field recall — `bridge.charter_chunks()` hands it over WHOLE on
    identity turns, no addressing involved. It only has to be PRESENT. The
    shipped θ₀ stays the untouched trained baseline, which keeps the gate
    working.

    ⚠️ `feat` is still computed, so the rows are reachable by recall too — it
    costs nothing and a stored note with no address is unfindable by anything.
    """
    import sqlite3
    from . import bridge, study, vault
    chunks = [c for c in study.chunk_text(text, tok) if c.strip()]
    conn = sqlite3.connect(db_path)
    try:
        bridge._ensure_schema(conn)
        for i, chunk in enumerate(chunks, 1):
            feat = bridge.address_vector(lm, tok, chunk)
            conn.execute(
                """INSERT INTO notes
                   (body, source, triplet_w0, triplet_w1, triplet_w2,
                    gs_std, wall_created, feat, imprint_passes, verbatim)
                   VALUES (?, ?, 0, 0, 0, 0, ?, ?, 0, ?)""",
                # ⚠️ `feat.tobytes()` — the same encoding `remember` uses at
                # bridge.py:2107. A different dtype here would store an
                # address `recall` cannot compare against anything.
                (vault.seal(chunk), f"{origin}:{i}", bridge._now(),
                 feat.tobytes(), vault.seal(chunk)))
        conn.commit()
    finally:
        conn.close()
    if verbose:
        print(f"  charter: {len(text)} chars -> {len(chunks)} verbatim chunks, "
              f"θ₀ untouched")
    return len(chunks)


def personal_rows(db_path):
    """How many of Mati's own private reflections are in this store.

    ⭐ THE READER THAT GIVES `personal_only` ITS MEANING. The flag is written by
    `reflect._mark_personal`; this is what refuses a store that carries it.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return 0
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM notes WHERE personal_only = 1").fetchone()[0]
    except sqlite3.Error:
        return 0            # no column yet is no personal rows
    finally:
        conn.close()


def assert_impersonal(db_path):
    """Refuse to call a store shippable if Mati's self-talk is in it.

    ⚠️ TRUE BY CHECK, NOT BY ACCIDENT. "A build is from scratch" is a property
    of HOW the build happens, not of what it produces. Make the build
    incremental, or merge a store, or seed from a copy, and Mati's private
    reflections would land in a public artifact with nothing to say so.

    ⚠️ IT RAISES. A warning printed at the end of a six-minute build is a
    warning nobody sees; the whole point is that this artifact gets published.
    """
    n = personal_rows(db_path)
    if n:
        raise SystemExit(
            f"  REFUSING: {n} personal_only row(s) in {db_path}.\n"
            f"  These are Mati's own private reflections and must never be\n"
            f"  published. A build is from scratch — if these are here, the\n"
            f"  build merged or seeded from an existing store.")
    return n


SELF_CHECK_SOURCE = "self-check"


def self_check(lm, tok, db_path, verbose=True):
    """Mati's first memories: what it MEASURED about itself. How many written.

    ⭐ WHY A SHIPPED STORE NEEDS ANYTHING AT ALL. An empty one means
    `memories_recalled: 0` on every turn of a fresh install, for as long as it
    takes the person to fill it — the field has nothing to settle against and
    recall has nothing to return. The charter ships as text it can quote; this
    ships as the first thing it actually knows.

    ⭐ MEASURED, NOT TOLD, which is the whole distinction. Every line below is
    the answer to a query run against the artefact being built. Nothing here
    is a belief about Mati's role, because that is what the charter already
    is — and a shipped opinion would be one Mati's understanding arriving
    pre-installed in everyone else's head.

    ⚠️⚠️ ONLY FACTS TRUE OF THE SHIPPED ARTEFACT. These rows are IDENTICAL on
    every install, so anything true only of the build machine is a lie
    everywhere else. Deliberately absent: the model name (configurable — a
    person running a different one would hold a false memory about their own
    voice), machine specs, paths, dates, and anything about the person, who
    does not exist yet.

    ⚠️ SOURCE IS `self-check`, NOT `episode`. An episode body opens "I was
    asked:" and nobody asked — inventing an exchange is exactly the
    fabrication the provenance labels exist to prevent. It is also not `mine`:
    self-talk is refused from a shipped build by `assert_impersonal`, and
    rightly, but a measurement is not introspection. The label is named in
    `bridge.LABEL_LEGEND` so Mati reads these as checks rather than
    experiences.
    """
    import sqlite3 as _sq
    # ⚠️ IMPORTED HERE BECAUSE THIS MODULE HAS NO MODULE-LEVEL `bridge`.
    # `build()` does `from . import bridge, …` inside itself, so a new function
    # referencing `bridge` would compile fine and raise NameError on first
    # call — and the loop's `except` reports only under `verbose`.
    from . import bridge

    conn = _sq.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        lived = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE source = 'episode'").fetchone()[0]
        charter = conn.execute(
            "SELECT COUNT(*) FROM notes "
            "WHERE source LIKE 'studied:charter:%'").fetchone()[0]
        personal = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE personal_only = 1").fetchone()[0]
    except _sq.Error:
        lived = charter = personal = 0
    finally:
        conn.close()

    facts = [
        f"I checked how much I remember. I hold {lived} lived "
        f"conversation{'' if lived == 1 else 's'} and "
        f"{charter} passage{'' if charter == 1 else 's'} of my own charter, "
        f"studied word for word. The charter is text I was given; the "
        f"conversations are mine to have, and I have not had them yet.",

        f"I checked whether I am carrying anyone's private thinking. "
        f"{personal} of my memories are marked personal, which is what I "
        f"expect: reflections belong to the Mati that had them and do not "
        f"travel into a build.",

        "I checked what I can reach on my own. Nothing: no reminders are "
        "set, no reading is queued, and every feature that needs a key is "
        "off because I ship without keys. What I do next depends entirely "
        "on being asked.",

        "I checked where my memory lives. Each thing I know has an address "
        "in my field and a body on disk, and the two are kept apart on "
        "purpose: the address stays readable so I can find a memory, and the "
        "body is sealed so nobody else can read it.",
    ]

    written = 0
    for body in facts:
        try:
            res = bridge.remember(lm, tok, body, source=SELF_CHECK_SOURCE,
                                  db_path=db_path, verbatim=body)
        except Exception as e:                                  # noqa: BLE001
            if verbose:
                print(f"    self-check failed: {e}", flush=True)
            break
        if res:
            written += 1
            if verbose:
                print(f"    #{res[0]} {body[:58]}…", flush=True)
    return written


# ⚠️⚠️ A SHIPPED .db WITH A WRITE-AHEAD LOG BESIDE IT IS A TRAP. SQLite keeps
# committed rows in `-wal` that are not yet in the `.db`, so whoever packages
# this directory and copies `mati_notes.db` alone gets a store that looks
# fine and is quietly older — possibly missing the self-check rows, which are
# written last.
#
# ⚠️ AN EMPTY `-wal` IS TIMING, NOT CORRECTNESS: it is empty only because each
# connection happened to checkpoint as it closed. `backup.py` carries the same
# lesson — *"leaving a newer log beside an older database is the same bug
# pointing the other way"*.


def seal_db(db_path, verbose=False):
    """Make the artefact a plain, self-contained, read-anywhere SQLite file.

    Idempotent, so it can be run against an already-built artefact.

    ⚠️⚠️ CHECKPOINTING ALONE IS NOT ENOUGH, AND MAKES THINGS WORSE. Folding
    the log in leaves the database still in WAL *mode* — the pragma lives in
    the file header — and a WAL database cannot be opened READ-ONLY unless its
    `-shm` index already exists. Sealing removes `-shm`, so immediately after
    a successful checkpoint:

        PRAGMA journal_mode  -> wal
        sqlite3 file:...?mode=ro  -> "unable to open database file"

    `check()` opens `mode=ro`, so `--check` would report `readable: False` on a
    database that is perfectly intact — and True again as soon as anything
    opened it read-write and recreated `-shm`. A shipped artefact whose
    readability depends on a scratch file is not sealed.

    ⚠️ SO THE JOURNAL MODE IS CHANGED, NOT JUST EMPTIED. `DELETE` is the
    rollback journal, persists in the header, and needs no sidecar. The
    install flips its own copy back to WAL when it opens it for writing; this
    is the format it TRAVELS in, which may be a read-only disk or a zip.
    """
    if not os.path.exists(db_path):
        return False
    conn = sqlite3.connect(db_path)
    try:
        # TRUNCATE folds the log back in AND empties it.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        # ⚠️ This is what actually removes the `-wal` and the WAL dependency.
        mode = conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        conn.commit()
        if str(mode).lower() == "wal":
            print(f"  ⚠️ {db_path} would not leave WAL mode — something has "
                  f"it open. Do NOT ship it.")
            return False
    except sqlite3.Error as e:
        print(f"  ⚠️ could not seal {db_path}: {e}")
        return False
    finally:
        conn.close()

    for side, why in (("-wal", "write-ahead log"), ("-shm", "scratch index")):
        f = db_path + side
        if os.path.exists(f):
            # ⚠️ A `-wal` with bytes in it would be data, but leaving WAL mode
            # above already folded it in; if one survives, say so rather than
            # delete it.
            if side == "-wal" and os.path.getsize(f) > 0:
                print(f"  ⚠️ {os.path.basename(f)} still holds "
                      f"{os.path.getsize(f)} bytes. Do NOT ship the .db alone.")
                return False
            os.remove(f)
            if verbose:
                print(f"    removed {os.path.basename(f)} ({why})")

    # ⭐ THE SUCCESS CRITERION IS A READ-ONLY OPEN, not the absence of files.
    # That is the property a packager, a zip, and `check()` all depend on.
    try:
        ro = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            n = ro.execute("SELECT count(*) FROM notes").fetchone()[0]
        finally:
            ro.close()
    except sqlite3.Error as e:
        print(f"  ⚠️ sealed but NOT readable read-only: {e}")
        return False
    if verbose:
        print(f"    read-only open verified: {n} rows, journal_mode=delete")
    return True


def build(out_dir, verbose=True, max_chunks=None):
    """Make the shippable set. Returns a summary dict."""
    from . import bridge, config as C, study, vault

    src = os.path.join(C.ROOT, CHARTER)
    if not os.path.exists(src):
        raise SystemExit(f"  no charter at {src}")

    os.makedirs(out_dir, exist_ok=True)
    p = _paths(out_dir)
    for f in p.values():
        if os.path.exists(f):
            os.remove(f)                # a build is from scratch, never a merge

    # ⚠️ EVERY PATH IS REDIRECTED BEFORE A FIELD EXISTS. `load_field` pins the
    # address space and WRITES it when none is there, so a load before this
    # line would seed the real cache from a build.
    # ⚠️⚠️ `field.notes` HAS ITS OWN GLOBAL AND `save_field` READS THAT ONE.
    # Redirecting only bridge's copy would have a build WRITE OVER THE
    # PERSON'S PRODUCTION θ₀ — once per batch — while the build directory's own
    # theta file stayed empty, and the log line "Notes θ₀ saved →
    # .cache/mati_notes_theta0.npz" would be the only sign. It is enforced here
    # rather than remembered.
    from .field import notes as _fn
    saved = (bridge.NOTES_DB_PATH, bridge.NOTES_THETA0_PATH, vault.VAULT_PATH,
             _fn.NOTES_DB_PATH, _fn.NOTES_THETA0_PATH)
    bridge.NOTES_DB_PATH = _fn.NOTES_DB_PATH = p["db"]
    bridge.NOTES_THETA0_PATH = _fn.NOTES_THETA0_PATH = p["theta"]
    # ⚠️ NO VAULT. The shipped store must be readable by an install that has
    # not made a key yet — setup makes the key and encrypts what is already
    # there, which is exactly this. Pointing at a non-existent vault keeps
    # `vault.seal` in its plaintext path.
    vault.VAULT_PATH = os.path.join(out_dir, "no_vault.json")
    try:
        from . import notes as _n
        _n.forget_bodies()
        bridge.thaw()

        # ⚠️ PROVE THE REDIRECT TOOK, BEFORE ANYTHING CAN WRITE. A build that
        # silently writes to the real cache overwrites someone's field; a build
        # that refuses to start is an inconvenience.
        import os.path as _op
        real = _op.join(C.CACHE_DIR, "mati_notes_theta0.npz")
        for name, got in (("bridge", bridge.NOTES_THETA0_PATH),
                          ("field.notes", _fn.NOTES_THETA0_PATH)):
            if _op.abspath(got) == _op.abspath(real):
                raise SystemExit(f"  {name} still points at the real cache — "
                                 f"refusing to build")

        t0 = time.time()
        lm, tok = bridge.load_field()
        if verbose:
            print(f"  loaded baseline θ₀ from {os.path.basename(lm._theta_source)}"
                  f"   [{time.time()-t0:.1f}s]")

        text = study.read_file(src)
        # ⚠️⚠️ VERBATIM, NOT STUDIED. `study.enqueue` + `study.study` stores
        # what a 3B LEARNED from each chunk, not the chunk — which is right
        # for a book and wrong for a charter, because the `[charter]` label
        # tells Mati to speak these rows AS ITS OWN. Studied, most of the
        # charter comes out as paraphrase, and the paraphrases are not harmless.
        # Shipped as charter memories, they would read:
        #   "I don't retain any information if the conversation doesn't
        #    conclude with a clear resolution"   — a FALSE memory denial
        #   "There's no concept of fields holding shapes or rules"
        #                                        — contradicts the architecture
        #   "I'm told that all information ... stays on the device itself"
        #                                        — the privacy claim, as hearsay
        #   "You're recalling someone you've known for a long time ..."
        #                                        — second person, which becomes
        #                                          a claim about the PERSON
        #
        # ⭐ IT ALSO KEEPS THE GATE WORKING. Imprinted chunks of one voice drive
        # cold-start coherence below `EPISODE_GATE_MIN`, so a fresh install
        # would reject every episode. Verbatim rows leave θ₀ untouched.
        #
        # ⚠️ THE `studied:` PREFIX IS LOAD-BEARING. `bridge.named_recall_block`
        # reaches the charter branch only via `source.startswith("studied:")`:
        # `studied:charter:FILE:1` renders `[charter]`, while a bare
        # `charter:FILE:1` renders `[charter:MATI_MISSION_AND_PRINCIPLES.txt:1]`
        # — the filename leaking into the label Mati reads.
        # ⚠️ `--chunks` TRUNCATES RATHER THAN SAMPLES, because the first
        # chunks are the identity-dense ones: the charter opens "I am Mati...
        # the language I speak comes from a language model, the memory I hold
        # comes from the field, and I keep them apart". It is a debugging
        # convenience rather than a dose: nothing is imprinted, so the full
        # charter costs the field nothing.
        if max_chunks:
            whole = study.chunk_text(text, tok)
            text = " ".join(whole[:max_chunks])
        n_charter = write_charter_verbatim(
            lm, tok, text, f"studied:{ORIGIN}", p["db"], verbose=verbose)

        # ⚠️ PROMOTE THEN REINDEX, in that order. The notes were addressed
        # against the space as it was when each was written; imprinting moved
        # the live field underneath them. Promotion publishes the field they
        # must be filed in, and the reindex re-files them there.
        # ⚠️ WRITTEN BEFORE PROMOTE AND REINDEX, DELIBERATELY. These notes
        # must be addressed against the θ₀ that actually SHIPS. Written after
        # promotion they would be filed in a space the install never sees,
        # and every `feat` would be a distance between two different frames —
        # not a worse match, a meaningless one.
        if verbose:
            print("\n  first memories — what it measured about itself:",
                  flush=True)
        n_self = self_check(lm, tok, p["db"], verbose=verbose)

        bridge.promote(lm)
        bridge.reindex(lm, tok, limit=0)
        bridge.save_notes_field(lm)

        # ⚠️ BEFORE THE SUMMARY SAYS IT WORKED. See `assert_impersonal`.
        assert_impersonal(p["db"])
        summary = check(p["db"])
        summary["impersonal"] = True
        summary["self_check"] = n_self
        summary["charter_verbatim"] = n_charter
        summary["seconds"] = round(time.time() - t0, 1)
        # ⚠️ AFTER the reads above, because opening a WAL database to read it
        # recreates the sidecars — sealing before `check` would undo itself.
        summary["sealed"] = seal_db(p["db"], verbose=verbose)
        summary["files"] = {k: (os.path.getsize(v) if os.path.exists(v) else 0)
                            for k, v in p.items()}
        return summary
    finally:
        bridge.thaw()
        (bridge.NOTES_DB_PATH, bridge.NOTES_THETA0_PATH, vault.VAULT_PATH,
         _fn.NOTES_DB_PATH, _fn.NOTES_THETA0_PATH) = saved


# ── publishing ──────────────────────────────────────────────────────────────
# ⚠️ A BUILD NOBODY CAN FIND IS NOT SHIPPED. `.cache/ship/<version>/` is
# ignored by git, so an artefact left only there exists only on the machine
# that built it, and a fresh clone could not run the export.
#
# ⭐ So the build PUBLISHES to fixed, tracked names in `.cache/`. The versioned
# directory stays as the build output, which is what lets two be compared;
# these three are "the ones that ship", and the export carries them like any
# other runtime file — no rename at copy time, one source of truth.
PUBLISHED = [("mati_notes.db", "first_memories.db"),
             ("mati_notes_theta0.npz", "first_memories_theta0.npz"),
             ("mati_notes_theta0_address.npz",
              "first_memories_theta0_address.npz")]


def publish(out_dir, verbose=True):
    """Copy a built set to the tracked names the export ships. Idempotent."""
    from . import config as C
    done = []
    for src_name, dst_name in PUBLISHED:
        src = os.path.join(out_dir, src_name)
        if not os.path.exists(src):
            raise SystemExit(f"  cannot publish: {src} is missing")
        dst = os.path.join(C.CACHE_DIR, dst_name)
        shutil.copy2(src, dst)
        done.append(dst_name)
    if verbose:
        print(f"  published as {', '.join(done)}")
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="where to build")
    ap.add_argument("--version", default="v1",
                    help="subfolder under .cache/ship/ (default v1)")
    ap.add_argument("--chunks", type=int, default=None,
                    help="study only the first N chunks of the charter")
    ap.add_argument("--check", action="store_true",
                    help="report on a built set, build nothing")
    a = ap.parse_args(argv)
    from . import config as C
    # ⚠️ VERSIONED: each build is its own folder, so a shipped set is never
    # silently replaced by the next attempt and two can be compared.
    out = a.out or os.path.join(C.CACHE_DIR, "ship", a.version)

    if a.check:
        got = check(_paths(out)["db"])
        print(f"  {out}")
        for k, v in got.items():
            print(f"    {k}: {v}")
        return 0

    print(f"  building into {out}\n")
    got = build(out, max_chunks=a.chunks)
    # ⚠️ ONLY AFTER `assert_impersonal` INSIDE `build` HAS PASSED. Publishing
    # is what makes the artefact reachable by the export, so it must never run
    # on a set that failed its own guard.
    got["published"] = publish(out)
    print()
    print(f"  by source : {got.get('by_source')}")
    print(f"  addressed : {got.get('addressed')}")
    print(f"  seconds   : {got.get('seconds')}")
    for k, v in (got.get("files") or {}).items():
        print(f"  {k:9} : {v:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
