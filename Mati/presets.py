"""
Ten station buttons, like a car radio.

The music service is a car radio with station presets, not a streaming app
with a music library — and it never picks a random station.

⚠️ A RANDOM STATION IS NOT THE DIRECTORY'S FAULT. Searching by genre and
taking the top hit changes the station whenever the click counts do, and
nothing in that stores *your* stations. A different directory would have the
same problem; a preset is the fix.

⚠️ THE RESOLVED STREAM URL IS WHAT GETS PINNED, not the station's name or its
directory id. Radio Browser is community-edited: names get retyped, entries get
merged, ids churn. The URL is the thing that still plays tomorrow. The name is
kept too, but only to say out loud.

⚠️ TEN SLOTS, FIXED. Not "ten by default" — a car radio has the buttons it has,
and a preset you have to scroll to find is a search result wearing a number.
Saving to a full radio replaces a slot you name, and Mati says which station
it displaced rather than discarding one quietly.

⚠️ CLEAR, NOT SEALED. A station name and a public stream URL are not confidant
material — this is the dial position, not what was said. It follows `agenda`'s
split rather than the accountant's: nothing here needs the vault to be read, so
presets survive a locked field and the radio still works before Touch ID.
"""

from __future__ import annotations

import sqlite3

from . import bridge
from . import vault

TABLE = "radio_presets"
SLOTS = 10


def _conn(db_path=None):
    return sqlite3.connect(db_path or bridge.NOTES_DB_PATH, timeout=30)


def ensure(db_path=None):
    """Create the table. Idempotent; safe on every entry point."""
    conn = _conn(db_path)
    try:
        # ⚠️ `slot` IS THE PRIMARY KEY, so a save to an occupied button
        # replaces it by construction rather than by a SELECT-then-UPDATE that
        # could race two requests into two rows for button 3.
        conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                slot     INTEGER PRIMARY KEY,
                name     TEXT NOT NULL,
                url      TEXT NOT NULL,
                bitrate  INTEGER,
                codec    TEXT,
                country  TEXT,
                saved_at TEXT
            )""")
        conn.commit()
    finally:
        conn.close()


def _valid(slot):
    try:
        n = int(slot)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= SLOTS else None


def save(slot, station, db_path=None, now=None):
    """Pin a station to a button. (ok, message, displaced-or-None).

    ⚠️ RETURNS WHAT IT REPLACED. Overwriting button 3 silently is how you lose
    a station you cannot name any more — the directory has thousands and you
    found this one once.
    """
    n = _valid(slot)
    if n is None:
        return False, f"presets are 1 to {SLOTS}", None
    url = (station or {}).get("url")
    name = (station or {}).get("name")
    if not url or not name:
        return False, "that is not a station I can pin", None

    ensure(db_path)
    before = recall(n, db_path=db_path)
    conn = _conn(db_path)
    try:
        conn.execute(
            f"INSERT OR REPLACE INTO {TABLE} "
            f"(slot, name, url, bitrate, codec, country, saved_at) "
            f"VALUES (?,?,?,?,?,?,?)",
            # ⚠️ NAME AND URL ARE SEALED. Which five stations a person
            # pinned is their taste, and a stream URL identifies exactly what
            # they listen to. Safe to seal because this table is only ever
            # queried by `slot`, an integer primary key — nothing compares
            # these two, so nondeterministic sealing costs nothing.
            #
            # ⚠️ `codec` and `country` stay CLEAR: "MP3" and a country from
            # the station directory are facts about the STREAM, not about the
            # listener, and keeping them readable keeps `bitrate` usable.
            (n, vault.seal(name), vault.seal(url), station.get("bitrate"),
             station.get("codec"), station.get("country"),
             now or bridge._now()))
        conn.commit()
    finally:
        conn.close()

    msg = f"{name} is preset {n}"
    if before and before["url"] != url:
        msg += f", replacing {before['name']}"
    return True, msg, before


def recall(slot, db_path=None):
    """What is on a button, or None."""
    n = _valid(slot)
    if n is None:
        return None
    ensure(db_path)
    conn = _conn(db_path)
    try:
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            f"SELECT slot, name, url, bitrate, codec, country, saved_at "
            f"FROM {TABLE} WHERE slot=?", (n,)).fetchone()
    finally:
        conn.close()
    return _open(dict(r)) if r else None


def _open(row):
    """Unseal a preset row.

    ⚠️ BOTH READERS GO THROUGH HERE ON PURPOSE. Sealing a column and then
    forgetting one of two SELECTs is how a feature half-breaks, and
    `all_presets` is the one the app's buttons are built from.
    """
    for k in ("name", "url"):
        if row.get(k):
            try:
                row[k] = vault.open_body(row[k])
            except Exception:
                # ⚠️ A LOCKED VAULT MUST NOT INVENT A STATION. Handing back
                # the ciphertext would put base64 on a button, and guessing
                # would play the wrong thing.
                row[k] = None
    return row


def all_presets(db_path=None):
    """Every button that has something on it, in slot order.

    ⚠️ RETURNS ONLY WHAT IS SET. An empty radio returns `[]` — not ten blanks.
    "No data" and "nothing set" read the same to a caller that counts rows, and
    `accountant.spent` is the precedent for keeping them distinguishable.
    """
    ensure(db_path)
    conn = _conn(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT slot, name, url, bitrate, codec, country, saved_at "
            f"FROM {TABLE} ORDER BY slot").fetchall()
    finally:
        conn.close()
    return [_open(dict(r)) for r in rows]


def clear(slot, db_path=None):
    """Empty a button. Returns what was on it, or None."""
    n = _valid(slot)
    if n is None:
        return None
    was = recall(n, db_path=db_path)
    if was is None:
        return None
    conn = _conn(db_path)
    try:
        conn.execute(f"DELETE FROM {TABLE} WHERE slot=?", (n,))
        conn.commit()
    finally:
        conn.close()
    return was


def free_slot(db_path=None):
    """The lowest empty button, or None when all ten are taken."""
    taken = {p["slot"] for p in all_presets(db_path=db_path)}
    for n in range(1, SLOTS + 1):
        if n not in taken:
            return n
    return None


def describe(db_path=None):
    """For `/system` and the setup script."""
    return {"slots": SLOTS, "set": len(all_presets(db_path=db_path))}
