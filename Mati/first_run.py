"""Put Mati's first memories in place, once, on a brand-new install.

⚠️⚠️ WHY THE SHIPPED STORE IS NOT SIMPLY NAMED `mati_notes.db`. Two reasons,
and both are the kind that bite later:

  1. `.gitignore` UNTRACKS THE PERSONAL FIELD ON PURPOSE. `mati_notes.db` is
     one person's conversations; the rule exists because a fresh clone once
     arrived carrying somebody else's memories. Re-admitting that name so the
     artefact could ship under it would have started tracking the developer's
     own store — 6,669 real memories — in the repository the rename existed to
     clean.

  2. A TRACKED FILE AT THE LIVE PATH BECOMES THE PERSON'S OWN STORE. They
     clone, they talk to Mati, and their memories accumulate inside a file
     git is managing. The next `git pull` has an opinion about their
     conversations. Nothing good is downstream of that.

⭐ So the build ships `first_memories.db` and its θ₀, and this module copies
them into place when — and only when — there is no store yet. The live store
stays untracked and personal, which is what the ignore rule is protecting.

⚠️ COPY, NEVER MOVE. The shipped files stay where they are: a `git pull` must
not find them missing, and a person who deletes their store to start over
should get the first memories back rather than an empty field.
"""

from __future__ import annotations

import os
import shutil

# (shipped name, the name Mati reads). The address space travels with the
# store: `feat` is a distance measured against a particular θ₀, so a store
# seeded without its own θ₀ is filed in a frame the install does not have.
SEEDS = [("first_memories.db", "mati_notes.db"),
         ("first_memories_theta0.npz", "mati_notes_theta0.npz"),
         ("first_memories_theta0_address.npz",
          "mati_notes_theta0_address.npz")]


def seed(cache_dir, verbose=True):
    """Copy the shipped first memories into place. Returns what it wrote.

    ⚠️ IT DOES NOTHING IF A STORE ALREADY EXISTS, and the store is the only
    thing it checks. Deciding per-file would let a half-seeded install happen:
    a θ₀ copied beside somebody's real database would re-file every address
    they have against a frame that is not theirs.
    """
    live_db = os.path.join(cache_dir, "mati_notes.db")
    if os.path.exists(live_db):
        return []
    shipped = os.path.join(cache_dir, SEEDS[0][0])
    if not os.path.exists(shipped):
        # Nothing shipped — a build without the artefact, or a developer
        # tree. An empty first turn is the correct outcome, not an error.
        return []

    wrote = []
    for src_name, dst_name in SEEDS:
        src = os.path.join(cache_dir, src_name)
        dst = os.path.join(cache_dir, dst_name)
        if not os.path.exists(src) or os.path.exists(dst):
            continue
        try:
            shutil.copy2(src, dst)
            wrote.append(dst_name)
        except OSError as e:
            # ⚠️ LOUD, AND IT LEAVES WHAT IT MANAGED. A partially seeded
            # install is worth saying out loud; silently continuing is how it
            # would be discovered as "Mati knows nothing about itself".
            print(f"  [mati] ⚠️ could not place {dst_name}: {e}")
    if verbose and wrote:
        print(f"  [mati] first memories placed ({len(wrote)} files)")
    return wrote
