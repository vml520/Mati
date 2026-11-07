"""
What this Mac can actually do, measured before anything is promised.

    python3 -m Mati.machine            what this machine can run

⚠️ THIS EXISTS SO NOTHING LATER HAS TO APOLOGISE. Step 0 of setup:
the checks are "not preferences — whether Mati can run at all, stated up
front rather than discovered halfway through." A 4 GB Mac should be told, not
left to find out by being slow.

⚠️ AND IT CHOOSES THE AVATAR DEFAULT. The user: *"Avatar should be set depending
on machine specs; off on computers like mine."* The animation is a React
component rendering every frame beside a 3B model and a phase field — on 8 GB
that is real contention, so the default is derived here rather than fixed.

⚠️ NO psutil, NO DEPENDENCIES. `sysctl` and `shutil` are already on the
machine; a spec check that needs installing before it can tell you whether you
can install is not a spec check.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys

# ⚠️ 8 GB IS THE FLOOR, NOT A SUGGESTION — it is the number the README and the
# release plan both commit to. Below it Mati runs; it just runs badly, and
# saying so is the point of this module.
RAM_FLOOR_GB = 8.0
RAM_COMFORTABLE_GB = 16.0
# The checkpoint, the vocab and room for the store to grow.
DISK_NEEDED_GB = 2.0


def _sysctl(key):
    try:
        out = subprocess.run(["/usr/sbin/sysctl", "-n", key],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def ram_gb():
    """Physical memory in GB, or None if it cannot be read."""
    raw = _sysctl("hw.memsize")
    try:
        return round(int(raw) / (1024 ** 3), 1) if raw else None
    except (TypeError, ValueError):
        return None


def cpu_name():
    return _sysctl("machdep.cpu.brand_string") or platform.processor() or "?"


def apple_silicon():
    """⚠️ `platform.machine()` reports x86_64 under Rosetta."""
    if platform.machine() == "arm64":
        return True
    return _sysctl("sysctl.proc_translated") == "1"


def free_disk_gb(path=None):
    from . import config as C
    try:
        return round(shutil.disk_usage(path or C.CACHE_DIR).free / (1024 ** 3), 1)
    except Exception:
        return None


def ollama_ready():
    """(reachable, [model names]). ⚠️ Not 'installed' — reachable.

    ⚠️ `setup._pulled_models()` RETURNS `([(name, size), ...], error)` — a pair,
    where the first element is a list of TUPLES. My first version treated it as
    a list of dicts and called `.get("name")`, which raised, was swallowed, and
    reported "Ollama is not reachable" on a machine that had just served a
    reply. A false blocker in Step 0 stops setup dead on a working install,
    which is worse than the missing check it replaced.
    """
    try:
        from . import setup as _s
        pulled, err = _s._pulled_models()
    except Exception:
        return (False, [])
    if err and not pulled:
        return (False, [])
    names = []
    for m in pulled or []:
        # Tolerate either shape rather than assume the one seen today.
        if isinstance(m, (tuple, list)) and m:
            names.append(str(m[0]))
        elif isinstance(m, dict) and m.get("name"):
            names.append(str(m["name"]))
        elif isinstance(m, str):
            names.append(m)
    return (True, names)


def profile():
    """Everything Step 0 needs, as one dict."""
    ram = ram_gb()
    reachable, models = ollama_ready()
    return {
        "macos": platform.system() == "Darwin",
        "release": platform.mac_ver()[0] or None,
        "arm": apple_silicon(),
        "cpu": cpu_name(),
        "ram_gb": ram,
        "disk_gb": free_disk_gb(),
        "ollama": reachable,
        "models": models,
    }


def avatar_default(p=None):
    """`on` | `still` | `off`, chosen from the machine.

    ⚠️ THE THREE TIERS ARE NOT ARBITRARY, and they are deliberately cautious
    on the boundary. The animation competes with the model for the same cores:

        < 8 GB or Intel        off     the model is already the problem
        8-16 GB Apple silicon  still   art without a render loop
        16 GB+ Apple silicon   on

    ⚠️ Intel is `off` at ANY memory size. The user runs an Intel Mac and asked
    for off on "computers like mine"; the 3B is 55-102s to a complete reply
    there, and spending frames on an axolotl during that is the wrong trade.
    """
    p = p or profile()
    ram = p.get("ram_gb") or 0
    if not p.get("arm") or ram < RAM_FLOOR_GB:
        return "off"
    if ram < RAM_COMFORTABLE_GB:
        return "still"
    return "on"


def blockers(p=None):
    """What would stop Mati running. [] means it will run."""
    p = p or profile()
    out = []
    if not p["macos"]:
        out.append("Mati runs on macOS only. This is "
                   f"{platform.system()}.")
    if not p["ollama"]:
        out.append("Ollama is not reachable. Install it from "
                   "https://ollama.com, then: ollama pull llama3.2:3b")
    elif not p["models"]:
        out.append("Ollama is running but no model is pulled. Try: "
                   "ollama pull llama3.2:3b")
    if p["disk_gb"] is not None and p["disk_gb"] < DISK_NEEDED_GB:
        out.append(f"{DISK_NEEDED_GB} GB of free space is needed; "
                   f"{p['disk_gb']} GB is free.")
    return out


def warnings(p=None):
    """What will work but will disappoint. Said BEFORE, never after."""
    p = p or profile()
    out = []
    ram = p.get("ram_gb")
    if ram is not None and ram < RAM_FLOOR_GB:
        out.append(f"{ram} GB of memory is below the {RAM_FLOOR_GB:.0f} GB "
                   "floor. A 1B model is the realistic choice here, and "
                   "replies will still be slow.")
    elif ram is not None and not p["arm"]:
        # ⚠️ THE MEASURED NUMBER, not a hedge. This is the honest-latency
        # moment the spec asks for, and it is the one on the release plan.
        out.append("On an Intel Mac a 3B model takes roughly 55-102 seconds "
                   "to finish a reply. That is normal here, not a fault.")
    return out


def describe(p=None):
    p = p or profile()
    chip = "Apple silicon" if p["arm"] else "Intel"
    ram = f"{p['ram_gb']} GB" if p["ram_gb"] else "? GB"
    disk = f"{p['disk_gb']} GB free" if p["disk_gb"] is not None else "?"
    lines = [
        f"  {chip}, {ram} memory, {disk}",
        f"  {p['cpu']}",
        f"  macOS {p['release'] or '?'}",
        f"  Ollama {'reachable' if p['ollama'] else 'NOT reachable'}"
        + (f", {len(p['models'])} model(s): {', '.join(p['models'][:4])}"
           if p["models"] else ""),
        f"  avatar default: {avatar_default(p)}",
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    p = profile()
    if a.json:
        import json
        print(json.dumps({**p, "avatar_default": avatar_default(p),
                          "blockers": blockers(p), "warnings": warnings(p)},
                         indent=2))
        return 0
    print("\n  This machine\n")
    print(describe(p))
    for b in blockers(p):
        print(f"\n  ⚠️ {b}")
    for w in warnings(p):
        print(f"\n  {w}")
    if not blockers(p):
        print("\n  Mati will run here.\n")
        return 0
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
