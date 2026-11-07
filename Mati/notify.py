"""
Reaching the person when no conversation is open.

⚠️ `_maybe_announce` PUBLISHED INTO A STREAM NOBODY CONSUMED. It ran on every
idle pass, checked consent and quiet hours correctly, and then called
`telemetry.publish(announce=…)` — and nothing rendered it. Checked :
`announce` appeared in no consumer anywhere, and `mark_delivered` is
called only from the turn path and from `inquiry`. So a due reminder reached
the person **only when they started a conversation**.

That is the fifth "built, correct, and never read" in this codebase, after the
activity log, the library catalogue, the book title and the morning report's
six sections. It is also the one with teeth: rent due on the 1st, told on the
3rd.

⚠️ THIS DOES NOT WIDEN WHAT MAY INTERRUPT. `schedule.may_interrupt` still
decides, per kind, with quiet hours overriding consent. Nothing becomes
permitted that was not already permitted — what changes is that a permitted
item now actually arrives.

⚠️ osascript, NOT UserNotifications. `UNUserNotificationCenter` instantiates
from unbundled Python but delivery needs a bundle identifier, which Mati does
not have until it is packaged. `/usr/bin/osascript` ships with macOS, works
from a detached process (verified), and adds no dependency.

⚠️ THE BODY IS PASSED AS AN ARGUMENT, NEVER INTERPOLATED. A reminder body is
text a language model wrote or a person typed; building an AppleScript string
around it is an injection waiting to happen. `on run argv` keeps it data.

⚠️ A NOTIFICATION IS VISIBLE ON A LOCKED SCREEN. Reminder bodies can be
personal and payment reminders will be. The title is always just "Mati" and
the body is what the person themselves asked to be reminded of — but this is
the one place Mati's contents leave the encrypted store and appear on glass,
and that is worth knowing rather than discovering.
"""

from __future__ import annotations

import subprocess
import sys

# A notification banner truncates anyway; sending more is noise in the log.
MAX_BODY = 200

TITLE = "Mati"


def available():
    """Can this machine show a notification at all?"""
    import os
    return os.path.exists("/usr/bin/osascript")


def send(body, subtitle=None, title=TITLE, timeout=10):
    """Post one notification. Returns True if it was handed off.

    ⚠️ Returns False rather than raising. A notification that cannot be shown
    must never cost the caller anything — the item stays queued and the person
    is told at the top of the next conversation, which is the behaviour that
    existed before this module.
    """
    body = " ".join((body or "").split())[:MAX_BODY]
    if not body or not available():
        return False
    script = ["on run argv",
              "display notification (item 1 of argv) with title (item 2 of argv)"
              + (" subtitle (item 3 of argv)" if subtitle else ""),
              "end run"]
    args = ["/usr/bin/osascript"]
    for line in script:
        args += ["-e", line]
    args += ["--", body, title]
    if subtitle:
        args.append(" ".join(str(subtitle).split())[:80])
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout)
        return r.returncode == 0
    except Exception as e:                                   # noqa: BLE001
        print(f"  [notify] {e}", file=sys.stderr)
        return False
