"""
Reaching the person when no conversation is open.

⚠️ WITHOUT THIS, A DUE REMINDER ARRIVES ONLY WHEN A CONVERSATION STARTS. The
idle pass checks consent and quiet hours and publishes an `announce` into the
telemetry stream, which nothing renders, and `mark_delivered` is otherwise
called only from the turn path and from `inquiry`. That is the case with
teeth: rent due on the 1st, told on the 3rd.

⚠️ THIS DOES NOT WIDEN WHAT MAY INTERRUPT. `schedule.may_interrupt` still
decides, per kind, with quiet hours overriding consent. Nothing becomes
permitted that was not already permitted — what this adds is that a permitted
item actually arrives.

⚠️ osascript, NOT UserNotifications. `UNUserNotificationCenter` instantiates
from unbundled Python but delivery needs a bundle identifier, which Mati does
not have until it is packaged. `/usr/bin/osascript` ships with macOS, works
from a detached process, and adds no dependency.

⚠️ THE BODY IS PASSED AS AN ARGUMENT, NEVER INTERPOLATED. A reminder body is
text a language model wrote or a person typed; building an AppleScript string
around it is an injection waiting to happen. `on run argv` keeps it data.

⚠️ A NOTIFICATION IS VISIBLE ON A LOCKED SCREEN. Reminder bodies can be
personal and payment reminders will be. The title is only the assistant's
name and the body is what the person themselves asked to be reminded of — but
this is the one place Mati's contents leave the encrypted store and appear on
glass, and that is worth knowing rather than discovering.
"""

from __future__ import annotations

import subprocess
import sys

# A notification banner truncates anyway; sending more is noise in the log.
MAX_BODY = 200

# The title is the assistant's name — the one this person gave it, read when
# the notification is sent.


def available():
    """Can this machine show a notification at all?"""
    import os
    return os.path.exists("/usr/bin/osascript")


def send(body, subtitle=None, title=None, timeout=10):
    """Post one notification. Returns True if it was handed off.

    ⚠️ Returns False rather than raising. A notification that cannot be shown
    must never cost the caller anything — the item stays queued and the person
    is told at the top of the next conversation, as when notifications are
    off.
    """
    body = " ".join((body or "").split())[:MAX_BODY]
    if not body or not available():
        return False
    if not title:
        from . import config as C
        title = C.assistant_name()
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
