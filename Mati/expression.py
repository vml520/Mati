"""
Which clip the avatar is playing — one at a time, with transitions.

    from Mati import expression
    expression.read(stage="idle", info={"challenging": {...}})
    # {'name': 'standing', 'clip': 'standing', 'loop': True, ...}
    expression.transition("thinking", "standing")
    # ['thinking-to-standing', 'standing']

⚠️⚠️ SCOPE CHANGED, AND THIS FILE SAID OTHERWISE FOR SIX WEEKS. It described
a set of short video clips, decided  — "**Not a procedurally
animated avatar**" in as many words. The app replaced those seven mp4s with a
procedural animation on  and untracked the video . The
clips are gone; this module's description of them was not, and a reader
(including me) took the stale half as current and concluded the avatar had
been lost.

**WHAT THIS MODULE ACTUALLY DECIDES: WHICH STATE MATI IS IN.** One name, from
the stage and the turn's info. The frontend owns the rendering and maps each
name to a scene in its animation component — that mapping is `SCENE` in the
app's `index.html`, not here, because the scene vocabulary belongs to the art.

**THE LIVE FIELD IS EXPRESSED BY THE FRAME COLOUR ONLY.** Not by the face. See
`signal.py`, which turns coherence into an ambient frame hue. Everything in this
module is *what Mati is doing*, which is discrete and state-shaped; the
continuous state rides entirely on the frame. ⭐ That split still holds and is
the reason the change above cost nothing structural — the face was never the
field's channel.

⚠️ **CONSEQUENCE: THE BASIN → HEAD ANGLE MAPPING IS NOT USED THIS RELEASE.**
`head_angle()` and `BASIN_CALIBRATION` are kept because they are measured and
correct, but nothing renders them. They belong to the animated-avatar version.
Do not describe Mati as expressing its basin through the avatar — it does not.

TWO RULES SURVIVE FROM THE DESIGN, AND STILL DECIDE THINGS.

**ONE CLIP AT A TIME.** A turn can set half a dozen flags at once — corrected
*and* recomputed *and* consolidated. `read()` returns exactly one clip, chosen
from the STAGE alone, so the frontend never invents its own ordering and two
clients never disagree about what Mati is doing. Same reasoning as the colour
signal being computed here.

The flags come back beside it as `also`, all of them, unranked. They used to
each have their own head loop and a PRIORITY list to pick between them; that
list existed only because one slot could not hold a turn that was genuinely two
things at once. Nothing has to lose now — see MOMENTS.

**STILLNESS IS ATTENTION.** `idle` is the only state whose clip wanders; every
other loop is still. This is how animals signal attention and it needs no
explaining — a creature that goes still is listening. It costs nothing to
render, because it is the removal of motion rather than the addition of any.

⚠️ **`attending` IS THE CLIENT'S, NOT THE SERVER'S.** It belongs to the moment
the user starts typing, which the server never witnesses — by the time a turn
arrives the moment has passed. A frontend raises it locally on first keystroke.
Waiting for `/events` would put attention *after* the thing it is attending to.

⚠️ **NO LEGEND, EVER.** If a user has to be told what a clip means, it failed.
`meaning` is for developers and the record; it must not be rendered.
"""

# ── the clips ───────────────────────────────────────────────────────────────
#
# `art` is direction for whoever animates it, not a runtime channel. The
# renderer plays a clip; it does not compose a head from parts. That is the
# whole simplification of this release.
#
# THREE KINDS OF RENDER:
#   head    the animated head, framed as in conversation
#   scene   the avatar is replaced — full figure, underwater
#   absent  nothing at all
#
# Underwater is where the two non-conversational states live, and the "no
# swimming, no floating" rule still holds for every `head` clip. That rule
# exists to keep a CONVERSATION uncluttered, and neither of these is one.
CLIPS = {
    # ── the user is here ────────────────────────────────────────────────────
    "attending": {
        "render": "head", "loop": True, "owner": "client",
        "art": "turned to the user, level, still — eyes on them",
        "meaning": "you have my attention",
    },
    "thinking": {
        "render": "head", "loop": True, "owner": "server",
        "art": "tilted slightly down and away, unfocused",
        "meaning": "working on it",
    },

    # ── the correction pair ─────────────────────────────────────────────────
    #
    # The two halves of one social moment, and the reason any of this exists.
    # Being corrected is when a user spends effort; it has to be acknowledged
    # rather than silently swallowed.

    # ── background ──────────────────────────────────────────────────────────
    #
    # Study, inquiry and reindex share ONE clip. A user does not need to tell
    # "reindexing" from "studying" at a glance, and giving each its own is how
    # an avatar becomes a status bar.
    "away": {
        "render": "head", "loop": True, "owner": "server",
        "art": "turned aside, attention elsewhere",
        "meaning": "busy with something of its own",
    },

    # ── underwater ──────────────────────────────────────────────────────────
    #
    # Both are the full figure in the underwater scene, and they differ by
    # MOTION rather than by expression: dreaming swims, asleep rests. That
    # distinction survives being pre-rendered, and survives a static
    # user-supplied avatar too, because neither uses the avatar image at all.
    "dreaming": {
        "render": "scene", "loop": True, "owner": "server",
        "art": "the swimming loop — moving through the scene",
        "meaning": "dream cycle running; NOT the idle animation",
    },
    "asleep": {
        "render": "scene", "loop": True, "owner": "server",
        "art": "the underwater scene, at rest — settled, gills barely moving, "
               "eyes closed. Water moves; the Axolotl does not.",
        "meaning": "asleep — the field is snapshotted and let go, but Mati is "
                   "still here",
    },

    # ── the two edges ───────────────────────────────────────────────────────
    #
    # SHUTDOWN IS THE ONE WITH NO FACE. Asleep is a rest Mati returns from, so
    # it gets a scene. Shutdown is Mati not running: the process is gone, the
    # field is on disk, and there is nothing to express. Anything drawn here
    # would be inventing a presence that is not there.
    "shutdown": {
        "render": "absent", "loop": False, "owner": "server",
        "art": "nothing. An empty frame.",
        "meaning": "not running — memories snapshotted, safe to copy away",
    },
    # Locked is not shutdown and not asleep: the process is up, waiting for
    # Touch ID. The user must see something to act on.
    "locked": {
        "render": "head", "loop": True, "owner": "server",
        "art": "level, still, eyes closed",
        "meaning": "vault sealed, waiting to be let in",
    },

    # ── the default ─────────────────────────────────────────────────────────
    "idle": {
        "render": "head", "loop": True, "owner": "server", "wander": True,
        "art": "the only clip that moves on its own — a slow, small wander. "
               "Measured: free-running the dynamics drifts the live phases "
               "1.22 rad while θ₀ moves exactly 0.00000000, so this is real "
               "liveness at no cost to memory.",
        "meaning": "awake, nothing in hand",
    },
}

DEFAULT = "idle"

# ⚠️ THERE IS NO LONGER A CLIP PER MOMENT — simplified.
#
# `reckoning`, `standing` and `yielding` were separate head loops for
# computing, disagreeing and deferring. They are all `thinking`: from the
# viewer's side Mati is working on their turn, and three near-identical
# framings of the same pose are three assets to draw, light and match for a
# distinction nobody reads off a face.
#
# THE MEANINGS ARE NOT LOST, they moved layers. The flags still arrive on every
# read as `moments`, and the avatar already carries live state through FRAME
# COLOUR rather than animation. Colour is the right home for "it disagreed":
# it can be present at the same time as the pose instead of replacing it, which
# is what the old PRIORITY list was really working around — it had to pick one
# clip when a turn was genuinely two things at once.
#
# What this costs: 3 loops and 2 transitions — 18 assets down to 13
# (7 loops + 6 transitions). `shutdown` is not counted either way:
# it renders `absent`, so there is nothing to draw.
MOMENTS = ("challenging", "deferred", "computed")

# ── transitions ─────────────────────────────────────────────────────────────
#
# WHY NOT EVERY PAIR. Eleven states is 110 ordered pairs, which is not an asset
# list, it is a reason to abandon the idea.
#
# THE OBSERVATION THAT MAKES IT TRACTABLE: every `head` clip shares the same
# framing — same subject, same distance, same shot. Cutting between two of them
# is a CROSSFADE, which is a real technique and costs no asset at all. A first
# draft of this routed every transition through `idle` with a settle-out and a
# rise-in per state, and came to 32 clips. Crossfading the head loops brings it
# to 18 with nothing lost, because a settle-and-rise between two similar poses
# is motion nobody asked for.
#
# So a bespoke clip is needed in exactly two situations:
#
#   1. THE SHOT CHANGES — head to scene or back. Surfacing and submerging are
#      the transitions a viewer would certainly notice, and a crossfade between
#      a framed head and a full underwater figure reads as a glitch.
#   2. THE MOMENT IS THE POINT. The correction beat has to land off the
#      thinking pose; dissolving through neutral throws away the thing it
#      exists to show.
DIRECT = {
    # 1 — the shot changes.
    ("idle", "asleep"), ("asleep", "idle"),
    ("idle", "dreaming"), ("dreaming", "idle"),
    # Both already underwater: a dissolve within the scene, not a round trip
    # to the surface and back.
    ("asleep", "dreaming"), ("dreaming", "asleep"),
    # ⚠️ "2 — the moment is the point" is gone with the clips it served:
    # ("thinking","standing") and ("thinking","yielding") were bespoke beats
    # off the thinking pose, and there is no longer a pose to beat into.
}

# SHOT CHANGES ROUTE THROUGH `idle`, and only shot changes do.
#
# There are 28 head↔scene pairs and only 4 of them are above. The other 24 are
# unreachable in practice — nothing goes from `standing` straight to `asleep`,
# because sleeping happens from rest; nothing goes from `asleep` straight to
# `thinking`, because waking surfaces first. Rather than commission 24 clips
# for journeys nobody takes, a shot change that has no clip settles to `idle`
# and leaves from there. The head→idle leg is a crossfade, and the idle→scene
# leg already exists, so this costs nothing.
#
# Routing is applied ONLY to shot changes. Head-to-head still crossfades
# directly: making those detour through idle was the 32-clip first draft, and
# it inserted motion nobody asked for between two similar poses.


def missing_transitions():
    """Shot changes that cannot even be routed. Should be empty — a guard.

    A shot change is coverable if it is DIRECT, or if both legs through `idle`
    exist. This fails loudly if someone removes an `idle`↔scene clip, which
    would silently strand every state on the far side of it.
    """
    out = []
    for a, ca in CLIPS.items():
        for b, cb in CLIPS.items():
            if a == b or "absent" in (ca["render"], cb["render"]):
                continue
            if ca["render"] == cb["render"] or (a, b) in DIRECT:
                continue
            # Needs the idle→b leg (a→idle is a crossfade when a is a head,
            # and a DIRECT clip when a is a scene).
            leg_in = (DEFAULT, b) in DIRECT or CLIPS[b]["render"] == "head"
            leg_out = (a, DEFAULT) in DIRECT or ca["render"] == "head"
            if not (leg_in and leg_out):
                out.append((a, b))
    return out


def clip_names():
    """Every clip an animator has to produce."""
    loops = sorted(n for n, c in CLIPS.items() if c["render"] != "absent")
    direct = [f"{a}-to-{b}" for a, b in sorted(DIRECT)]
    return {"loops": loops, "transitions": direct,
            "crossfaded": "every other pair — head loops share framing",
            "total": len(loops) + len(direct)}


def _step(clip, fade):
    return {"clip": clip, "fade": fade}


def transition(frm, to):
    """How to get from one state to another. Ends on the target's loop.

    Returns {"play": [{clip, fade}, …], "clips": [names]}. `fade` on a step
    means crossfade into it; False means cut. Per-step rather than one flag for
    the whole sequence, because a routed shot change genuinely mixes the two:
    surfacing plays a real clip (cut), then settles into a head loop (fade).
    """
    if to not in CLIPS:
        raise ValueError(f"unknown expression {to!r} — have {sorted(CLIPS)}")

    def done(play):
        return {"play": play, "clips": [s["clip"] for s in play]}

    if frm == to or frm is None or frm not in CLIPS:
        return done([_step(to, False)])             # cold start: just begin
    if CLIPS[to]["render"] == "absent":
        return done([_step(to, True)])              # fade to nothing
    if (frm, to) in DIRECT:
        return done([_step(f"{frm}-to-{to}", False), _step(to, False)])

    if CLIPS[frm]["render"] != CLIPS[to]["render"]:
        # A shot change with no clip of its own: settle to idle, leave from
        # there. Each leg uses its own clip when one exists rather than
        # crossfading between two different framings.
        play = []
        if frm != DEFAULT:
            if (frm, DEFAULT) in DIRECT:
                play.append(_step(f"{frm}-to-{DEFAULT}", False))
                play.append(_step(DEFAULT, False))
            else:
                play.append(_step(DEFAULT, True))
        if (DEFAULT, to) in DIRECT:
            play.append(_step(f"{DEFAULT}-to-{to}", False))
            play.append(_step(to, False))
        else:
            play.append(_step(to, True))
        return {**done(play), "via": DEFAULT}

    return done([_step(to, True)])                  # head to head: crossfade


# ── basin → head angle: MEASURED, AND NOT USED THIS RELEASE ─────────────────
#
# Kept because it is correct and was expensive to get right, and because the
# animated-avatar version will want it. ⚠️ Nothing renders it today — the live
# field reaches the user through the frame colour alone.
#
# Measured over the real episode store:
#     w0 0.1225-0.4312 (mean 0.1988)   w1 0.2800-0.5613 (0.4504)
#     w2 0.0972-0.4380 (0.3625)
# ⚠️ Not 0-1: mapped naively the head would never move. Calibrated on MEAN and
# spread rather than min/max, because w0's mean sits below its range midpoint
# and a range mapping left a typical turn rendering the head 6 degrees off to
# one side permanently.
# ⚠️ w1 and w2 correlate +0.950 (n=14, one person's history) — three rotations,
# about two real degrees of freedom, which is why roll's range is small.
BASIN_CALIBRATION = ((0.1988, 0.2218), (0.4504, 0.1210), (0.3625, 0.1790))
HEAD_RANGE = {"yaw": 12.0, "pitch": 8.0, "roll": 3.0}
USED_THIS_RELEASE = False


def _clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def head_angle(basin, calibration=None):
    """Basin triplet → head rotation in degrees. NOT rendered this release."""
    cal = calibration or BASIN_CALIBRATION
    if not basin or len(basin) < 3:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    out = {}
    for value, (mean, spread), axis in zip(basin, cal, ("yaw", "pitch", "roll")):
        t = _clamp((float(value) - mean) / (spread or 1.0), -1.0, 1.0)
        out[axis] = round(t * HEAD_RANGE[axis], 2)
    return out


# ── reading the state ───────────────────────────────────────────────────────

def _from_stage(stage):
    if not stage:
        return DEFAULT
    if stage == "shutdown":
        return "shutdown"
    if stage == "asleep":
        return "asleep"
    if stage == "locked":
        return "locked"
    if stage == "dreaming":
        return "dreaming"
    if stage == "thinking":
        return "thinking"
    if stage.startswith("worker:") or stage in ("loading field", "waking",
                                                "starting", "waiting for field"):
        return "away"
    return DEFAULT


def read(stage=None, info=None, was=None):
    """Pick the one clip for this moment.

    `was` is the expression currently playing, if the caller is tracking it —
    supply it and `play` comes back as the transition sequence.
    """
    info = info or {}

    # Reported, not rendered — see MOMENTS. A turn can be several of these at
    # once, and unlike a clip they do not have to compete for the one slot.
    moments = [m for m in MOMENTS if info.get(m)]

    name = _from_stage(stage)
    # A shutdown telemetry event carries the flag rather than a stage of its own.
    if info.get("shutdown"):
        name = "shutdown"

    out = {"name": name, "clip": name, **CLIPS[name]}
    # `also` kept as the key so nothing downstream has to change, but it now
    # carries MOMENT names ("challenging"), not clip names ("standing").
    out["also"] = moments
    out.update(transition(was, name))
    if info.get("will_research"):
        out["researching"] = info["will_research"]
    return out


def describe():
    """What exists, and what deliberately does not. For the frontend."""
    return {
        "expressions": sorted(CLIPS),
        "default": DEFAULT,
        # No longer a clip ranking — these are reported alongside the
        # clip, all at once, for the frame-colour layer to carry.
        "moments": list(MOMENTS),
        # ⚠️ NOT AN ASSET LIST ANY MORE. This was `clip_names()` — 7 loops and
        # 6 transitions, "13 clips to produce" — which stopped being true when
        # the app went procedural: the animation has a scene table, so there
        # is nothing to load per state and no missing-file case. `clip_names()`
        # is kept below and still tested, because the names are the STATES and
        # the frontend maps them; what is retired is calling them files.
        "states": sorted(CLIPS),
        "transitions_retired": "the animation moves between cues itself",
        "render_modes": {
            "head": "the animated head, as in conversation",
            "scene": "the avatar is replaced — full figure, underwater",
            "absent": "nothing rendered",
        },
        "live_field_channel": "frame colour only — see signal.py",
        "client_owned": [k for k, v in CLIPS.items() if v["owner"] == "client"],
        "not_this_release": {
            "basin_to_head_angle": "measured and kept, but nothing renders it; "
                                   "belongs to the animated-avatar version",
            "pre_rendered_clips": "retired  — the app animates "
                                  "procedurally from a scene table",
        },
        "rules": [
            "one clip at a time — never blend two",
            "the live field reaches the user through the frame colour ONLY",
            "`idle` is the only clip that moves on its own; stillness IS attention",
            "raise `attending` locally on first keystroke, not from /events",
            "asleep is a scene Mati returns from; SHUTDOWN is the one with no face",
            "head loops crossfade into each other — a clip is only needed "
            "when the SHOT changes, or when the moment is the point",
            "never label a clip in the UI — if it needs a legend it failed",
        ],
    }


if __name__ == "__main__":
    import json
    d = describe()
    print(json.dumps(d, indent=2))
    # ⚠️ IT USED TO PRINT "13 clips to produce". It does not any more: the app
    # animates from a scene table, so there is nothing to produce. What a
    # frontend needs from here is the STATE NAMES and which of them it must
    # map — the mapping itself lives in the app.
    names = clip_names()
    print(f"\n  {len(d['states'])} states; the frontend maps each to a scene\n")
    for n in d["states"]:
        c = CLIPS[n]
        print(f"    {n:<10} {c['render']:<7} {c['meaning']}")
    print(f"\n  (transition sequences are still computed by `read(was=...)` "
          f"and are\n   unused by the current app — {len(names['transitions'])} "
          f"of them)\n")
