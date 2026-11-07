"""
Field state as a colour signal — the pipe, not the language.

    from Mati import signal
    signal.encode(coherence=0.83)
    # {'encoder': 'steady', 'components': [...], 'period_ms': None, ...}

THE LONG IDEA is that Mati communicates in colour — eventually between agents,
first with the person in front of it. How is undecided: a **pulse** of colours
spelling out symbols over time, or a **range of frequencies** carrying words at
once. This module deliberately does not choose. It is the connection those
would be built on.

WHAT THAT COSTS, IN DESIGN TERMS. Both futures have to stay reachable, so the
signal shape carries what each would need even though today's encoder uses
neither:

    components   a LIST, always — one entry now. A spectral encoding puts
                 several present-at-once frequencies here.
    period_ms    None now, meaning static. A temporal encoding sets it, and
                 the components become a repeating sequence.

An encoder that emitted a bare colour would be simpler and would have to be
thrown away twice.

WHAT IT ENCODES TODAY. One thing: **how settled the field is.** Coherence,
which is already computed every turn and already published — so this costs
nothing on the backend. It is also the state the system prompt ALREADY asks
Mati to express in words ("let low coherence show as genuine uncertainty"), so
the frame and the sentence agree rather than saying two things.

⚠️ **THE RANGE IS NARROW AND THAT IS THE WHOLE PROBLEM.** Coherence sits around
0.77–0.94 in real use, not 0–1. Mapped naively the frame would be one colour
forever — a decoration that never moves. `CALIBRATION` holds the observed
range and the encoder normalises against it. If the field changes, recalibrate,
or the signal goes dead without anything failing.

⚠️ **COLOUR IS NEVER THE ONLY CHANNEL.** Roughly 8% of men cannot separate some
hues. Every component carries `level` alongside its hue, so a renderer can vary
thickness, opacity or a pulse in step. Baked into the data rather than left to
the frontend to remember.

⚠️ **IT MUST NEVER NEED EXPLAINING.** The UI spec says "no readouts". An ambient
hue is felt rather than read, which is why it does not violate that — but a
legend saying "amber means unsure" turns it straight back into a readout. If a
user has to be told what the colour means, this has failed and should be cut.
`meaning` is carried for developers and the record, and must not be rendered.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── palettes ────────────────────────────────────────────────────────────────
#
# DEFAULT IS ONE HUE. The brief was "one colour", and sweeping between two hues
# turned out to collide with the app: interpolating amber (38) to Maya blue
# (191) passes through GREEN at ~114, and the chat bubbles are blue and green.
# A mid-coherence frame would have read as a bubble colour.
#
# So `maya` holds the hue at Maya blue and moves saturation and lightness
# instead: settled is full Maya blue, unsettled is pale and washed out. That
# collides with nothing, keeps Mati its own colour at every moment, is more
# minimal, and is inherently colour-blind safe — a lightness axis needs no hue
# discrimination at all.
#
# `thermal` keeps the two-hue version available for when more bandwidth is
# wanted, and routes AWAY from green through violet rather than through it.
HUE_MAYA = 191

PALETTES = {
    "maya":    {"hue": lambda t: HUE_MAYA,
                "sat": lambda t: 0.12 + 0.63 * t,
                "light": lambda t: 0.72 - 0.22 * t},
    # 191 -> 255 -> 320: blue through violet to magenta. Avoids green entirely.
    "thermal": {"hue": lambda t: 320 - 129 * t,
                "sat": lambda t: 0.30 + 0.40 * t,
                "light": lambda t: 0.60},
}
DEFAULT_PALETTE = "maya"

# Observed coherence over real turns. NOT 0–1: mapped against the
# full range the frame would never move. Re-measure if the field is retrained.
CALIBRATION = (0.75, 0.95)

STEADY = "steady"


def _clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def normalise(coherence, calibration=None):
    """Coherence → 0..1 across the range it actually occupies.

    Returns 0 for the least settled end, 1 for the most. Outside the calibrated
    range it saturates rather than clipping to a surprise.
    """
    lo, hi = calibration or CALIBRATION
    if hi <= lo:
        return 0.5
    return _clamp((float(coherence) - lo) / (hi - lo))


def encode(coherence, encoder=STEADY, calibration=None,
           palette=DEFAULT_PALETTE):
    """Field state → a colour signal a renderer can draw.

    Today `steady` is the only encoder: one component, no period. The shape is
    what a pulse or a spectrum would also fill in, so those arrive as new
    encoders rather than as a new format.
    """
    if encoder != STEADY:
        raise ValueError(
            f"unknown encoder {encoder!r} — only {STEADY!r} exists. A pulse or "
            f"spectral encoding goes here, and the signal shape already has "
            f"room for it (components, period_ms).")

    if palette not in PALETTES:
        raise ValueError(f"unknown palette {palette!r} — have {sorted(PALETTES)}")
    pal = PALETTES[palette]
    settled = normalise(coherence, calibration)

    return {
        "encoder": STEADY,
        "palette": palette,
        # A list even with one entry: a spectral encoding puts several
        # simultaneous components here and nothing downstream has to change.
        "components": [{
            "hue": round(pal["hue"](settled), 1),
            "saturation": round(pal["sat"](settled), 3),
            "lightness": round(pal["light"](settled), 3),
            # The redundant, non-colour channel. A renderer varies thickness,
            # opacity or pulse with this so the signal survives colour-blindness.
            "level": round(settled, 3),
        }],
        # None = static. A temporal encoding sets this and `components` becomes
        # a sequence to walk.
        "period_ms": None,
        # For developers and the record. NEVER render this: the moment the
        # colour is labelled it stops being ambient and becomes a readout.
        "meaning": "settledness",
        "raw": round(float(coherence), 4),
    }


CARRIER = "carrier"


def carrier(symbols, rate=10.0, palette=DEFAULT_PALETTE):
    """The temporal encoder this module reserved room for. ⚠️ AGENT-FACING.

    ⚠️ THIS FILE PREDICTED THIS FUNCTION AND I NEARLY MISSED IT. The docstring
    above says the long idea is "a **pulse** of colours spelling out symbols
    over time", that `components` is a list so a sequence can live there, and
    that `period_ms` is None "now, meaning static. A temporal encoding sets
    it". `Mati/cuttlefish.py` was then built as a PARALLEL module that emitted
    hex strings and never touched any of it. This is the join: cuttlefish
    decides WHAT to say, `signal` remains the one pipe that says it.

    ⚠️ THE TWO ENCODERS CARRY DIFFERENT AUDIENCES ON ONE SURFACE, and they
    must never overlap in time. `steady` is for the person — ambient, slow,
    "how settled am I". `carrier` is for another machine — a handshake, a few
    seconds, then gone. Fax and voice shared a line by being in one mode at a
    time; see STUDY_cuttlefish.md §5.

    ⚠️ THE COLOURS COME FROM `cuttlefish.ALPHABET` AND ARE NOT NEGOTIABLE
    HERE. They were derived by maximising the worst-case separation across
    normal, deuteranopic, protanopic and tritanopic vision at once (49.6 ΔE).
    Re-deriving them from a palette would quietly throw that away.
    """
    from . import cuttlefish

    if palette not in PALETTES:
        raise ValueError(f"unknown palette {palette!r} — have {sorted(PALETTES)}")

    comps = []
    for s in symbols:
        h, sat, light = _hex_to_hsl(cuttlefish.ALPHABET[int(s) & 3])
        comps.append({
            "hue": round(h, 1),
            "saturation": round(sat, 3),
            "lightness": round(light, 3),
            # ⚠️ The non-colour channel still carries the symbol, so a
            # renderer that cannot show hue at all can still transmit.
            "level": round((int(s) & 3) / 3.0, 3),
            "symbol": int(s) & 3,
        })
    return {
        "encoder": CARRIER,
        "palette": palette,
        "components": comps,
        # No longer None: this is the sequence's step, and the whole point of
        # the field existing.
        "period_ms": round(1000.0 / float(rate), 1),
        "meaning": "handshake",
        "raw": None,
    }


def _hex_to_hsl(hexcol):
    """#RRGGBB → (hue 0-360, saturation 0-1, lightness 0-1)."""
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    mx, mn = max(r, g, b), min(r, g, b)
    light = (mx + mn) / 2.0
    if mx == mn:
        return 0.0, 0.0, light
    d = mx - mn
    sat = d / (2.0 - mx - mn) if light > 0.5 else d / (mx + mn)
    if mx == r:
        hue = ((g - b) / d) % 6
    elif mx == g:
        hue = (b - r) / d + 2
    else:
        hue = (r - g) / d + 4
    return hue * 60.0, sat, light


def describe():
    """What exists, and what deliberately does not. For the frontend."""
    return {
        # ⚠️ THIS LIST WENT STALE THE MOMENT `carrier` WAS ADDED, and it is
        # read by the frontend — a module describing itself wrongly is worse
        # than one that says nothing. `pulse` moved from `planned` to built
        # and is called `carrier`; `spectral` is still genuinely absent.
        "encoders": [STEADY, CARRIER],
        "planned": ["spectral (frequencies at once)"],
        "audiences": {
            STEADY: "the person — ambient, slow, 'how settled am I'",
            CARRIER: "another machine — a handshake, a few seconds, then gone",
        },
        # ⚠️ NEVER BOTH AT ONCE. Fax and voice shared a line by being in one
        # mode at a time; see STUDY_cuttlefish.md §5.
        "exclusive": True,
        "carrier_bits_per_second": 20,
        "calibration": CALIBRATION,
        "palettes": sorted(PALETTES),
        "default_palette": DEFAULT_PALETTE,
        "axis": {"hue": HUE_MAYA,
                 "why": "one hue, varying saturation and lightness — a hue "
                        "sweep would pass through green and collide with the "
                        "blue/green chat bubbles"},
        "rules": [
            "never label the colour in the UI — if it needs a legend it failed",
            "always render `level` as a non-colour channel too",
            "recalibrate if the field is retrained, or the signal goes dead",
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
    print()
    for pal in sorted(PALETTES):
        print(f"  palette {pal!r}")
        print(f"  {'coherence':>10}{'settled':>9}{'hue':>7}{'sat':>7}{'light':>7}")
        for c in (0.75, 0.80, 0.85, 0.90, 0.95):
            k = encode(c, palette=pal)["components"][0]
            print(f"  {c:>10.2f}{k['level']:>9.3f}{k['hue']:>7.1f}"
                  f"{k['saturation']:>7.3f}{k['lightness']:>7.3f}")
        print()
