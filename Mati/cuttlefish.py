"""
The cuttlefish channel — a handshake in colour, not a language.

`STUDY_cuttlefish.md` has the measurements. The short version, because it
decides everything in this file:

    4 colours, ~10 symbols/s, ~20 bits/s.

A 100-character sentence would take forty seconds, so ⚠️ THIS CHANNEL CANNOT
CARRY LANGUAGE AND MUST NEVER BE ASKED TO. It carries a HANDSHAKE: who I am,
what I speak, where to reach me — and then the real conversation moves
somewhere with room in it. That is the user's fax/modem analogy taken at its
word: the screech was never the conversation, it was two machines agreeing
how to have one.

⚠️ THE ALPHABET IS FOUR COLOURS AND THAT IS A CEILING, NOT A STARTING POINT.
Measured: with realistic auto white balance, 6 colours already errs at 11%.
Auto white balance is a control loop pulling every frame toward grey, which
means the camera ACTIVELY ERASES the signal — it is not additive noise, and
⚠️ differential encoding (the textbook answer for a channel with no absolute
reference) was TESTED AND MADE IT WORSE: 0.2% -> 0.4% at four colours, 34% ->
56% at sixteen. DPSK wins when the impairment is shared between neighbouring
symbols; this one reacts to each symbol's own content.

⚠️ THE COLOURS ARE COLOURBLIND-SAFE AND THAT WAS FREE AT FOUR. Chosen by
maximising the worst pairwise CIELAB separation across normal, deuteranopic,
protanopic and tritanopic vision at once: the worst case is 49.6 ΔE, where
2.3 is a just-noticeable difference and 10 reads as obviously different. At
sixteen colours there would have been no room for this.

⚠️ THE ALPHABET IS FOUR COLOURS BECAUSE THE MEDIUM IS A SCREEN. The user wants
the camera FLASH eventually, and that changes the arithmetic by ~100x —
rolling shutter makes every sensor ROW a time sample, so the same camera goes
from 30 samples/s to a measured 21,600. This codec is deliberately independent
of the medium: it maps bytes to symbols and back, and whether a symbol is
painted as a colour or blinked as a pulse is the transmitter's business.
Moving to the flash means a new ALPHABET (two symbols) and a new renderer, not
a new protocol. See STUDY_cuttlefish.md §4b.

⚠️ EVERY CAPACITY NUMBER IS SIMULATED. This module's loopback proves the
encoder and decoder agree through a real pixel round trip; it does NOT prove
a camera can read a screen. That needs hardware and is preregistered in the
study.
"""

from __future__ import annotations

# ── the alphabet ─────────────────────────────────────────────────────────────
# Derived, not chosen by eye. See the module docstring and
# `Mati/study_cuttlefish.py` for the search that produced them.
ALPHABET = ("#F90B45", "#3488E7", "#257014", "#F0C6CA")
BITS_PER_SYMBOL = 2

# ⚠️ THE PREAMBLE IS UGLY ON PURPOSE. A modem's screech is unpleasant and that
# is the feature — nobody mistakes it for speech. This must be impossible to
# mistake for the idle coherence colour, which is slow and calm. It alternates
# the two most separated symbols, then marks the end with a pair that cannot
# continue the alternation.
PREAMBLE = (0, 3, 0, 3, 0, 3, 1, 2)
VERSION = 1
MAX_PAYLOAD = 24                      # bytes; ~10s at 10 symbols/s

# The four modes a frame can be in. ⚠️ ONE AT A TIME — fax and voice shared a
# line by never mixing, and the frame's other job is showing coherence to a
# person. A pet that strobes a 64-bit identity is not a pet.
IDLE, CALLING, CARRIER, CLOSING = "idle", "calling", "carrier", "closing"


class Undecodable(ValueError):
    """The symbols did not form a frame. ⚠️ NOT the same as an empty payload."""


# ── bytes <-> symbols ────────────────────────────────────────────────────────

def to_symbols(data):
    """bytes -> list of 0..3, most significant pair first."""
    out = []
    for b in data:
        out += [(b >> 6) & 3, (b >> 4) & 3, (b >> 2) & 3, b & 3]
    return out


def from_symbols(syms):
    """list of 0..3 -> bytes. ⚠️ Refuses a partial byte rather than padding."""
    if len(syms) % 4:
        raise Undecodable(f"{len(syms)} symbols is not a whole number of bytes")
    out = bytearray()
    for i in range(0, len(syms), 4):
        a, b, c, d = syms[i:i + 4]
        out.append((a << 6) | (b << 4) | (c << 2) | d)
    return bytes(out)


def _crc8(data):
    """CRC-8/ATM. Small, and enough to catch what a 50-symbol frame can suffer."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


# ── frames ───────────────────────────────────────────────────────────────────

def encode(payload, version=VERSION):
    """Wrap a payload into a full frame of symbols.

    preamble · version · length · payload · crc8
    """
    if not isinstance(payload, (bytes, bytearray)):
        payload = str(payload).encode("utf-8")
    if len(payload) > MAX_PAYLOAD:
        # ⚠️ REFUSED, not truncated. A silently shortened identity is worse
        # than no identity — it would still pass its own checksum.
        raise ValueError(f"{len(payload)} bytes is past the {MAX_PAYLOAD}-byte "
                         f"limit; this channel is a handshake, not a message")
    body = bytes([version, len(payload)]) + bytes(payload)
    return list(PREAMBLE) + to_symbols(body + bytes([_crc8(body)]))


def decode(symbols):
    """Symbols -> payload bytes. Raises `Undecodable` rather than guessing.

    ⚠️ SYNCS ON THE PREAMBLE ANYWHERE IN THE STREAM. A receiver starts looking
    mid-transmission far more often than at the start, and the checksum is
    what settles whether a candidate sync was real.
    """
    syms = [int(s) for s in symbols]
    n = len(PREAMBLE)
    for start in range(0, max(1, len(syms) - n + 1)):
        if tuple(syms[start:start + n]) != PREAMBLE:
            continue
        rest = syms[start + n:]
        if len(rest) < 8:
            continue
        try:
            head = from_symbols(rest[:8])
        except Undecodable:
            continue
        version, length = head[0], head[1]
        need = (2 + length + 1) * 4
        if len(rest) < need:
            continue
        try:
            full = from_symbols(rest[:need])
        except Undecodable:
            continue
        body, crc = full[:-1], full[-1]
        if _crc8(body) != crc:
            # ⚠️ A bad checksum means this sync was a coincidence or the frame
            # was corrupted. Keep looking rather than returning something.
            continue
        return {"version": version, "payload": body[2:]}
    raise Undecodable("no complete frame in those symbols")


def handshake(node_id, capability=b""):
    """The frame Mati actually sends: who I am, and where to continue.

    ⚠️ NOT A MESSAGE. `node_id` is an identifier and `capability` is a pointer
    to a channel with room in it. If either grows into prose, the design has
    drifted and `encode` will refuse it.
    """
    if isinstance(node_id, str):
        node_id = node_id.encode("utf-8")
    if isinstance(capability, str):
        capability = capability.encode("utf-8")
    return encode(bytes(node_id[:8]) + bytes(capability[:14]))


def duration_seconds(symbols, rate=10.0):
    """How long a frame takes to send. Honest about the cost of any payload."""
    return len(symbols) / float(rate)


# ── rendering, so a frame is pixels ──────────────────────────────────────────

def _rgb(sym):
    h = ALPHABET[int(sym) & 3].lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def render(symbols, cell=24, gap=0):
    """Symbols -> an RGB pixel grid, one square per symbol.

    A strip rather than the real circular frame, because what the loopback
    has to prove is that the CODEC survives pixels — the shape it is painted
    in is the avatar's business.
    """
    w = len(symbols) * (cell + gap)
    rows = []
    for y in range(cell):
        row = []
        for s in symbols:
            row += [_rgb(s)] * cell
            row += [(0, 0, 0)] * gap
        rows.append(row)
    return rows, w, cell


def read(rows, n_symbols, cell=24, gap=0):
    """Pixels -> symbols, sampling the middle of each cell.

    ⚠️ SAMPLES THE CENTRE, NOT THE EDGE. Any real capture blurs across a
    boundary, and a decoder that trusted edge pixels would be reading two
    symbols at once.
    """
    import math
    mid = rows[len(rows) // 2]
    pal = [_rgb(i) for i in range(4)]
    out = []
    for i in range(n_symbols):
        x = i * (cell + gap) + cell // 2
        px = mid[min(x, len(mid) - 1)]
        best, bd = 0, 1e18
        for k, p in enumerate(pal):
            d = sum((px[c] - p[c]) ** 2 for c in range(3))
            if d < bd:
                bd, best = d, k
        out.append(best)
    return out


def to_png(rows, path):
    """Write the grid as a PNG, so a frame can actually be looked at.

    Stdlib only — `zlib` and `struct`. No dependency for a debugging aid.
    """
    import struct, zlib
    h, w = len(rows), len(rows[0])
    raw = b"".join(b"\x00" + bytes(v for px in r for v in px) for r in rows)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)
    return path


# ── the mode machine ─────────────────────────────────────────────────────────

class Frame:
    """What colour the avatar's frame should be right now.

    ⚠️ IDLE IS THE NORMAL STATE and it is the coherence colour — the pet. The
    carrier is a few seconds, rare, and returns to idle leaving no trace.

    ⚠️ OFF BY DEFAULT. Nothing in Mati interrupts unless turned on per kind,
    and a screen that starts flashing at another machine without being asked
    is the same category of act.
    """

    def __init__(self, enabled=False, rate=10.0):
        self.enabled = bool(enabled)
        self.rate = float(rate)
        self.mode = IDLE
        self._queue = []
        self._idle_colour = None

    def set_coherence(self, coherence):
        """Idle from the field, through `signal.py` — the one pipe.

        ⚠️ THE IDLE COLOUR IS NOT THIS MODULE'S TO INVENT. `signal.encode`
        already turns coherence into a colour, is already published in
        `/state`, and is already calibrated (0.75–0.95, because that is the
        range coherence actually occupies). Picking a colour here would give
        the frame two disagreeing sources for the same state.
        """
        from . import signal
        c = signal.encode(coherence)["components"][0]
        self._idle_colour = (f"hsl({c['hue']:.0f} {c['saturation'] * 100:.0f}% "
                             f"{c['lightness'] * 100:.0f}%)")
        return self._idle_colour

    def as_signal(self):
        """The carrier as a `signal.py` payload, or None when idle.

        ⚠️ This is how the frame reaches a renderer: one shape, two encoders.
        """
        if self.mode not in (CALLING, CARRIER) or not self._queue:
            return None
        from . import signal
        return signal.carrier(self._queue, rate=self.rate)

    def call(self, node_id, capability=b""):
        """Begin a handshake. Returns (symbols, seconds) or None when off."""
        if not self.enabled:
            return None
        self._queue = handshake(node_id, capability)
        self.mode = CALLING
        # ⚠️ A COPY. Returning `self._queue` handed the caller the live list
        # that `step()` then drains underneath them — by the time they looked
        # at "the frame they sent", it was empty. Found by the loopback test
        # comparing what went out against what was returned.
        return list(self._queue), duration_seconds(self._queue, self.rate)

    def step(self):
        """The next symbol, or None when there is nothing to send."""
        if not self._queue:
            if self.mode in (CALLING, CARRIER):
                self.mode = CLOSING
            elif self.mode == CLOSING:
                self.mode = IDLE
            return None
        if self.mode == CALLING:
            self.mode = CARRIER
        return self._queue.pop(0)

    def colour(self):
        """The CSS colour for right now."""
        if self.mode in (CALLING, CARRIER) and self._queue:
            return ALPHABET[self._queue[0] & 3]
        return self._idle_colour


def status():
    return {
        "alphabet": list(ALPHABET),
        "bits_per_symbol": BITS_PER_SYMBOL,
        "symbols_per_second": 10,
        "bits_per_second": 20,
        "max_payload_bytes": MAX_PAYLOAD,
        "note": "a handshake, not a message; ~20 bits/s is slower than a "
                "1977 acoustic coupler",
        "verified": "codec through a pixel round trip; NOT through a camera",
    }
