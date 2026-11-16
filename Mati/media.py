"""
One door for anything that plays. ⚠️ "play" DOES NOT MEAN MUSIC.

"play" should wake the media module, not assume music. A front door that
assumes music sends everything else to a library search for a song by that
name:

    "play a podcast about history"  -> a track called "a podcast about history"
    "play radio from Mexico"        -> a track called "radio from Mexico"
    "I want to listen to something" -> not recognised at all

and leaves the other backends with no recogniser, so nothing a person SAYS can
reach them.

WHAT THIS DOES. It classifies first and routes second:

    radio    an internet stream   -> radio.py   (in-process, no app)
    track    music               -> radio.py   (a STATION; see below)
    podcast  a show              -> podcast.py
    film     moving pictures     -> archive_org.py (plays in the app)
    preset   a button, 1–10      -> presets.py

⚠️ NO APPLE MUSIC. Music works like a car radio with station presets, not a
streaming app with a library. So `track` is still a RECOGNISER — "play some
jazz music" has to keep working — but it routes to a station. Playlists,
ratings and reviews are recognised and declined rather than silently dropped,
because a request deserves a sentence saying it is not something Mati does.

⚠️ ONE LIBRARY READ EXISTS AND IS NOT REACHABLE FROM HERE:
`dj.import_taste()`, run once at setup, genres only.

⚠️ AND STOP IS IMMEDIATE. Nothing of Mati's lives in another application, so
stopping waits on nothing: stop is the one command where latency is the whole
problem.

⚠️ AN UNCLEAR REQUEST IS ASKED ABOUT, NOT GUESSED. "play something" has no
kind in it, and picking one means starting audio nobody asked for — the first
irreversible thing Mati does and the one the person hears immediately. So the
answer is a question. A false negative costs one sentence; a false positive
costs a room.

⚠️ NOTHING HERE PLAYS ANYTHING ITSELF. Each backend keeps its own rules —
radio opens no application, film hands a URL to the page, and `dj` records any
launch so it can be closed again. This module only decides who is asked.
"""

from __future__ import annotations

import re

from . import presets

# ── the verb ─────────────────────────────────────────────────────────────────
# ⚠️ A VERB IS REQUIRED. "some jazz would be nice" is not an instruction, and
# treating it as one is how audio starts unbidden.
# ⚠️ "turn on" IS A PLAY VERB — "turn on the radio". `_STOP` is tested first
# and owns "turn (it) off", so the two cannot collide.
# ⚠️ "show me" AND "find me" ARE PLAY VERBS. "show me a video about physics"
# matching no verb would send the whole thing to the model, which thinks for a
# minute and says nothing useful — indistinguishable, from outside, from a
# hang.
# Verbs that START something. ⚠️ Kept separate from `_PLAY` because "show me"
# is a request to SEE a list as much as to begin playing one.
_STARTS = re.compile(
    r"\b(play|put on|turn on|listen to|watch|stream|start)\b", re.I)

_PLAY = re.compile(
    r"\b(play|put on|turn on|listen to|watch|stream|start"
    r"|show me|find me|pull up|bring up)\b", re.I)

# ⚠️ THE ONE-TIME LIBRARY READ, and it must be ASKED FOR. It is the only thing
# here that touches Music.app, and it reads genres so a cold radio has
# somewhere to start.
_SETUP_MUSIC = re.compile(
    r"\b(?:set ?up|setup|import|read|scan)\b[^.?!]{0,24}?"
    r"\b(?:music|library|itunes|genres?|taste)\b"
    r"|\b(?:music|library)\b[^.?!]{0,12}?\bset ?up\b", re.I)
_STOP = re.compile(
    r"\b(stop|pause|silence|turn (it )?off)\b"
    # ⚠️ "enough" AND "quiet" ONLY AS A COMMAND, NOT AS ORDINARY WORDS. A
    # budget paragraph saying "I would like to spend ENOUGH to buy a coffee"
    # must not be answered with "Stopped Bristol Sound Webradio".
    #
    # In the media sense these words END an utterance: "enough", "that's
    # enough", "enough already", "quiet please". As quantifier and adjective
    # they sit mid-sentence — "enough to buy", "enough for two", "a quiet
    # neighbourhood" — so they are accepted only at the end of the message or
    # before punctuation.
    r"|\b(enough|quiet)\b\s*(?:already|please|now)?\s*[.!]*\s*$",
    re.I)

# How long a message may be and still be a bare CONTROL command.
#
# ⚠️ CONTROL VERBS CARRY NO OBJECT, WHICH IS WHY THEY NEED THIS AND `play`
# DOES NOT. "stop", "louder", "what's playing" cannot be qualified, so they
# are terse by nature; a long message containing one is a message that happens
# to use the word. A play request NAMES something, so it is legitimately
# longer ("play something by Alicia Keys from her early albums").
#
# ⚠️ THE SAME GUARD AS `report.is_only_request` and
# `continuity.is_only_question`: a keyword match inside prose that was about
# something else is not a command.
_CONTROL_MAX_WORDS = 7
_LOUDER = re.compile(r"\b(louder|turn (it )?up|volume up)\b", re.I)
_SOFTER = re.compile(r"\b(quieter|softer|turn (it )?down|volume down)\b", re.I)
_WHAT = re.compile(r"\b(what(?:'s| is) (?:this|that|playing|on)|"
                   r"what am i listening to|now playing)\b", re.I)

# ⚠️ PLAYLISTS AND REVIEWS ARE READS, NOT PLAY REQUESTS, so they get their own
# verbs rather than being squeezed into `play`. "what do I think of X" must
# never start audio.
_PLAYLISTS = re.compile(r"\b(my playlists?|playlists?|what playlists)\b", re.I)
_REVIEWS = re.compile(
    r"\b(what (?:do|did) i (?:think|rate)|my ratings?|my reviews?|"
    r"how (?:did|do) i rate|rated|what i like|"
    r"my opinion (?:of|on)|opinion (?:of|on))\b", re.I)
_RATE = re.compile(
    r"\b(rate|give)\b.{0,40}?\b(\d(?:\.\d)?)\s*stars?\b", re.I)

# ── the kinds, each with the words that actually name it ─────────────────────
# ⚠️ ORDER MATTERS AND IS DELIBERATE. `film` is checked before `track` because
# "play the movie theme" is about a film only if nothing more specific is
# present, and `podcast` before `radio` because "radio documentary" is a show.
KINDS = (
    # ⚠️ "video" WAS MISSING, AND IT IS THE COMMONEST WORD OF THE SET. "play
    # nasa videos" matched NO kind, so a named thing with no kind went to the
    # radio and searched for a STATION called "nasa videos".
    # ⚠️ THE SHELF NAMES ARE FILM WORDS TOO, and leaving them out meant "play
    # some shorts" and "play some cartoons" never reached the film path at
    # all — they matched no kind, so a named thing with no kind went to the
    # RADIO and searched for a station called "cartoons".
    #
    # ⚠️ TESTING `_find_film` DIRECTLY MAKES IT LOOK RIGHT. Calling the
    # function bypasses `hears`, which is where the routing lives — the only
    # reason this surfaced is that the live check went through the server.
    ("film", re.compile(
        r"\b(movie|movies|film|films|video|videos|footage|clip|clips|"
        r"documentar(?:y|ies)|pel[ií]cula|pel[ií]culas|v[ií]deos?|cine|"
        r"cartoon|cartoons|shorts|newsreels?|serials?|western|westerns|"
        r"noir|silent)\b",
        re.I)),
    ("podcast", re.compile(
        r"\b(podcast|podcasts|episode|episodes|audiobook|"
        r"the daily|series)\b", re.I)),
    ("radio", re.compile(
        r"\b(radio|station|stations|stream|live|"
        r"emisora|radioemisora)\b", re.I)),
    ("track", re.compile(
        r"\b(song|songs|track|tracks|album|albums|music|musica|"
        r"m[uú]sica|artist|band)\b", re.I)),
)

# Genre words strongly imply a stream rather than a library search, because the
# library is one person's collection and a genre is a whole station.
_GENRE = re.compile(
    r"\b(jazz|classical|ambient|lo-?fi|rock|metal|pop|blues|reggae|"
    r"electronic|techno|house|salsa|cumbia|ranchera|mariachi|folk|"
    # ⚠️ `hip[- ]?hop`, not `hip-?hop`. People type it with a SPACE, and the
    # hyphen-only form silently dropped the commonest spelling of one of the
    # commonest genres — it still played, by falling through to `what`, so
    # nothing looked broken.
    r"country|hip[- ]?hop|rap|soul|funk|punk|opera|baroque|bossa|"
    r"news|talk|sports)\b", re.I)

_FILLER = re.compile(
    r"\b(some|a|an|the|me|my|us|for|please|something|anything|"
    r"now|then|up|on|of|by|from|in|about|i|want|would|like|to|"
    r"can|you|could|us)\b", re.I)


def _best_match(wanted, shows, key="title"):
    """The first result that actually resembles the request, or None.

    ⚠️ ONLY FOR NAME SEARCHES. `radio.near` and `radio.by_genre` match on
    TAGS, where the title is not what was searched and need not resemble
    anything — "Blue Note Radio" is a correct answer for "jazz". Applying
    this there rejects the right station. Resemblance is only meaningful
    against the thing that was actually searched.

    ⚠️ RESEMBLANCE IS SHARED CONTENT WORDS, deliberately crude. A title is
    short and a person types roughly what they remember, so one shared
    content word is a real signal and zero shared words is a real miss. The
    filler list is reused so "of" and "the" cannot carry a match on their own.
    """
    import re as _re
    want = {w.lower() for w in _re.findall(r"[\w']+", wanted or "")
            if not _FILLER.fullmatch(w) and len(w) > 2}
    if not want:
        return shows[0] if shows else None
    for sh in shows:
        have = {w.lower() for w in _re.findall(r"[\w']+", sh.get(key) or "")}
        if want & have:
            return sh
    return None

def _trim_filler(text):
    """Strip filler from the EDGES of a request, never from inside it.

    ⚠️ `_FILLER` OVER THE WHOLE STRING WOULD EAT TITLES. It removes `of`,
    `the`, `a`, `in`, `about`, `from`, `by` and `to` — every one of which is a
    legitimate word INSIDE a name:

        "play the Theories of Everything podcast" -> "Theories Everything"
        "play Lord of the Rings"                  -> "Lord Rings"

    and a search for "Theories Everything" finds a different show, which Mati
    would then report on — a wrong answer that looks like a confident one,
    from one missing preposition.

    ⚠️ The stripping is still RIGHT at the edges: "play some jazz" must
    become "jazz", and "I would like to listen to a podcast about physics"
    must become "physics". So this walks in from both ends and stops at the
    first word that is not filler, leaving whatever is between them alone.
    """
    words = " ".join((text or "").split()).split(" ")
    is_filler = lambda w: bool(w) and _FILLER.fullmatch(w) is not None
    while words and is_filler(words[0]):
        words.pop(0)
    while words and is_filler(words[-1]):
        words.pop()
    return " ".join(words)

# ── Presets: ten buttons ─────────────────────────────────────────────────────
# ⚠️ ANCHORED AND WHOLE. "play 3" is a button; "play 3 songs" is not, and
# "play 1990s rock" is not. Requiring the number to be the end of the request
# is what keeps a quantity or a year from being read as a slot.
_PLAY_PRESET = re.compile(
    r"\b(?:play|put on|tune(?: in)? to|go to|switch to)\s+"
    r"(?:preset|button|station|number|#)?\s*"
    r"(?P<n>10|[1-9])\s*$", re.I)

# ⚠️ THE NUMBER IS OPTIONAL. What a person actually says — "add this to my
# presets", "save it as a preset", "make this a preset" — has no slot in it,
# and requiring one would send all of them to `_LIST_PRESETS`: Mati would read
# the list back ending "8 buttons still free", and that reads as confirmation
# while nothing was written.
#
# ⚠️ A LIST THAT LOOKS LIKE A CONFIRMATION IS WORSE THAN AN ERROR.
#
# Without a number the next free button is used — `_save_preset` resolves it,
# and says which one it took.
_SAVE_PRESET = re.compile(
    # "make"/"turn" are safe HERE only because this branch still requires
    # the word preset/button/slot to follow. They are deliberately absent
    # from the bare "save this" branch below, where they would catch
    # "make this quieter" and "turn this off".
    r"\b(?:save|store|keep|pin|add|make|turn)\b"
    r"(?:(?![.?!]).){0,30}?"
    r"\b(?:preset|presets|button|slot)\b"
    r"\s*(?:#\s*)?(?P<n>10|[1-9])?\b"
    # "save this station" / "keep this one" — no the word preset at all, but
    # unambiguous while something is playing.
    r"|\b(?:save|keep|pin)\b\s+(?:this|that|it)\b"
    r"(?:\s+(?:station|one|stream))?\s*$",
    re.I)

# ⚠️ NOT WHEN A NUMBER FOLLOWS. Without the lookahead this matched the bare
# word "preset" in "play preset 3" and answered with the list.
_LIST_PRESETS = re.compile(
    r"\b(?:what(?:'s| is| are)?\s+)?(?:my\s+)?presets?\b(?!\s*#?\s*\d)"
    r"|\bwhat(?:'s| is)? on (?:my )?(?:presets|buttons)\b", re.I)

# ── Moving through what a search found ───────────────────────────────────────
# ⚠️ Some people say "station" instead of "preset". `_PLAY_PRESET` already
# accepts "station 3", and this is the other half: a person who does not want
# a NUMBER, they want the NEXT one.
_NEXT = re.compile(
    r"\b(?:next|another|different|other|one more|skip|try another)\b"
    r"[^.?!]{0,20}?\b(?:station|one|stream)\b"
    r"|\b(?:next|skip)\s*$"
    r"|\bsomething else\b", re.I)

# ── Referring back to what was offered ───────────────────────────────────────
# ⚠️ A HELD LIST MUST BE WALKABLE IN EVERY DIRECTION, not only forward:
#   "Let me hear the first station again"
#   "I mean the first of the hip hop stations that you presented"
#   "That's not hip hop"
# Otherwise Mati replies conversationally while the radio ignores the request.
_BACK = re.compile(
    r"\b(?:go back|previous|last one|the one before|back to the (?:last|previous))\b"
    r"|\bprevious station\b", re.I)

# ⚠️ ORDINALS ONLY — A NUMBER MEANS A PRESET BUTTON. "play station 3" is button
# 3. A bare "station 3" meaning "the third you offered" would make the same
# words mean two things depending on whether a verb was present.
#
# ⚠️ `stations?`, NOT `station`. "the first of the hip hop STATIONS that you
# presented" would fail, because `\bstation\b` will not match inside
# "stations".
_NTH = re.compile(
    r"\b(?:the\s+)?(?P<ord>first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)"
    r"\b[^.?!]{0,40}?\b(?:stations?|ones?)\b", re.I)

_ORDS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
         "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}

# ⚠️ WHAT MAKES AN UTTERANCE ABOUT MEMORY RATHER THAN MUSIC. Used only to
# STOP a media match, never to make one — see the guard in `hears`.
# ⚠️ "talk" IS DELIBERATELY ABSENT: there are talk stations, and "next talk
# station" must keep working.
_ABOUT_RECALL = re.compile(
    r"\b(?:conversations?|remember|recall|memor(?:y|ies)|"
    r"we (?:discussed|talked|said)|you (?:said|told)|told me|"
    r"sessions?|earlier today|yesterday we)\b", re.I)

# A radio noun present means the media reading stands.
_RADIO_NOUN = re.compile(
    r"\b(?:stations?|streams?|presets?|radio|song|songs|track|tracks|"
    r"album|albums|artist|artists|playlist|playlists|podcast|podcasts|"
    r"episode|episodes|film|films|movie|movies|music)\b", re.I)

_LIST_FOUND = re.compile(
    r"\b(?:what|which)\s+(?:stations?|ones?)\b"
    r"|\blist (?:the )?stations?\b"
    r"|\bwhat did you (?:find|present|offer)\b", re.I)

# ⚠️ "that station from yesterday". `dj.what_played` resolves it against the
# RECORD.
#
# ⚠️ EVERY PHRASE HERE MUST BE ONE `what_played` CAN ACT ON, and every phrase
# it can act on must be here — otherwise one half can answer and the other
# never asks. The tests check the two agree.
_WHENS = (r"yesterday|last night|last evening|"
          r"this morning|this afternoon|this evening|tonight|today|earlier|"
          r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
          r"[a-z]+\s+night")

_WHEN_PLAYED = re.compile(
    r"\b(?:that|the)\s+(?:station|one|stream)\b[^.?!]{0,16}?"
    r"\b(?:from|on)\s+(?P<when>" + _WHENS + r")\b"
    # "what was I listening to this morning" names no station and still asks
    # the same question.
    r"|\bwhat\s+(?:was|were)\s+(?:i|we)\s+listening\s+to\b[^.?!]{0,12}?"
    r"\b(?P<when2>" + _WHENS + r")\b", re.I)

# Everything the person said AFTER the verb — that is where the subject is.
# ⚠️ MUST LIST EVERY VERB `_PLAY` LISTS. Adding "turn on" to `_PLAY` and not
# here left "turn on the radio" with the subject "turn" — the slice never
# happened, so the verb itself became the thing being searched for.
_AFTER_VERB = re.compile(
    r"\b(?:play|put on|turn on|listen to|watch|stream|start"
    r"|show me|find me|pull up|bring up)\b\s*(.*)$", re.I)


# A sentence that ASKS about something rather than instructing.
_ASKING = re.compile(r"^\s*(?:what|which|who|whom|whose|why|how|when)\b", re.I)

# ⚠️ BUT A QUESTION CAN STILL BE A REQUEST. "Can you play that station?" and
# "What should I play?" are instructions wearing a question mark, and the
# earlier `_DOING` work already settled that for small talk. These forms are
# exempt from the guard.
_REQUEST_FORM = re.compile(
    r"\b(?:can|could|would|will|should|shall)\s+(?:you|i|we)\b"
    r"|\bplease\s+(?:play|put on|turn on|start)\b"
    r"|\b(?:let'?s|lets)\s+(?:play|listen|watch)\b", re.I)

# "play" as part of another verb phrase, never an instruction.
_PLAY_AS_NOUN = re.compile(
    r"\b(?:did|does|do|had|has|have|was|were|is|are)\s+"
    r"(?:\w+\s+){0,4}?(?:play|played|playing)\b"
    r"|\b(?:like|love|want|wanted|used|going|plan|planned|try|tried|learn|"
    r"learned|able|hate|refuse)\s+to\s+play\b"
    r"|\brole\b[^.?!]{0,30}\bplay\b"
    r"|\bpart\b[^.?!]{0,30}\bplay\b", re.I)


# Verbs that mean "play" in any sentence, and verbs that only mean it when a
# media word follows. See the "START" note in `hears`. ⚠️ Together these must
# cover every verb `_PLAY` lists.
_STRONG_PLAY = re.compile(
    r"\b(play|put on|turn on|listen to|watch|stream)\b", re.I)
_WEAK_PLAY = re.compile(
    r"\b(start|show me|find me|pull up|bring up)\b", re.I)
_WEAK_WINDOW = 4          # words after the verb that may name the media


def _weak_verb_names_media(t):
    """Does a weak play verb have a media word right after it?"""
    for m in _WEAK_PLAY.finditer(t):
        window = " ".join(t[m.end():].split()[:_WEAK_WINDOW])
        if (_GENRE.search(window) or _RADIO_NOUN.search(window)
                or any(p.search(window) for _, p in KINDS)):
            return True
    return False


def _asking_about(t):
    """True when this is a question ABOUT something, not an instruction.

    ⚠️ THE EXEMPTION IS NOT OPTIONAL. Without `_REQUEST_FORM`, "Can you play
    that station?" would be refused as a question — which is the opposite
    error and just as visible.
    """
    if _REQUEST_FORM.search(t):
        return False
    return bool(_ASKING.match(t) or _PLAY_AS_NOUN.search(t))


def _subject(text):
    """What was named. ⚠️ Kind words are only removed if something remains.

    Stripped unconditionally, "listen to the daily" would know it was a
    PODCAST and lose WHICH podcast — the words that named it are the same words
    that classified it — and "I want to listen to something" would come through
    as the subject "I want to".
    """
    m = _AFTER_VERB.search(text or "")
    t = m.group(1) if m else (text or "")
    t = re.sub(r"[^\w\s'&-]", " ", t)

    # ⚠️ "<KIND> ABOUT <X>" MEANS X IS THE SUBJECT. Checked before any kind
    # word is removed, because both halves are often kind words themselves:
    # "a podcast about current movies" lost `movies` to the strip below and
    # answered with "current". The topic of a show is not a competing kind,
    # it is the thing being asked for.
    about = re.search(r"\babout\s+(.+)$", t, re.I)
    if about:
        topic = _trim_filler(about.group(1))
        if topic:
            return topic

    # Try removing the kind words; keep that only if a subject survives.
    stripped = t
    for _, pat in KINDS:
        stripped = pat.sub(" ", stripped)
    without_filler = _trim_filler(stripped)
    if without_filler:
        return without_filler
    # Nothing left once the kind was removed — so the kind WAS the subject,
    # as in "listen to the daily".
    bare = _trim_filler(t)
    # ⚠️ UNLESS IT IS A GENERIC KIND WORD. "put on some music" leaves "music",
    # and searching a library for a song called "music" finds nothing while
    # looking like it tried. A generic word is NO subject.
    if bare.lower() in ("music", "song", "songs", "track", "tracks", "album",
                        "movie", "movies", "film", "films", "podcast",
                        "podcasts", "radio", "station", "episode", "series",
                        "documentary", "musica", "música",
                        # ⚠️ Added with the `film` kind words. "play some
                        # videos" left the subject "videos", and searching a
                        # film shelf for the word "videos" is the same
                        # mistake `_GENRE` words were already protected from.
                        "video", "videos", "footage", "clip", "clips",
                        "vídeo", "vídeos"):
        return ""
    return bare


# ⚠️ `_subject` IS FOR PLAY REQUESTS ONLY. It slices after a play verb, so on
# "what do I think of X" it returned "what do think X" — the question words
# survived and the title was buried in them. Reviews and ratings name a thing
# in a different sentence shape, so they get their own extractor.
_OF = re.compile(
    r"\b(?:think (?:of|about)|rate|rated|give|opinion (?:of|on)|"
    r"feel about)\b\s*(.+)$", re.I)
_TRAILING_STARS = re.compile(r"\s*\b\d(?:\.\d)?\s*stars?\b.*$", re.I)


def _named_thing(text):
    """The thing a review or rating is ABOUT. "" when nothing was named."""
    m = _OF.search(text or "")
    s = m.group(1) if m else ""
    s = _TRAILING_STARS.sub("", s)                 # "… 4 stars" is the rating
    s = re.sub(r"^\s*(?:the|my|a|an)\s+", "", s, flags=re.I)
    s = re.sub(r"[?!.,;:]+\s*$", "", s)
    return " ".join(s.split())


def hears(text):
    """What was asked of the media, or None. ⚠️ Never a guess about the KIND.

    Returns a dict with `verb`, and for a play request either a definite
    `kind` or `kind=None`, which means ASK.
    """
    t = " ".join((text or "").split())
    if not t:
        return None

    # ⚠️ A BARE CONTROL COMMAND IS SHORT. See `_CONTROL_MAX_WORDS`: these
    # verbs take no object, so a long message containing one is prose that
    # happens to use the word, not an instruction. Checked once here rather
    # than at each branch so a later control verb cannot forget it.
    terse = len(t.split()) <= _CONTROL_MAX_WORDS

    if terse and _WHAT.search(t):
        return {"verb": "what"}
    # ⚠️ Before the play check, so "what do I think of X" is never a request
    # to play X.
    m = _RATE.search(t)
    if m:
        return {"verb": "rate", "what": _named_thing(t),
                "stars": float(m.group(2))}
    if _REVIEWS.search(t):
        # ⚠️ "" means "all of them", a name means "just this one".
        return {"verb": "reviews", "what": _named_thing(t)}
    # ⚠️ `_STARTS`, NOT `_PLAY`. This guard exists so "play my playlists" is a
    # play rather than a read — but "show me" is a play verb, and "show me my
    # playlists" must not be caught by it. "show me" DISPLAYS; it does not
    # start audio, and the distinction is the whole point of the guard.
    if _PLAYLISTS.search(t) and not _STARTS.search(t):
        return {"verb": "playlists"}
    # ⚠️ Stop is checked before play so "stop playing" is a stop.
    if terse and _STOP.search(t) and not re.search(r"\bplay\b(?!ing)", t, re.I):
        return {"verb": "stop"}
    if terse and _LOUDER.search(t):
        return {"verb": "louder"}
    if terse and _SOFTER.search(t):
        return {"verb": "softer"}
    # ⚠️ ALL THREE PRESET FORMS ARE CHECKED BEFORE THE `_PLAY` GATE, and the
    # ORDER AMONG THEM MATTERS. Two bugs came from getting this wrong:
    # "play preset 3" was swallowed by the LIST pattern (which matches the bare
    # word "preset"), and "tune to 7" returned None because `tune` is not in
    # `_PLAY` and the gate ran first. Save, then tune, then list.
    if _SETUP_MUSIC.search(t):
        return {"verb": "setup_music"}

    m = _SAVE_PRESET.search(t)
    if m:
        n = m.groupdict().get("n")
        return {"verb": "save_preset", "slot": int(n) if n else None}

    m = _PLAY_PRESET.search(t)
    if m:
        n = int(m.group("n"))
        if 1 <= n <= presets.SLOTS:
            return {"verb": "play", "slot": n, "kind": "radio", "what": None,
                    "genre": None}

    # ⚠️ BEFORE `_LIST_PRESETS` and before the play gate, for the same reason
    # the preset forms are: "next station" has no play verb in it.
    m = _WHEN_PLAYED.search(t)
    if m:
        when = m.group("when") or m.group("when2")
        return {"verb": "played_when", "when": when.lower()}
    # ⚠️⚠️ A MEMORY QUESTION IS NOT A RADIO COMMAND.
    #
    #   "What do you recall from our previous conversations?"
    #       must not be {verb: back} — "I do not have a list of stations to
    #       move through"
    #   "Can you remember things between sessions, or do you start over?"
    #       must not be {verb: play, what: "over..."} — a station starting
    #
    # `_BACK` matches a BARE "previous", and `previous` is a far commoner word
    # about conversations than about stations. A false negative costs one
    # sentence, a false positive costs a room — and `_asking_about` sits BELOW
    # this branch, so `_BACK`/`_NEXT`/`_NTH` would never reach it.
    #
    # ⚠️ THE GUARD CANNOT JUST BE "DOES IT MENTION A CONVERSATION". "next talk
    # station" and "play another station and remember it" are real radio
    # commands that mention one. So it requires a radio NOUN to be absent too:
    # with a station in the sentence the command stands, without one a memory
    # word wins.
    if _ABOUT_RECALL.search(t) and not _RADIO_NOUN.search(t):
        return None

    # ⚠️ BEFORE `_NEXT`, because "the one before" contains "one" and
    # "previous station" would otherwise be read as a request for another.
    if _BACK.search(t):
        return {"verb": "back"}
    m = _NTH.search(t)
    if m:
        n = _ORDS.get((m.group("ord") or "").lower())
        if n:
            return {"verb": "nth", "n": int(n)}
    if _LIST_FOUND.search(t):
        return {"verb": "listing"}
    if _NEXT.search(t):
        return {"verb": "next"}

    if _LIST_PRESETS.search(t):
        return {"verb": "presets"}

    # ⚠️ "play" IS ALSO AN ORDINARY ENGLISH VERB, and matching it ANYWHERE turns
    # questions into commands:
    #
    #   "What role did the calmecac play in Aztec education?"
    #       -> a PLAY command for "Aztec education"
    #   "What part did Chaplin play in the film?"
    #       -> kind=film — a history question would start a VIDEO
    #
    # A false negative costs one sentence, a false positive costs a room. Same
    # shape as `_DOING.search` answering "What have you read recently?" as an
    # instruction.
    if _asking_about(t):
        return None

    if not _PLAY.search(t):
        return None

    # ⚠️⚠️ "START" IS NOT A PLAY VERB ON ITS OWN. Someone working on a story
    # says
    #
    #   "I think I want to start the story with the Akkadian version of …"
    #   "Actually I want to start with Genesis 5."
    #
    # and each must not come back as "I do not know “…” as a station" in no
    # time at all, by code. `start`, `show me`, `find me`, `pull up` and `bring
    # up` are ordinary English; `play`, `listen to` and `watch` are not. So a
    # WEAK verb counts only when the thing right after it is a media word —
    # "start the radio", "start some jazz", "find me a podcast". A false
    # negative costs one sentence, a false positive costs the conversation.
    if not _STRONG_PLAY.search(t) and not _weak_verb_names_media(t):
        return None

    # ⚠️ EARLIEST MATCH WINS, NOT TABLE ORDER. The first kind in the TABLE
    # would be `film`, so "play a podcast about current movies" would classify
    # as a FILM and search for a movie called "current" — `movies` appears in
    # the table before `podcast` does, never mind that it comes later in the
    # sentence.
    #
    # ⚠️ THE HEAD NOUN COMES FIRST IN ENGLISH for this construction: "a
    # podcast about films" is a podcast, "a film about podcasts" is a film,
    # and in both the kind is the word nearer the start. Position is the
    # signal; table order is an accident of how the list was written.
    _hits = [(m.start(), name) for name, pat in KINDS
             for m in [pat.search(t)] if m]
    kind = min(_hits)[1] if _hits else None
    subject = _subject(t)

    # "watch" can only mean pictures; "listen to" can only mean sound.
    if kind is None and re.search(r"\bwatch\b", t, re.I):
        kind = "film"
    # A bare genre with no other signal is a station, not a library search:
    # "play some jazz" is a request for jazz, not for a song called "jazz".
    if kind is None and _GENRE.search(t):
        kind = "radio"
    # A named thing with no kind is most likely in the library — but only if
    # something was actually named.
    if kind == "track" and not subject:
        kind = "radio"

    # ⚠️ WHICH WORD MADE IT A FILM, because "play a movie" and "play some
    # videos" are the same kind and want different things — a studio picture
    # against a looping reel of shorts. Without this the two are
    # indistinguishable by the time `_find_film` sees them.
    # ⚠️ "documentary" IS NOT A FEATURE WORD, though it is a film word. A bare
    # "watch a documentary" answered with a studio comedy — right kind, wrong
    # thing entirely. It gets the archival shelf instead; see `_find_film`.
    feature = bool(re.search(r"\b(movie|movies|film|films|"
                             r"pel[ií]cula|pel[ií]culas|cine)\b", t, re.I))
    documentary = bool(re.search(r"\bdocumentar(?:y|ies)\b", t, re.I))

    return {"verb": "play", "kind": kind, "what": subject,
            "feature": feature if kind == "film" else None,
            "documentary": documentary if kind == "film" else None,
            "genre": (_GENRE.search(t).group(1).lower()
                      if _GENRE.search(t) else None)}


# ── routing ──────────────────────────────────────────────────────────────────

def _play_preset(slot, db_path=None):
    """Tune to a button. ⚠️ Plays the PINNED URL, never a fresh search."""
    from . import radio
    p = presets.recall(slot, db_path=db_path)
    if p is None:
        free = presets.free_slot(db_path=db_path)
        return (f"Preset {slot} is empty. Play a station and say “save as "
                f"preset {slot}”."
                + (f" {free} is free too." if free and free != int(slot) else ""))
    ok, msg = radio.play({"name": p["name"], "url": p["url"],
                          "bitrate": p["bitrate"], "country": p["country"]})
    if not ok:
        # ⚠️ NAMES THE BUTTON. A pinned station that has gone off the air is a
        # different problem from a bad search, and the person can re-save it.
        return f"Preset {slot} is {p['name']}, but it would not play — {msg}"
    # ⚠️ THE URL GOES IN TOO. Without `detail` the history records that a
    # station played and not WHICH stream, so "play that station from earlier"
    # could name it and not replay it.
    _remember(kind="preset", name=p["name"], detail=p["url"], db_path=db_path)
    return f"Preset {slot}: {msg}"


def _save_preset(slot, db_path=None):
    """Pin whatever is playing. ⚠️ Only what is ACTUALLY playing.

    `slot` may be None — "add this to my presets" names no button, and
    demanding one is what made the whole feature unreachable. The next free
    one is used and named back.
    """
    from . import radio
    now = radio.playing()
    if not now:
        return ("Nothing is playing, so there is nothing to save. Start a "
                "station first, then say “save as preset”.")
    if slot is None:
        slot = presets.free_slot(db_path=db_path)
        if slot is None:
            return (f"All {presets.SLOTS} buttons are taken. Say “save as "
                    f"preset 4” to replace one.")
    ok, msg, displaced = presets.save(slot, now, db_path=db_path)
    return msg if ok else f"I could not: {msg}"


def _list_presets(db_path=None):
    """⚠️ EMPTY AND UNREADABLE ARE DIFFERENT SENTENCES, as everywhere here."""
    rows = presets.all_presets(db_path=db_path)
    if not rows:
        return (f"No presets yet. There are {presets.SLOTS} buttons — play a "
                f"station and say “save as preset 1”.")
    listed = "; ".join(f"{p['slot']}: {p['name']}" for p in rows)
    free = presets.SLOTS - len(rows)
    return (f"{listed}."
            + (f" {free} button{'s' if free != 1 else ''} still free."
               if free else " All ten are set."))


def _remember(kind, name, detail=None, db_path=None):
    """Note that something was played. ⚠️ Best effort; never breaks a turn.

    ⚠️ `detail` IS THE STREAM URL AND IT IS NOT OPTIONAL IN PRACTICE. It is
    what makes "that station from earlier" replayable rather than merely
    nameable.
    """
    try:
        from . import dj
        dj.remember_play(kind=kind, name=name, detail=detail, db_path=db_path)
    except Exception:                                        # noqa: BLE001
        pass


def _ask_which(req):
    """⚠️ THE HONEST ANSWER TO AN UNCLEAR REQUEST. Not a default."""
    what = req.get("what")
    if what:
        return (f"Music, a station, a podcast or a film for “{what}”? "
                f"I did not want to guess and start the wrong one.")
    return ("Music, radio, a podcast or a film? I would rather ask than "
            "start the wrong thing.")


def answer(req, db_path=None):
    """Do it, and say what was done. Returns a sentence, or None."""
    if not req:
        return None
    verb = req.get("verb")

    if verb == "what":
        return now_playing_line(db_path=db_path) or "Nothing is playing."

    # ⚠️ PLAYLISTS, REVIEWS AND RATINGS ARE DECLINED AT THE DOOR. All three
    # would read Music.app, and all three belong to the streaming-library shape
    # Mati does not have. A car radio has buttons, not a star rating.
    #
    # ⚠️ The ONE permitted reading lives elsewhere and is not reachable from
    # here: `dj.import_taste()`, run once at setup, which records genres so a
    # first set of presets can be suggested.
    if verb in ("playlists", "reviews", "rate"):
        return ("I do not read your music library any more — the radio is "
                "stations and presets now. Ask me for a station, or save one "
                "to a preset.")

    if verb == "setup_music":
        # ⚠️ THE ONLY PATH TO THE LIBRARY, and it refuses to run twice.
        from . import dj
        ok, msg = dj.import_taste()
        return msg if ok else f"I could not: {msg}"

    if verb == "next":
        return _next_station(db_path=db_path)

    if verb == "back":
        from . import radio
        st, why = radio.back_found()
        if st is None:
            return why
        return _play_one(st, db_path=db_path)

    if verb == "nth":
        from . import radio
        st = radio.nth_found(req.get("n"))
        if st is None:
            n = radio.found_count()
            return (f"I only found {n} station(s) last time."
                    if n else "I do not have a list of stations to pick from.")
        return _play_one(st, db_path=db_path)

    if verb == "listing":
        from . import radio
        rows = radio.listing()
        if not rows:
            return "I have not looked for stations yet this session."
        return "; ".join(
            f"{r['n']}: {r['name']}"
            + (f" ({r['km']:.0f} km)" if isinstance(r.get("km"), (int, float))
               else "")
            for r in rows) + ". Say a number, or “next station”."

    if verb == "played_when":
        return _played_when(req.get("when"), db_path=db_path)

    if verb == "presets":
        return _list_presets(db_path=db_path)

    if verb == "save_preset":
        return _save_preset(req.get("slot"), db_path=db_path)

    if verb == "stop":
        return stop_all(db_path=db_path)

    if verb in ("louder", "softer"):
        from . import radio
        cur = radio.volume()
        if cur is None:
            return "Nothing is playing here to adjust."
        lvl = radio.volume(cur + (0.15 if verb == "louder" else -0.15))
        return f"Volume {int(round(lvl * 100))}%."

    if verb != "play":
        return None

    # ⚠️ A PRESET NUMBER OUTRANKS EVERYTHING. "play 3" is a button, not a
    # search for a station called 3.
    slot = req.get("slot")
    if slot is not None:
        return _play_preset(slot, db_path=db_path)

    kind = req.get("kind")
    if kind is None:
        # ⚠️ NO LIBRARY LOOK-UP. A named thing is a station name to look for in
        # the directory.
        named = req.get("what")
        if named:
            return _play_radio(dict(req, genre=None), db_path=db_path)
        return _ask_which(req)

    what = req.get("what") or ""
    try:
        if kind == "radio":
            return _play_radio(req, db_path=db_path)
        if kind == "track":
            # ⚠️ MUSIC IS A STATION NOW. `track` stays as a RECOGNISER so
            # "play some jazz music" is still understood — what changes is
            # where it goes. The kind is kept rather than deleted because
            # deleting it would make "play a song" unrecognised entirely,
            # which is the failure `test_media` exists to prevent.
            return _play_radio(req, db_path=db_path)
        if kind == "podcast":
            return _find_podcast(what)
        if kind == "film":
            return _find_film(what, feature=req.get("feature"),
                              documentary=req.get("documentary"))
    except Exception as e:                                   # noqa: BLE001
        # ⚠️ One backend failing is one sentence, never a broken turn.
        return f"I could not start that ({type(e).__name__})."
    return None


def _play_radio(req, db_path=None):
    """Find stations and start the first that plays. ⚠️ Keeps the rest.

    ⚠️ NEAREST FIRST, WHEN WE KNOW WHERE YOU ARE. `radio.near` narrows to the
    saved country and orders by distance, so a local station comes before one
    across an ocean. It returns [] when no location is set, and a plain search
    is the fallback rather than an error — a person who has not told Mati
    where they are should still get music.
    """
    from . import radio, dj
    want = req.get("genre") or req.get("what") or ""
    if not want:
        return _switch_on(db_path=db_path)

    found = radio.near(tag=want, limit=6) or []
    by_tag = bool(found)
    if not found:
        found = radio.by_genre(want, limit=6) or []
        by_tag = bool(found)
    if not found:
        # ⚠️ THE NAME PATH, AND THE ONLY ONE A TITLE CHECK APPLIES TO.
        # `near` and `by_genre` match on TAGS, and a jazz station is rarely
        # NAMED jazz — "Blue Note Radio" and "Smooth Lounge FM" are correct
        # answers whose titles share nothing with the request. Checking
        # titles there would reject the right station. Resemblance is only
        # meaningful against the thing that was actually searched.
        found = radio.search(want, limit=6) or []
        if found and _best_match(want, found, key="name") is None:
            return _ask_instead(want, "station")
    if not found:
        return _ask_instead(want, "station")

    # ⚠️ HELD BEFORE PLAYING, so "the next station" works even if the first
    # one fails to start.
    radio.remember_found(found, wanted=want)
    return _start_from(found, want, db_path=db_path)


def _ask_instead(want, what_of="station"):
    """Nothing matched — so ASK, rather than report a dead end.

    Ask instead of trying to play random text, and instead of telling someone
    their request was not found: both are one rule. A request that found
    nothing is usually not a request for something that does not exist — it is
    a request Mati READ WRONG, and "I could not find a station for X" hands
    that back as the person's problem.

    This matters more because the 3B classifies intent (`Mati/intent.py`): a
    message can be labelled `play` on good evidence while the words after the
    verb are not the name of anything. The right answer asks the one thing that
    would make the search work.

    ⚠️ IT NAMES WHAT IT DID NOT UNDERSTAND. "I do not know it" is no more use
    than "not found" — quoting the words back is what lets the person see that
    Mati heard "a documentary about teotihuacan" as a station name and say
    so.

    ⚠️ AND IT IS NOT `_switch_on`. That is the other case: "play some music"
    names NOTHING, and the answer there is to PLAY. Nothing named means pick
    something; something named that missed means ask.
    """
    w = " ".join((want or "").split())
    if not w:
        return _switch_on()
    return (f"I do not know “{w}” as a {what_of}. Is it a genre, a place, "
            f"or the name of something — or did I take the wrong part of "
            f"what you said?")


def _switch_on(db_path=None):
    """"Play some music" with nothing named. ⚠️ PLAYS, it does not interrogate.

    "play some music" means music, not "A station of what? Name a genre, a
    country or a station."

    ⚠️ "AN UNCLEAR REQUEST IS ASKED ABOUT" BELONGS ELSEWHERE. That rule exists
    so audio never starts unbidden, and it earns its keep on "play something",
    where the KIND is unknown — a film, a podcast and a station are different
    rooms. But "play some music" is not unclear. The kind is known. A car
    radio's power button plays what was on last; it does not ask which genre.

    The order is a radio's own: what was last on, then the lowest button, then
    — only with no history and no presets at all — a question.
    """
    from . import radio, dj
    # 1. What was on last. ⚠️ Needs the URL, not just the name: the history
    #    records both, and a name alone leaves "that station from earlier"
    #    unplayable.
    try:
        for row in dj.history(limit=20, db_path=db_path):
            if row.get("kind") in ("station", "preset") and row.get("detail"):
                ok, msg = radio.play({"name": row["name"],
                                      "url": row["detail"]})
                if ok:
                    _remember(kind="station", name=row["name"],
                              detail=row["detail"], db_path=db_path)
                    return f"{msg}. Back on what you had last."
                break        # it was there and would not play; fall through
    except Exception:                                        # noqa: BLE001
        pass

    # 2. The lowest button that has something on it.
    rows = presets.all_presets(db_path=db_path)
    if rows:
        return _play_preset(rows[0]["slot"], db_path=db_path)

    # 3. ⚠️ THE IMPORTED GENRES — AND THIS IS WHAT THE IMPORT IS FOR, so Mati
    #    has a starting point. A cold radio with no history and no presets is
    #    exactly the moment that one-time read exists to answer; without this
    #    branch it would be stored and never read.
    liked = dj.taste()
    if liked:
        for genre in liked[:3]:
            found = (radio.near(tag=genre, limit=6)
                     or radio.by_genre(genre, limit=6))
            if found:
                radio.remember_found(found, wanted=genre)
                said = _start_from(found, genre, db_path=db_path)
                # ⚠️ SAYS WHERE THE CHOICE CAME FROM. Mati picking a genre
                # out of nowhere is a guess; picking one because it is in the
                # library is a reason, and the person can correct it.
                return f"{said} You have a lot of {genre} in your library."

    # 4. ⚠️ ONLY NOW A QUESTION, and it says what would fix it.
    if liked is None:
        return ("Nothing has been on yet and no presets are set. I have not "
                "read your music library — say “set up my music” and I will "
                "take the genres from it once, so I have somewhere to start.")
    return ("Nothing has been on yet, no presets are set, and your library "
            "gave me no genres to go on. Name one — “play some jazz” — and "
            "I will save it to a button.")


def _start_from(found, want, db_path=None):
    """Play the first of `found` that starts. Shared by search and `next`."""
    from . import radio, dj
    for st in found:
        ok, msg = radio.play(st)
        if ok:
            radio.mark_played(st)
            dj.remember_play(kind="station", name=st["name"],
                             detail=st["url"], db_path=db_path)
            km = st.get("km")
            where = f" — {km:.0f} km away" if isinstance(km, (int, float)) else ""
            more = radio.remaining()
            tail = (f" Say “next station” for one of {more} more."
                    if more > 0 else "")
            return f"{msg}{where}.{tail}"
    return f"I found {len(found)} stations for “{want}” and none would start."


def _play_one(st, db_path=None):
    """Play a station already in the held list, and move the cursor to it."""
    from . import radio
    ok, msg = radio.play(st)
    if not ok:
        return f"{st.get('name')} would not play — {msg}"
    radio.mark_played(st)
    _remember(kind="station", name=st["name"], detail=st.get("url"),
              db_path=db_path)
    km = st.get("km")
    where = f" — {km:.0f} km away" if isinstance(km, (int, float)) else ""
    return f"{msg}{where}."


def _next_station(db_path=None):
    """Walk the last search. ⚠️ KEEPS WALKING past one that will not play.

    One station from the list handed to `_start_from`, and failure reported if
    it did not start, would let a single dead stream end the walk with the
    cursor sitting still — a list of six unusable after the first bad entry,
    and every "next station" answered "none would start".

    ⚠️ AND NO "None" IN THE MESSAGE. `_start_from` called with `want=None`
    would interpolate it, and a person reading that learns nothing.
    """
    from . import radio
    tried = []
    while True:
        st, rest = radio.next_found()
        if st is None:
            if tried:
                # ⚠️ NAMES WHAT WAS TRIED. "none would start" after silently
                # skipping four stations is a different fact from "the list
                # is finished", and only one of them suggests the network.
                return (f"{rest} I tried {len(tried)} that would not start.")
            return rest                  # the explanation, not a station
        ok, msg = radio.play(st)
        if ok:
            radio.mark_played(st)
            _remember(kind="station", name=st["name"], detail=st.get("url"),
                      db_path=db_path)
            km = st.get("km")
            where = (f" — {km:.0f} km away"
                     if isinstance(km, (int, float)) else "")
            more = radio.remaining()
            tail = (f" Say “next station” for one of {more} more."
                    if more > 0 else "")
            skipped = (f" (skipped {len(tried)} that would not start)"
                       if tried else "")
            return f"{msg}{where}.{skipped}{tail}"
        # Advance the cursor past the dead one so the loop makes progress.
        radio.mark_played(st)
        tried.append(st.get("name"))


def _played_when(when_words, db_path=None):
    """"that station from yesterday". ⚠️ From the RECORD, never a guess —
    `dj.what_played` reads the history that is recorded as things play.
    """
    from . import radio, dj
    row = dj.what_played(when_words, db_path=db_path)
    if not row:
        return (f"I do not have a station in my record for “{when_words}”. "
                f"Name a genre and I will find one.")
    url, name = row.get("detail"), row.get("name")
    if not url:
        return f"I played {name} then, but I did not keep the station address."
    ok, msg = radio.play({"name": name, "url": url})
    if not ok:
        return f"That was {name}, but it would not play now — {msg}"
    dj.remember_play(kind="station", name=name, detail=url, db_path=db_path)
    return f"{msg}. That is what you had on {when_words}."


def _find_podcast(what):
    from . import podcast
    if not what:
        return "Which show?"
    shows = podcast.search(what, limit=3)
    if not shows:
        return _ask_instead(what, "podcast")
    # ⚠️ THE TOP HIT IS NOT AUTOMATICALLY THE ANSWER. Asked for "Theories of
    # Everything", gpodder can rank a show sharing not one word with the request
    # first, and taking it would report on something else with no sign anything
    # had gone wrong.
    #
    # ⚠️ A SEARCH RANKING IS NOT A MATCH. gpodder returns its best guess
    # whatever is asked; it never says "nothing like this exists". So the
    # resemblance has to be checked here, by the caller that knows what was
    # wanted. Same rule the radio follows: ask rather than play something
    # random and call it an answer.
    s = _best_match(what, shows)
    if s is None:
        return _ask_instead(what, "podcast")
    try:
        eps = podcast.episodes(s["feed"], limit=1)
    except podcast.FeedUnavailable as e:
        return f"{s['title']} is listed but its feed would not load ({e})."
    if not eps:
        return f"{s['title']} has no episodes I can reach."
    # ⚠️ A TUPLE NOW. `podcast.play` returned a bare bool while it went through
    # Music.app; it goes through `radio.play` and reports WHY it failed, which
    # is the difference between "would not play" and "the feed is down".
    ok, msg = podcast.play(eps[0])
    if not ok:
        return f"I found {s['title']} but the episode would not play — {msg}"
    _remember(kind="podcast", name=f"{eps[0]['title']} — {s['title']}",
              detail=eps[0].get("url"))
    return f"{eps[0]['title']} — {s['title']}."


def _find_film(what, feature=None, documentary=None):
    """⚠️ Openly-licensed only, and the app plays it — see archive_org.

    ⚠️ THE REQUEST MAY NAME A SHELF RATHER THAN A TITLE. "play nasa videos" is
    a request to browse the NASA collection, not to search the feature-film
    shelf for the word "nasa" — which finds *Voyage to the Planet of
    Prehistoric Women*. `shelf_for` spots the shelf.

    ⚠️ AND A BARE REQUEST SPLITS TWO WAYS. Unless someone is looking for
    something different, "film" and "movie" mean a studio picture, so "play a
    movie" gets one and "play some videos" gets the looping reel — same kind,
    different intent, and `feature` is which word was used.
    """
    from . import archive_org as A

    # ⚠️ A BARE DOCUMENTARY REQUEST GETS THE ARCHIVAL SHELF. Prelinger is
    # educational and ephemeral film — actual documentary material — where the
    # studio list is comedies and thrillers.
    if not (what or "").strip() and documentary:
        what = "prelinger"

    if not (what or "").strip() and feature:
        # ⚠️ A NAMED REQUEST STILL SEARCHES EVERYTHING. This is the default for
        # someone who did not say what they wanted, which is exactly the case
        # the curated list is for — "unless someone is looking for something
        # different".
        picks = A.studio_pick(limit=1)
        if picks:
            p = picks[0]
            return {"say": f"{p['title']} — {p['studio']}, {p['licence']}.",
                    "play_video": p}

    # ⚠️ NOTHING NAMED MEANS THE REEL: a short playlist of shorts and cartoons
    # from the era, looping, so it can play at the same time as music. The
    # single most-downloaded feature film would be arbitrary, and often has no
    # playable file.
    if not (what or "").strip():
        reel = A.reel(limit=6)
        if reel:
            total = sum(r.get("megabytes") or 0 for r in reel)
            return {"say": (f"{len(reel)} shorts on a loop, silent era — "
                            f"starting with {reel[0]['title']}."),
                    "play_reel": reel, "reel_mb": round(total)}
        # Fall through to the ordinary search if the reel came back empty.

    # ⚠️ WHO CHOSE DECIDES WHETHER IT IS SCREENED, and that is the whole rule.
    # Nothing inappropriate should play by accident, and any available film
    # should play when asked for by name, subject or genre.
    #
    # A shelf browsed with NOTHING NAMED is Mati choosing, and Mati choosing
    # can land on an exploitation short. A shelf plus a search term, or a bare
    # title, is the PERSON choosing, and is never screened.
    shelf, rest = A.shelf_for(what)
    if shelf:
        # A wider pull, because screening removes rows and a list of five
        # could screen down to nothing.
        films = A.search(rest or None, collection=shelf,
                         limit=5 if rest else 20)
        if not rest:
            films = [f for f in (films or [])
                     if A.ok_unattended(A.describe(f))]
        what = rest or shelf
    else:
        # `what` is non-empty here — the empty cases returned above — so this
        # is always a named request and is deliberately NOT screened.
        films = A.search(what or None, limit=5)
    if films is None:
        return "I only look for films on shelves I know: " + ", ".join(A.collections())
    if not films:
        return (f"Nothing on the public-domain shelves for “{what}”."
                if what else "Nothing came back from the film shelves.")
    # ⚠️ A NAME SEARCH, SO THE NAME HAS TO MATCH. Same rule as the podcast
    # path: a search engine returns its best guess whatever it is asked and
    # never says "nothing like this exists", so the resemblance is checked
    # by the caller that knows what was wanted.
    f = _best_match(what, films, key="title") if what else films[0]
    if f is None:
        return _ask_instead(what, "film")
    got, why = A.playable(f["id"])
    if not got:
        return f"{A.describe(f)} is listed but {why}."
    # ⚠️ The URL goes to the caller, which hands it to the page. Nothing opens.
    return {"say": f"{A.describe(f)} — {got['licence']}.", "play_video": got}


# ── everything at once ───────────────────────────────────────────────────────

def stop_all(db_path=None):
    """Silence everything Mati started. ⚠️ Immediate, and that is the point.

    Stop is the one command where latency IS the problem. Nothing of Mati's
    lives in another application, so stopping waits on nothing: the player is
    in this process, and `radio.stop()` is a pause and a dropped reference.
    """
    from . import radio
    was = radio.stop()
    return f"Stopped {was['name']}." if was else "Nothing was playing."


def pause_all():
    from . import radio
    was = radio.pause()
    return f"Paused {was['name']}." if was else "Nothing was playing."


def resume_all():
    from . import radio
    ok, msg = radio.resume()
    return f"Playing {msg}." if ok else f"Could not resume: {msg}."


def paused_now(db_path=None):
    """The paused station, shaped like `playing_now`, or None.

    ⚠️ KEPT APART FROM `playing_now`, whose truth is "sound is coming out".
    Folding a paused station into it would make every caller that reads it as
    "the radio is on" wrong.
    """
    from . import radio
    p = radio.paused()
    if not p:
        return None
    slot = None
    try:
        for x in presets.all_presets(db_path=db_path):
            if x["url"] == p.get("url"):
                slot = x["slot"]
                break
    except Exception:                                        # noqa: BLE001
        pass
    return {"name": p.get("name"), "country": p.get("country"),
            "bitrate": p.get("bitrate"), "preset": slot,
            "tuning": bool(p.get("tuning"))}


def now_playing_line(db_path=None):
    """One sentence about what is on, or None. ⚠️ Asks, never remembers.

    ⚠️ ONLY WHAT MATI IS PLAYING. Asking Music.app would report a track the
    PERSON started there as though Mati were playing it, and cost an
    AppleScript round trip on every status check, including the ones the
    indicator makes.
    """
    from . import radio
    now = radio.playing()
    if not now:
        return None
    return f"{now['name']}" + (f", {now['country']}"
                               if now.get("country") else "")


def playing_now(db_path=None):
    """Structured state for the app's indicator, or None.

    ⚠️ ASKS THE PLAYER, and `radio.playing()` only answers yes at
    ReadyToPlay — so this cannot show a station as on while the room is
    silent, which is the bug the indicator exists to make visible.
    """
    from . import radio
    now = radio.playing()
    if not now:
        return None
    slot = None
    try:
        for p in presets.all_presets(db_path=db_path):
            if p["url"] == now.get("url"):
                slot = p["slot"]
                break
    except Exception:                                        # noqa: BLE001
        pass
    return {"name": now.get("name"), "country": now.get("country"),
            "bitrate": now.get("bitrate"), "preset": slot}


def status(db_path=None):
    from . import radio
    return {"playing": now_playing_line(db_path=db_path),
            "kinds": [k for k, _ in KINDS],
            "radio": radio.status(),
            "note": "'play' does not assume music; an unclear request is "
                    "asked about rather than guessed"}
