"""
First run — lock the memories, and tell Mati who you are.

    python3 -m Mati.setup
    python3 -m Mati.setup --profile-only     # add facts later, keep the key
    python3 -m Mati.setup --show

Two jobs, and they belong together because both happen exactly once, before
Mati is any use.

1. THE VAULT. A key is made, wrapped into the keychain behind Touch ID and
   again behind a printed recovery passphrase, and every memory already stored
   is encrypted. See `Mati/vault.py` for why the check lives at the key and
   not at the app door.

2. WHO YOU ARE. A fresh install has no episodes, which is the correct state and
   also a bad first impression — ask it anything and it truthfully says it does
   not remember. Setup fixes that honestly rather than by faking a history: you
   tell Mati a few things about yourself, and they are imprinted into θ₀.

WHY THE PROFILE IS NOT LEFT TO RECALL. Each fact is imprinted (so the field is
genuinely shaped by knowing you) AND stored as a note that goes into every turn
unconditionally. The second part is what makes it reliable: retrieval here is
lexical, so "what is my name" would not dependably surface the note containing
it. An assistant that sometimes forgets who it is talking to is worse than one
that never knew, and this is cheap insurance against that.

Imprinting is ~1s per token, so the profile is deliberately a handful of short
facts rather than a biography.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import bridge
from . import config as C
from . import exclusive
from . import inquiry
from . import vault

# Asked in this order. Only the first is required — a name is what makes the
# difference between "I don't remember" and "I don't remember, Sam", and the
# rest are things people actually want an assistant to hold onto.
# ⚠️ "Anything else Mati should ALWAYS KNOW about you?" was cut.
# A program that opens by asking what it should always know about you reads as
# a program building a file on you — and the phrasing pressed for something
# permanent and important at the exact moment someone is least sure what they
# want to hand over. It is now an optional bio, in their words, at whatever
# length they like, or nothing at all.
QUESTIONS = [
    ("name",      "What should Mati call you?",                       True),
    # Separate from what to CALL them: someone called Sam may be Samantha on
    # a form, and Mati should use the first while knowing the second.
    ("fullname",  "And your full name? (optional)",                    False),
    ("work",      "What do you work on? (one line)",                   False),
    ("interests", "What are you interested in? (a few topics)",        False),
    ("use",       "What will you use Mati for?",                      False),
    ("bio",       "Anything you'd like to tell Mati about yourself? "
                  "(optional)",                                        False),
    ("style",     "How do you want it to talk to you? (e.g. plainly, "
                  "no filler)",                                        False),
]

# The answers that seed the first dream. Interests and intended use describe
# where conversations will actually go; name and talking style do not.
DREAM_SEED_FIELDS = ("interests", "use", "work")


def _fact(field, answer):
    """One profile fact, phrased as something the user said, in their words."""
    answer = " ".join(answer.split())
    return {
        "name":      f"The person I talk to is called {answer}.",
        "fullname":  f"Their full name is {answer}.",
        "work":      f"They work on {answer}.",
        "interests": f"They are interested in {answer}.",
        "use":       f"They will use me for {answer}.",
        "bio":       f"They told me this about themselves, in their words: "
                     f"{answer}",
        "style":     f"They asked me to talk to them like this: {answer}",
    }[field]


def ask(prompt, required=False):
    while True:
        try:
            answer = input(f"  {prompt}\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if answer or not required:
            return answer
        print("  (needed)")


def _pulled_models():
    """Which models Ollama already has. ([], reason) if it cannot be reached."""
    import json as _json
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(C.OLLAMA_URL + "/api/tags", timeout=4) as r:
            tags = _json.loads(r.read().decode())
    except urllib.error.URLError as e:
        return [], f"cannot reach Ollama at {C.OLLAMA_URL} ({e.reason})"
    except Exception as e:                       # malformed reply, wrong service
        return [], f"unexpected reply from {C.OLLAMA_URL}: {e}"
    out = []
    for m in tags.get("models", []):
        name = m.get("name")
        if name:
            out.append((name, int(m.get("size") or 0)))
    return out, ""


def choose_model():
    """Ask which local model Mati should speak through, and remember it.

    Mati's LLM supplies COMPOSITION only — identity, memory and provenance
    live elsewhere — so the choice is about fluency and speed, not character.
    What it does NOT do is change where anything goes: every option here runs
    on this machine, which is why this step exists at all rather than a list of
    hosted APIs.
    """
    models, why = _pulled_models()
    if not models:
        print(f"  Mati speaks through a local model, and {why}.")
        print("  Install it from https://ollama.com, then:\n")
        print("      ollama pull llama3.2:3b\n")
        print(f"  Keeping '{C.LOCAL_MODEL}' for now — Mati will say so plainly")
        print("  if it still cannot reach it when you start talking.\n")
        return None

    print("  Mati speaks through a language model on THIS machine.")
    print("  Nothing you say is sent anywhere. Models already installed:\n")
    cur = C.LOCAL_MODEL
    default_i = 1
    for i, (name, size) in enumerate(models, 1):
        gb = f"{size/1e9:.1f} GB" if size else "    ?"
        mark = "  <- in use" if name == cur else ""
        if name == cur:
            default_i = i
        print(f"    {i}. {name:<28} {gb}{mark}")
    print()
    print("  Bigger is more fluent and slower, and it shares memory with the")
    print("  field simulation. On 8 GB, a 3B is the comfortable ceiling.\n")

    ans = ask(f"Which one? (1-{len(models)}, return for {models[default_i-1][0]})")
    pick = models[default_i - 1][0]
    if ans:
        try:
            k = int(ans)
            if 1 <= k <= len(models):
                pick = models[k - 1][0]
            else:
                print(f"  {k} is not on the list — keeping {pick}.")
        except ValueError:
            # A name rather than a number is a reasonable thing to type.
            match = [n for n, _ in models if n == ans or n.startswith(ans)]
            if match:
                pick = match[0]
            else:
                print(f"  No model matches '{ans}' — keeping {pick}.")

    # Verify it actually answers before writing the choice down. A model that
    # is pulled but broken should be found here, not on the first real turn.
    print(f"\n  Checking {pick} responds…")
    try:
        from .llm import LocalLLM
        ok, detail = LocalLLM("You are a test.", model=pick).available()
    except Exception as e:
        ok, detail = False, str(e)
    if not ok:
        print(f"  {pick} did not check out: {detail}")
        print(f"  Leaving {C.LOCAL_MODEL} in place.\n")
        return None

    # ⚠️ THE CHOICE TOO, not only `local_model`: once a service has been
    # chosen, `model_choice` outranks `local_model`, and running this step
    # again would otherwise change nothing.
    C.save_settings(local_model=pick, provider="local",
                    model_choice=f"local:{pick}")
    print(f"  {detail}\n")
    print(f"  Mati will speak through {pick}. Change it any time with")
    print("  `python3 -m Mati.setup --model`, or for one run only:")
    print("      MATI_MODEL=other:tag python3 -m Mati.server")
    print("  In the window, with the terminal switched on, the arrow beside")
    print("  the model's name changes it too. Mati can connect to most")
    print("  popular models.\n")
    return pick


def choose_name():
    """What this person wants to call their assistant. Skippable.

    ⚠️ THE NAME IS THE PERSON'S CHOICE: Mati is a personal AI. Without this
    step, `assistant_name` would be settable only by hand-editing the settings
    JSON, which is not a choice anyone will make.

    ⚠️ THE PRODUCT STAYS MATI, and saying so here prevents the obvious
    confusion. The app, the commands and the docs do not change; what changes
    is what it calls ITSELF. A person who renames it and then cannot find
    `python3 -m Mati.setup` has been badly served by a half-explained option.
    """
    from . import config as _C
    current = _C.assistant_name()
    print(f"  This assistant is called {current}. You can call it something")
    print("  else — it is yours. The app and the commands stay 'Mati'.")
    answer = ask(f"What should it be called? (return keeps {current})",
                 required=False)
    if answer is None:                   # backed out of setup entirely
        return None
    name = " ".join((answer or "").split())
    if not name or name == current:
        print(f"  Keeping {current}.\n")
        return current
    # ⚠️ THE SAME CEILING AS `config.assistant_name`, checked here so the
    # person is told rather than silently ignored later.
    if len(name) > 32:
        print("  That is too long for a name — keeping "
              f"{current}.\n")
        return current
    _C.save_settings(assistant_name=name)
    print(f"  It will call itself {name} from now on.\n")
    return name


def _confirm_recorded(phrase, only):
    """Make the risk explicit and get an acknowledgement. True if given.

    ⚠️ A BARE RETURN IS NOT ENOUGH. "Write this down before continuing",
    continued on an empty line, makes the honest answer and the reflexive one
    the same keystroke — and an unrecorded passphrase is lost the one time it
    is needed.

    ⚠️ IT ASKS FOR UNDERSTANDING, NOT PROOF. Making the person type the
    passphrase back verifies more, but it is friction nobody asked for, and
    the thing actually missing is a plain statement of what gets lost. This
    says it, and asks them to say they have read it.

    ⭐ AND IT NAMES THE DURABLE ANSWER. A warning is not a remedy;
    `--recovery-kit` is, because it stops a remembered passphrase being the
    only way in.
    """
    print("  ⚠️  WHAT YOU CAN LOSE, PLAINLY\n")
    if only:
        print("  This passphrase is the ONLY key to Mati's memories. There is")
        print("  no copy in the keychain and no Touch ID fallback. If you lose")
        print("  it, every memory Mati has is unreadable — permanently, by")
        print("  anyone, including you. It cannot be reset or recovered, and")
        print("  it will not be shown again.\n")
    else:
        print("  Touch ID opens Mati day to day, so you will rarely type")
        print("  this. But it is the only way back if the keychain is ever")
        print("  lost — a wiped Mac, a new machine, a reset Touch ID. Without")
        print("  it, every memory is unreadable permanently. It cannot be")
        print("  reset or recovered, and it will not be shown again.\n")
    print("  ⭐ So that this is not your only copy, after setup:\n")
    print("        python3 -m Mati.vault --recovery-kit <path>\n")
    print("  That writes the key to a file you keep somewhere else — a drive")
    print("  in a drawer, a password manager — so remembering a passphrase")
    print("  stops being the only way in.\n")
    try:
        got = input("  Type 'I understand' to continue: ").strip().lower()
    except EOFError:
        # ⚠️ NON-INTERACTIVE: say so loudly rather than hanging a harness.
        print("\n  ⚠️ nothing on stdin — risk not acknowledged. If this was")
        print("  not a test, write the passphrase down NOW.\n")
        return False
    if got.replace("'", "").replace("\u2019", "") in ("i understand", "understand"):
        print()
        return True
    # ⚠️ NOT A LOOP. They have read it; refusing to proceed would only make
    # them kill the process with a half-made vault.
    print("\n  ⚠️ Not acknowledged — continuing anyway. The passphrase is")
    print("  above and this is the last time it is shown.\n")
    return False


def choose_place():
    """Ask roughly where the person is, for weather. Optional, and skippable.

    ⚠️ THIS ANSWER DOES NOT GO INTO THE FIELD, and that is the whole reason it
    is asked here rather than in `QUESTIONS`. Every profile answer is imprinted
    into θ₀ and injected on every turn; a field cannot forget, so "delete my
    location" would be a promise nothing could keep. This one lands in
    `mati_settings.json` as four keys that `weather.forget_place()` genuinely
    removes.

    ⚠️ It is also the only setup answer that will ever leave the machine, so it
    says so before asking, not after.
    """
    from . import weather

    print("  Mati can tell you the weather. To do that it has to ask a")
    print("  weather service about a place — the ONLY thing in Mati that")
    print("  goes out to the internet on your behalf.\n")
    print("  A town or city is enough. It is rounded to about a kilometre")
    print("  before it is saved, kept out of the field so it can be deleted,")
    print("  and never sent with anything else. Press return to skip.\n")

    name = ask("Roughly where are you? (town or city, optional)")
    if not name:
        print("  Skipped — Mati will say it doesn't know rather than guess.\n")
        return None
    try:
        hits = weather.find_place(name)
    except weather.Unavailable as e:
        # ⚠️ An outage is not a bad place name. Saying "no such place" here
        # would teach someone their own city does not exist.
        print(f"  {e}. Skipping for now; set it later with `/weather set`.\n")
        return None
    if not hits:
        print(f"  Nothing matched '{name}'. You can set it later with")
        print("  `/weather set <place>`.\n")
        return None

    pick = hits[0]
    if len(hits) > 1:
        print(f"\n  More than one place is called that:\n")
        for i, h in enumerate(hits, 1):
            print(f"    {i}. {h['label']}")
        ans = ask(f"\nWhich one? (1-{len(hits)}, return for 1)")
        if ans:
            try:
                k = int(ans)
                if 1 <= k <= len(hits):
                    pick = hits[k - 1]
                else:
                    print(f"  {k} is not on the list — using {pick['label']}.")
            except ValueError:
                print(f"  Not a number — using {pick['label']}.")

    label = weather.set_place(pick)
    p = weather.where() or {}
    print(f"\n  Weather for {label}.")
    print(f"  Saved as {p.get('latitude')}, {p.get('longitude')} — about a")
    print(f"  kilometre, in {p.get('units')} units. Remove it any time with")
    print("  `/weather forget`.\n")
    return label


def check_encyclopedia():
    """Report whether an offline encyclopedia is usable. Asks nothing.

    ⚠️ THIS STEP DOES NOT ASK A QUESTION, AND THAT IS DELIBERATE. It needs a
    binary Mati does not ship and a file of tens or hundreds of megabytes the
    person downloads themselves — so there is no answer that could be acted on
    here, and "would you like an encyclopedia?" during first run is a line in
    a wizard about a feature they have no use for yet. The setup principle is
    that a question is asked at the moment it blocks something; `asks.py`
    raises this one the first time a lookup actually fails.

    ⚠️ SO WHAT THIS DOES IS DETECT AND REPORT. If both pieces are already
    present it says so and the feature simply works. If not, it says the
    feature exists, in two lines, and moves on.
    """
    from . import encyclopedia

    st = encyclopedia.status()
    if st["connected"]:
        mb = st["bytes"] / 1e6
        print(f"  Offline encyclopedia: {len(st['zims'])} file(s), "
              f"{mb:.0f} MB. Mati can search these without going online.\n")
        return True
    print("  Offline encyclopedia: not set up, and optional.")
    print("  Mati can search Wikipedia offline if you install `kiwix-serve`")
    print(f"  and drop a .zim file in {st['zim_dir']}.")
    print("  Subject builds are small — mathematics 56 MB, physics 51 MB —")
    print("  and nothing goes online once a file is there. `/encyclopedia`")
    print("  in the terminal shows what is missing at any time.\n")
    return False


def gather():
    """Collect profile facts. Returns (facts, seed_text) or None if backed out."""
    facts, seeds = [], []
    for field, prompt, required in QUESTIONS:
        answer = ask(prompt, required)
        if answer is None:
            return None
        if answer:
            facts.append(_fact(field, answer))
            if field in DREAM_SEED_FIELDS:
                seeds.append(" ".join(answer.split()))
        print()
    return facts, " ".join(seeds)


def first_dream(lm, tok, seed_text, cycles=None, verbose=True):
    """Let Mati sleep on what it just learned about you. OFF by default.

    `DreamingLoop` free-dreams from whatever θ₀ currently is — "a resonance of
    whatever the field IS right now", per its own docstring — and deepens the
    attractors that settle coherently. The idea was that dreaming right after
    imprinting the person's interests would deepen those regions.

    ⚠️ IT DOES THE OPPOSITE: free dreaming erodes a fresh imprint faster than
    a topic never mentioned, so `config.SETUP_DREAM_CYCLES` is 0. This is kept
    so that can be re-measured, with its control topic, not because it helps.
    It never teaches Mati anything about those topics either: the field has no
    knowledge of Mesoamerican archaeology or protein folding; it has a shape.
    """
    cycles = C.SETUP_DREAM_CYCLES if cycles is None else cycles
    if not seed_text or cycles <= 0:
        return None

    # Control topic: unrelated to anything the user said, so a coherence change
    # on the seed can be read against a change on something it should not touch.
    control = "the tide tables for a harbour in winter"
    before = (bridge.field_state(lm, tok, seed_text)[1],
              bridge.field_state(lm, tok, control)[1])

    if verbose:
        print(f"  Sleeping on it — {cycles} dream cycles.", file=sys.stderr)
    out = bridge.dream(lm, tok, cycles=cycles, verbose=verbose)

    after = (bridge.field_state(lm, tok, seed_text)[1],
             bridge.field_state(lm, tok, control)[1])
    return {"cycles": cycles, "dream": out,
            "seed_coherence": (before[0], after[0]),
            "control_coherence": (before[1], after[1])}


def imprint_profile(lm, tok, facts, reindex=True, verbose=True):
    """Store each fact as a durable, always-present note. ~30s each.

    `reindex=False` when a dream follows — dreaming deforms θ₀ again and
    reindexes afterwards, so repairing the index twice is wasted work on the
    most expensive operation there is.
    """
    stored = []
    for fact in facts:
        if verbose:
            print(f"  learning: {fact}", file=sys.stderr)
        res = bridge.remember(lm, tok, fact, source=bridge.PROFILE_SOURCE)
        if res:
            stored.append(res[0])
    if stored and reindex:
        # Same rule as studying and dreaming: imprinting moved θ₀, so every
        # stored address is stale until the index is put back in one space.
        bridge.reindex(lm, tok)
        bridge.save_notes_field(lm)
    return stored


def show():
    facts = bridge.profile_notes()
    print(f"  vault      : {'set up' if vault.is_set_up() else 'NOT set up'}")
    print(f"  biometrics : {'available' if vault.biometrics_available() else 'no'}")
    if not facts:
        print("  profile    : empty — Mati does not know who you are")
        return 0
    print(f"  profile    : {len(facts)} fact(s)")
    for f in facts:
        print(f"    - {f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Set Mati up for the first time")
    ap.add_argument("--profile-only", action="store_true",
                    help="add facts about yourself; leave the vault alone")
    ap.add_argument("--vault-only", action="store_true",
                    help="encrypt the memories; skip the questions")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--model", action="store_true",
                    help="choose which local model Mati speaks through")
    ap.add_argument("--force", action="store_true",
                    help="run even if Mati holds the field (unsafe)")
    args = ap.parse_args()

    if args.model:
        # Standalone: changing the voice touches no memory, so it needs
        # neither the vault nor the field lock.
        print("\n  Mati — language model\n")
        choose_model()
        return 0

    if args.show:
        if vault.is_set_up() and not vault.unlocked():
            vault.unlock(reason="show what Mati knows about you")
        return show()

    # Setup imprints θ₀ and rewrites stored bodies. Mati must not be holding
    # the field while that happens.
    rc = exclusive.guard("setup", force=args.force)
    if rc:
        return rc

    print("\n  Mati — first run\n")

    # ── 0. the machine ───────────────────────────────────────────────────────
    # ⚠️ BEFORE ANY QUESTION IS ASKED. Step 0: these are not
    # preferences, they are whether Mati can run at all — and a 4 GB Mac
    # should be told rather than left to discover it by being slow. This is
    # also the honest-latency moment: said here, nothing later has to
    # apologise for 55-102s replies on an Intel Mac.
    if not args.profile_only:
        from . import machine as _mach
        _prof = _mach.profile()
        print(_mach.describe(_prof))
        print()
        _stop = _mach.blockers(_prof)
        for _b in _stop:
            print(f"  ⚠️ {_b}\n")
        for _w in _mach.warnings(_prof):
            print(f"  {_w}\n")
        if _stop:
            # ⚠️ A BLOCKER IS NOT A WARNING. Carrying on past "no model
            # pulled" produces a setup that completes and an Mati that
            # cannot answer, which is the failure this step exists to catch.
            print("  Fix the above and run setup again.\n")
            return 1
        # ⚠️ THE AVATAR DEFAULT COMES FROM THE MACHINE, not from a constant —
        # off on a slow machine. Written once, here, and freely changeable in
        # the app afterwards — this sets the STARTING point, not a policy.
        try:
            if not C._settings().get("avatar"):
                C.save_settings(avatar=_mach.avatar_default(_prof))
        except Exception:
            pass

    # ── 1. the vault ─────────────────────────────────────────────────────────
    if not args.profile_only:
        if vault.is_set_up():
            print("  Memories are already encrypted. Leaving the key alone.")
            print("  (`python3 -m Mati.vault --setup --force` replaces it, "
                  "which makes\n   every existing memory unreadable.)\n")
        else:
            # ⚠️ NO TOUCH ID MUST NOT MEAN NO ENCRYPTION. Every Mac is offered
            # the choice, and declining is a decision someone made rather than
            # a branch they never saw — otherwise an install would keep every
            # memory in the clear while the privacy wording read the same as
            # everywhere else.
            biometrics = vault.biometrics_available()
            if biometrics:
                print("  Lock Mati's memories?")
                print("    1  Touch ID, with a recovery passphrase  "
                      "(recommended)")
                print("    2  passphrase only")
                print("    3  not yet")
                pick = ask("1, 2 or 3", required=False) or "1"
            else:
                # ⚠️ ON A MAC WITH NO TOUCH ID MATI OPENS ITS OWN KEY. The
                # fingerprint was only ever a check that someone is present;
                # the key is protected by the login keychain either way. A
                # passphrase typed at every start is still offered, for
                # someone who wants the key kept off the machine.
                print("  There is no Touch ID on this Mac. Lock Mati's "
                      "memories?")
                print("    1  yes — Mati keeps the key in your keychain and "
                      "opens it\n       by itself  (recommended)")
                print("    2  passphrase only — typed every time Mati starts")
                print("    3  not yet — memories stay readable on this disk")
                pick = ask("1, 2 or 3", required=False) or "1"

            pick = pick.strip()[:1]
            if pick == "3":
                # ⚠️ SAY WHAT DECLINING MEANS. "Skipped" is not informative;
                # the consequence is.
                print("\n  Leaving memories unencrypted. Anything Mati "
                      "remembers is\n  readable by anything that can read "
                      "this disk. You can change\n  this later with "
                      "`python3 -m Mati.vault --setup`.\n")
            else:
                only = (pick == "2")
                phrase = vault.setup(passphrase_only=only,
                                     presence_check=biometrics)
                n = vault.migrate()
                # Sibling tables too, not just `notes` — see SEALED_COLUMNS.
                n += vault.migrate_tables(verbose=False)
                vault.harden_permissions()
                print("  " + "=" * 60)
                print("  Mati's memories are encrypted now.")
                print("\n  RECOVERY PASSPHRASE — write this down before "
                      "continuing:\n")
                print(f"      {phrase}\n")
                if only:
                    # ⚠️ A DIFFERENT WARNING, because the stakes are different.
                    # With no keychain copy this is the ONLY key in existence —
                    # there is no Touch ID fallback to be casual about.
                    print("  This passphrase is the ONLY way into Mati's "
                          "memories. There\n  is no key in the keychain — "
                          "that is what makes this mode\n  worth anything. "
                          "Lose it and the memories are gone for good.\n  "
                          "Nothing can recover it for you, and it will not be "
                          "shown again.")
                    print("\n  Mati starts locked each time. Open it with the "
                          "passphrase:\n      python3 -m Mati.vault --unlock")
                elif biometrics:
                    print("  Touch ID unlocks Mati day to day. This "
                          "passphrase is the\n  only way back if the keychain "
                          "is ever lost — a wiped Mac\n  or a new machine. "
                          "Nothing can recover it for you, and it\n  will "
                          "not be shown again.")
                else:
                    # ⚠️ SAY WHAT THIS DOES AND DOES NOT PROTECT. A mode that
                    # needs nothing from the person is easy to mistake for
                    # one that protects against everything.
                    print("  Mati opens its own key from your keychain when "
                          "it starts, so\n  there is nothing to type. That "
                          "protects these memories if the\n  files leave this "
                          "Mac — a backup, a copied folder, another\n  "
                          "account, a lost disk. It does not protect them from "
                          "software\n  running as you.")
                    print("\n  This passphrase is the only way back if the "
                          "keychain is ever\n  lost — a wiped Mac or a new "
                          "machine. Nothing can recover it for\n  you, and it "
                          "will not be shown again.")
                print("  " + "=" * 60)
                if n:
                    print(f"\n  {n} existing memories encrypted.")
                _confirm_recorded(phrase, only)

    if args.vault_only:
        return 0

    # ── 1b. the voice ────────────────────────────────────────────────────────
    # Before the questions: the answers get imprinted, and a first run that
    # cannot reach a model should say so here rather than after the user has
    # typed out who they are.
    if not args.profile_only:
        choose_model()
        choose_name()

    # ── 2. who you are ───────────────────────────────────────────────────────
    if vault.is_set_up() and not vault.unlocked():
        vault.unlock(reason="set up Mati")

    existing = bridge.profile_notes()
    if existing:
        print(f"  Mati already knows {len(existing)} thing(s) about you:")
        for f in existing:
            print(f"    - {f}")
        if (ask("Add more? (y/N)") or "n").lower() not in ("y", "yes"):
            return 0
        print()

    print("  A few questions, so Mati knows who it is talking to.")
    print("  Only the first is needed — press return to skip any other.\n")
    gathered = gather()
    if not gathered or not gathered[0]:
        print("  Nothing added. Mati will meet you as it goes.")
        return 0
    facts, seed_text = gathered

    # ⚠️ Asked BEFORE the imprint, so all the typing happens together — the
    # imprint below takes about half a minute per fact, and coming back to ask
    # one more question after a two-minute wait reads as an afterthought.
    try:
        choose_place()
    except Exception as e:                                   # noqa: BLE001
        print(f"  Skipping the weather question ({e}).\n")

    try:
        check_encyclopedia()
    except Exception as e:                                   # noqa: BLE001
        print(f"  Skipping the encyclopedia check ({e}).\n")

    print(f"  Learning {len(facts)} thing(s) — this is a real imprint into the")
    print("  field, so it takes about half a minute each.\n")
    lm, tok = bridge.load_field()
    stored = imprint_profile(lm, tok, facts, reindex=not seed_text)

    # The interests become things Mati will look into on its own when idle —
    # asking, comparing against what it holds, asking again. NOT free dreaming,
    # which erodes a fresh imprint rather than deepening it.
    topics = []
    for field, _prompt, _req in QUESTIONS:
        if field in ("interests", "use"):
            for fact in facts:
                if fact.startswith(("They are interested in",
                                    "They will use me for")):
                    topics += inquiry.add_topics_from(
                        fact.split(" in ", 1)[-1].split(" for ", 1)[-1])
    topics = sorted(set(topics))
    if topics:
        print("\n  Mati will look into these on its own when idle:")
        for t in topics:
            print(f"    - {t}")

    result = None
    if seed_text:
        print()
        result = first_dream(lm, tok, seed_text)
        bridge.save_notes_field(lm)

    print(f"\n  Done — {len(stored)} learned.")
    # ⚠️ NO PROMISE OF PERMANENT RECALL. "Mati will know these from now on, in
    # every conversation", said at the moment someone hands over their name,
    # reads as surveillance rather than reassurance — and what matters is that
    # the person stays in control of it.
    # ⚠️ Only promise what exists. `show()` is READ-ONLY, so a line offering to
    # "change or remove any of it with --show" would be a false assurance about
    # control over personal data.
    print("  See what Mati knows about you any time with")
    print("      python3 -m Mati.setup --show")
    if result:
        s0, s1 = result["seed_coherence"]
        c0, c1 = result["control_coherence"]
        print(f"\n  It slept on it for {result['cycles']} cycles. Settling on")
        print(f"  what you told it: {s0:.4f} -> {s1:.4f}")
        print(f"  on an unrelated topic:  {c0:.4f} -> {c1:.4f}")
        print("  (One observation, not evidence — the dream shapes where the")
        print("   field sits, it does not teach Mati about your topics.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
