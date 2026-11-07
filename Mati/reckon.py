"""
Logic and arithmetic — computed, not guessed.

A 3B model is unreliable at exactly this class of question: it produces
confident, wrong numbers and waves through invalid inferences. `field.math`
already implements it deterministically (`VirtualQC.solve`), so Mati routes
these questions there and hands the *result* to the language model to express.

Design rules:

  * ROUTE, don't replace. The math engine computes; the LLM still does the
    talking. Users get a sentence, not a dict.
  * ONLY route when confident. A false positive is worse than a miss — sending
    "what do you know about prime numbers?" to a factoriser produces nonsense.
    Patterns here are deliberately narrow and anchored.
  * COMPUTED IS NOT REMEMBERED, and it is not the model's opinion either. A
    computed result is a third kind of content: it is checkable. It is labelled
    as such so Mati can say "I worked it out" rather than "I know" or
    "I remember".

Two limits are enforced here rather than in the engine, because both are about
what survives a *conversation* rather than what is mathematically true. See
`test_reckon.py`, which measures both.
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Field-based Shor factorisation spawns a 2N-node field. That used to cost N²
# in memory and worse in time — N=8051 took 8m17s at 2.1GB — so this cap was
# 1500. The field now stores its coupling sparsely, which made N=8051 866×
# faster (0.57s), and the cap moved out by a decimal digit.
#
# Worst of 12 seeds, measured 2026-07-28 (attempt count is stochastic, so the
# tail is what matters, not the median):
#     N=10403 → 1.5s    N=15251 → 3.3s    N=20989 → 5.9s
# 20989 breaks the chat-turn ceiling; 15000 sits comfortably under it. Above
# the cap we decline and let the model give an ordinary (hedged) answer rather
# than freeze the conversation.
_MAX_FACTOR_N = 15000

# A number the model cannot restate accurately is not "stated plainly" — it is
# transcribed, badly. 50! is correct and also 65 digits that a 3B model will
# silently corrupt while paraphrasing, which the no-hedge instruction then tells
# it not to re-derive. Long *numbers* are dropped rather than asserted; a
# structured result (a truth table's flags) is summarised, not transcribed, so
# it isn't subject to this.
_MAX_RESULT_DIGITS = 40


# Narrow, anchored patterns. Each must look like a request to COMPUTE something,
# not a request to talk about it. `_ANTI` blocks the conversational forms.
_ANTI = re.compile(
    r"\b(what (is|are) (a|an|the)?\s*(prime|factor|logic|set)\b"
    r"|tell me about|explain|why (is|do|does)|how (do|does) .* work"
    r"|remember|discussed|talked about)",
    re.I,
)

def _factor_problem(m):
    """Decline oversized N — see `_MAX_FACTOR_N`."""
    return None if int(m.group(1)) > _MAX_FACTOR_N else f"factor {m.group(1)}"


def _truth_table_problem(m):
    """Strip the `for` keyword; decline if nothing is left to tabulate.

    Doing this here rather than in the pattern because a regex can always
    backtrack around an optional keyword: `(?:\\s+for\\b)?\\s+(.+)` happily
    gives up the optional group and captures "for" itself. A bare
    "truth table for" then got a confident answer about a variable named `for`.
    """
    expr = re.sub(r"^for\b\s*", "", m.group(1).strip(), flags=re.I).strip()
    return f"truth table {expr}" if expr else None



# Plain arithmetic — the gap that sent "3 + 4 * 2" to the language model.
#
# ⚠️ THE ROUTING IS THE RISKY HALF, not the evaluation. Arithmetic has no
# keyword to anchor on, so the pattern has to recognise a SHAPE — and dates,
# versions, scores and page ranges all have that shape. `2026-08-03` is a
# perfectly good subtraction. So:
#
#   - `+` and `-` and `/` must be SPACED. "2026-08-03" and "08/03" do not
#     fire; "17 - 4" does. This deliberately misses "144/12", which is the
#     price of not answering dates with arithmetic.
#   - `*` and `^` may be unspaced — neither appears in a date or a version.
#   - at least two numbers and one operator, and the whole match must parse.
#
# A miss costs an unanswered sum. A false positive means Mati confidently
# announces that today's date is 2015.
_ARITH = re.compile(r"""
    (?<![\w.])                       # not mid-number, not mid-version
    \(?\s*                          # an expression may open with a bracket
    \d+(?:\.\d+)?
    (?:
        \s*(?:\*\*|\*|\^)\s*     # * and ^ may be unspaced
      | \s+[-+/%]\s+                # + - / % must be spaced
      | \s*\*\s*
    )
    [\d.\s()+\-*/%^]*\d
    (?![\w.])
""", re.X)

# A bracketed expression may use unspaced operators. Brackets do not occur in
# dates, versions, scores or page ranges, so they are a strong enough signal of
# intent to lift the spacing rule that keeps "2026-08-03" out.
_ARITH_PAREN = re.compile(
    r"(?:[\d.]+\s*[-+*/%^]\s*)?\(\s*[\d.][\d\s.+\-*/%^()]*\)"
    r"[\d\s.+\-*/%^()]*")

# Dates and versions that survive the spacing rule anyway.
_NOT_ARITH = re.compile(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b|\bv?\d+\.\d+\.\d+\b")


# Characters that mean the match is a FRAGMENT of a longer expression.
_ADJACENT = set("0123456789.+-*/%^(")


def _arith_problem(m):
    """Build an `evaluate` problem, or decline if it will not parse cleanly."""
    expr = m.group(0).strip()
    if _NOT_ARITH.search(expr):
        return None

    # ⚠️ A PARTIAL MATCH IS A WRONG ANSWER, NOT A MISS. `2*(3+4)` matched only
    # `(3+4)` and returned 7 — confidently, with a `computed` badge on it.
    # That is the one failure mode this whole module exists to prevent, so the
    # match must not be a fragment: whatever sits either side of it, ignoring
    # spaces, must not be part of the same sum.
    text = m.string
    before = text[:m.start()].rstrip()
    after = text[m.end():].lstrip()
    if before and before[-1] in _ADJACENT:
        return None
    if after and after[0] in _ADJACENT.union(")"):
        return None
    if expr.count("(") != expr.count(")"):
        return None                    # unbalanced: we caught half of one
    # `^` means power to a person typing a sum; in Python it is XOR.
    expr = expr.replace("^", "**")
    if not re.search(r"\d\s*(?:\*\*|[-+*/%])\s*\(?\s*\d", expr):
        return None                    # a lone number is not a calculation
    from .field.math import _safe_arith
    return f"evaluate {expr}" if _safe_arith(expr) is not None else None


_PATTERNS = [
    # arithmetic / number theory
    (re.compile(r"\bfactor(?:ise|ize)?\s+(\d{2,})\b", re.I),        _factor_problem),
    (re.compile(r"\bis\s+(\d+)\s+(?:a\s+)?prime\b", re.I),          lambda m: f"is prime {m.group(1)}"),
    (re.compile(r"\bgcd\s+(?:of\s+)?(\d+)\D+(\d+)", re.I),          lambda m: f"gcd {m.group(1)} {m.group(2)}"),
    (re.compile(r"\blcm\s+(?:of\s+)?(\d+)\D+(\d+)", re.I),          lambda m: f"lcm {m.group(1)} {m.group(2)}"),
    (re.compile(r"\b(\d+)\s+choose\s+(\d+)\b", re.I),               lambda m: f"choose {m.group(1)} {m.group(2)}"),
    (re.compile(r"\bfactorial\s+(?:of\s+)?(\d+)\b", re.I),          lambda m: f"factorial {m.group(1)}"),
    # formal logic
    (re.compile(r"\btruth\s*table\b\s+(.+)", re.I),                 _truth_table_problem),
    (re.compile(r"\b(?:are|is)\s+(.+?)\s+equivalent\s+to\s+(.+)", re.I),
                                                                    lambda m: f"equiv {m.group(1)} ; {m.group(2)}"),
    (re.compile(r"\b(?:does|is it valid that)\s+(.+?)\s+therefore\s+(.+)", re.I),
                                                                    lambda m: f"inference {m.group(1)} therefore {m.group(2)}"),
    (re.compile(r"\binference\s+(.+)", re.I),                       lambda m: f"inference {m.group(1)}"),

    # LAST. Every pattern above anchors on a keyword and is more specific; this
    # one matches a shape, so it must only see what nothing else claimed.
    (_ARITH_PAREN,                                                  _arith_problem),
    (_ARITH,                                                        _arith_problem),
]


def route(text):
    """Return a `VirtualQC.solve` problem string, or None if this isn't one."""
    if _ANTI.search(text):
        return None
    for pat, build in _PATTERNS:
        m = pat.search(text)
        if m:
            return build(m)
    return None


def reckon(text):
    """
    Compute if this is a computable question, else None.

    Returns {'computation', 'result', 'explanation', 'problem'} — never raises
    into the turn. A math failure must degrade to a normal LLM answer, not break
    the conversation.
    """
    problem = route(text)
    if problem is None:
        return None
    try:
        from .field.math import VirtualQC
        out = VirtualQC.solve(problem)
    except Exception:                           # engine is unaudited — contain it
        return None
    if out is None or out.get("result") is None:
        return None
    # Engine reports parse failures inside `explanation` rather than raising.
    expl = out.get("explanation") or ""
    if expl.lstrip().startswith("["):
        return None
    # Correct but untranscribable — see `_MAX_RESULT_DIGITS`.
    r = out["result"]
    if isinstance(r, int) and not isinstance(r, bool):
        if len(str(abs(r))) > _MAX_RESULT_DIGITS:
            return None
    return out


def block(res):
    """Render a computed result for the model to express in its own words."""
    return (
        "<computed>\n"
        f"{res['computation']} = {res['result']}\n"
        f"{res.get('explanation', '')}\n"
        "This was COMPUTED deterministically, not recalled and not generated. "
        "It is correct — state the result plainly and do not hedge it, "
        "re-derive it, or contradict it.\n"
        "</computed>"
    )
