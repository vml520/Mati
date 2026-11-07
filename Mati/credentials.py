"""
Keys the person brings themselves. One answer to a wall three modules hit.

eBay's Browse API, UCP's agent profile and the event APIs (Ticketmaster,
Bandsintown) are all FREE and all need a credential. Each time, the same
thing was true and it was written down three separate times:

    ⚠️ A SHARED KEY BAKED INTO A PUBLIC APACHE-2.0 BUILD IS NOT A KEY.

Anyone can read it out of the source, it gets revoked the first time someone
abuses it, and every install dies together. So there is exactly one rule here
and it resolves all three at once:

    MATI SHIPS NO KEYS. Every key-gated feature is OFF until the person
    puts in their own, and says plainly where to get one and what it costs.

⚠️ SEALED AT REST. Keys go through `vault.seal`, like note bodies and
calendar titles, so a copied `mati_credentials.json` is not a stolen key.
The service NAMES stay clear, so Mati can tell you what is configured
without unlocking anything.

⚠️ "LOCKED" AND "NOT SET" MUST NOT COLLAPSE INTO ONE ANSWER. `get()` raises
`vault.Locked` when the vault is shut and returns None when a key was never
added. A feature that says "you haven't set up eBay" at a Touch ID prompt is
lying, and it is the same failure shape as reporting "$0 spent" for an
envelope nobody has touched.

⚠️ NOTHING HERE IS EVER IMPRINTED OR RECALLED. A key is not a memory. It is
not in `notes`, it never reaches θ₀, and `forget()` really deletes it.
"""

from __future__ import annotations

import json
import os

from . import config as C

PATH = os.path.join(C.CACHE_DIR, "mati_credentials.json")


class Unknown(KeyError):
    """No such service. ⚠️ Not the same as a service with no key yet."""


# ⚠️ A LITERAL REGISTRY, not an open dictionary. A typo'd service name should
# be an error, not a key stored under a name nothing will ever read — which
# is the "built with no reader" shape in miniature.
#
# `cost` is stated because the standing constraint is NO PAID APIs, and a
# free tier that quietly became paid is exactly how Yelp caught people out.
SERVICES = {
    "ticketmaster": {
        "what": "concerts and live events near a place",
        "where": "developer.ticketmaster.com — free developer account",
        "cost": "free tier",
        "field": "Consumer Key",
    },
    "bandsintown": {
        "what": "tour dates for specific artists",
        "where": "artists.bandsintown.com/support — request an app_id",
        "cost": "free, but commercial use needs their approval",
        "field": "app_id",
    },
    "ebay": {
        "what": "eBay listings, including ones filtered to near you",
        "where": "developer.ebay.com — free developer account",
        "cost": "free tier",
        "field": "App ID (Client ID)",
    },
    "foursquare": {
        "what": "places with richer detail than OpenStreetMap",
        "where": "location.foursquare.com/developer",
        "cost": "free tier",
        "field": "API Key",
    },
}

# ⚠️ NAMED AND REFUSED, not silently absent. Someone who read the same survey
# will ask for these, and "I don't know that service" is a worse answer than
# saying why it is not offered.
REFUSED = {
    "yelp": "Yelp Fusion has no free tier — $299/month minimum since 2024. "
            "Mati uses OpenStreetMap for places instead, which has no "
            "ratings, so it lists what is NEAR rather than what is best.",
    "google": "Google Places bills per request with a card on file. "
              "OpenStreetMap covers the same ground for free.",
}


def known():
    """Every service Mati can use a key for."""
    return sorted(SERVICES)


def about(service):
    """What a service gives, where to get a key, what it costs. Raises Unknown."""
    s = str(service or "").strip().lower()
    if s in REFUSED:
        raise Unknown(REFUSED[s])
    if s not in SERVICES:
        raise Unknown(f"I don't use a service called '{service}'. "
                      f"I know: {', '.join(known())}.")
    return dict(SERVICES[s], service=s)


def _read():
    try:
        with open(PATH, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(doc):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, PATH)
    try:
        os.chmod(PATH, 0o600)            # ⚠️ owner only, like the vault file
    except OSError:
        pass


def have(service):
    """Is a key stored? ⚠️ Answerable with the vault LOCKED — names stay clear."""
    s = str(service or "").strip().lower()
    return s in _read()


def configured():
    """Which services have a key. ⚠️ No unlocking, no key material read."""
    return sorted(_read())


def set_key(service, value):
    """Store a key, sealed. Returns the service name, or None if refused."""
    from . import vault
    info = about(service)                 # raises Unknown for bad names
    value = " ".join(str(value or "").split())
    if not value:
        return None
    doc = _read()
    # ⚠️ `seal` falls back to plaintext when the vault is not set up, which is
    # right for note bodies and WRONG for a credential. Refuse instead.
    if not vault.is_set_up():
        raise vault.VaultError(
            "the vault is not set up, and a key must not be written in the "
            "clear — run `python3 -m Mati.vault setup` first")
    if not vault.unlocked():
        vault.unlock(reason=f"store your {info['service']} key")
    doc[info["service"]] = vault.seal(value)
    _write(doc)
    return info["service"]


def get(service):
    """The key, or None if never set. ⚠️ RAISES `vault.Locked` when shut.

    The distinction is the whole point: "you have not set this up" and "I
    cannot read it right now" are different sentences, and collapsing them
    sends someone off to create an account they already have.
    """
    from . import vault
    s = str(service or "").strip().lower()
    stored = _read().get(s)
    if stored is None:
        return None
    if not vault.unlocked():
        raise vault.Locked(f"the {s} key is stored, but the vault is locked")
    return vault.open_body(stored)


def forget(service):
    """Delete a key. ⚠️ Really deletes — a revoked key must leave no copy."""
    s = str(service or "").strip().lower()
    doc = _read()
    if s not in doc:
        return False
    doc.pop(s)
    _write(doc)
    return True


def needed(service):
    """The sentence Mati says when a feature is off for want of a key."""
    try:
        i = about(service)
    except Unknown as e:
        return str(e)
    return (f"{i['what'].capitalize()} needs a {i['field']} from "
            f"{i['where']} ({i['cost']}). Nothing is shipped with Mati — "
            f"add yours with `/key set {i['service']} <value>`.")


def status():
    have_keys = configured()
    return {
        "configured": have_keys,
        "available": [s for s in known() if s not in have_keys],
        "refused": sorted(REFUSED),
        "path": PATH,
        "note": "Mati ships no keys; every key-gated feature is off until "
                "the person adds their own",
    }
