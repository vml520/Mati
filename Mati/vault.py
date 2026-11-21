"""
Mati's memories, encrypted at rest.

    python3 -m Mati.vault --setup       # once: make a key, print recovery phrase
    python3 -m Mati.vault --status
    python3 -m Mati.vault --unlock      # open it, and the running Mati with it
    python3 -m Mati.vault --export out.txt

WHY NOT JUST PROMPT BEFORE OPENING MATI. Because that protects nothing. The
episodes live in `.cache/mati_notes.db`, a plain SQLite file — anyone with a
shell could read every one of them without Mati running. And Mati ships open
source, so a prompt in Python is a prompt anyone can delete.

So the protection is a KEY, not a prompt: without the key the bodies are
ciphertext, and editing this source does not change that.

WHERE THE KEY LIVES. Three modes, chosen at setup and recorded in the vault
file (`unlock_method()` reads it back):

    touch_id     the key is in the login keychain, and Mati asks for a
                 fingerprint before reading it
    keychain     the key is in the login keychain, and Mati reads it when it
                 starts, unattended — for a Mac with no Touch ID
    passphrase   the key is nowhere on the machine; someone types the
                 passphrase each time Mati starts

⚠️ TOUCH ID HERE IS A PRESENCE CHECK, NOT ENFORCEMENT. The keychain item is an
ordinary generic password written and read by `/usr/bin/security`, and any
process running as the person reads it the same way with no fingerprint. A
biometric rule on the item itself needs a signed app with a keychain
entitlement, which a Python package cannot have. So `touch_id` and `keychain`
protect exactly the same things and differ only in whether someone must be
present when Mati starts. Only `passphrase` keeps the key off the machine.

WHAT IS AND IS NOT ENCRYPTED. Note bodies. Not the addresses.

    note_id | source | feat (plain) | body (ciphertext)

`recall` ranks on `feat`, the trajectory-feature address, and only needs the
body text for the handful of notes it actually surfaces. So the field keeps
working untouched, retrieval never needs the key, and the content is
unreadable at rest. Encrypting the whole database instead would have meant
SQLCipher (a dependency Mati does not have) or decrypting to a temp file —
which leaves plaintext on disk while running and after a crash.

THREAT MODEL, stated plainly. This protects against another account on the
machine, a stolen or imaged disk, a copied or synced `.cache/`, and Time
Machine backups. It does NOT protect against you: you can always unlock, and
anything you can unlock, software running as you can unlock too — in the two
keychain modes at any time, in `passphrase` mode while Mati is running. That
is the correct model for a personal AI, and overstating it would be worse than
not having it.

RECOVERY. The key is kept twice — in the keychain (except in `passphrase`
mode), and wrapped with a passphrase printed at setup. A wiped Mac or a new
machine loses the keychain copy; the passphrase is the only way back. Nothing
here can recover a lost key, and it does not pretend to.

CRYPTO. Nothing invented: AES-256-CBC via `/usr/bin/openssl` (LibreSSL, present
on every Mac), encrypt-then-MAC with HMAC-SHA256, PBKDF2 for the passphrase
wrap. The key never appears in argv — it goes to openssl on a file descriptor,
because argv is visible in `ps`.
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import config as C

OPENSSL = "/usr/bin/openssl"           # ships with macOS; not the Homebrew one
# ⚠️⚠️ NOT RENAMED WITH THE PRODUCT, AND THAT IS DELIBERATE.
#
# The data key lives in the macOS keychain UNDER THIS NAME. Changing it does
# not move the key — it orphans it, and the only person who would notice is
# someone who already had a vault, and then could not open it.
#
# ⭐ A KEYCHAIN IS NOT A PATH: there is no rename, only a write under a new
# name and an entry nobody looks for under the old one. Nothing a user ever
# sees contains this string, so consistency here buys nothing and risks
# everything. The tests pin it.
KEYCHAIN_SERVICE = "ollin-vault"
KEYCHAIN_ACCOUNT = "data-key"
# ⚠️ What the constants were at import. Anything else means a caller
# redirected them on purpose, and that intent outranks the vault's record.
_DEFAULT_KEYCHAIN = (KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT)
# ⚠️ C.CACHE_DIR, NOT C.ROOT/.cache, so it follows MATI_HOME like every other
# path in the package. Hardcoded to the repo's own `.cache`, three things would
# break:
#
#   - a sandbox or clean room would see the REAL vault as set up and try to
#     seal with a key its own store has no descriptor for;
#   - `transfer.manifest()` carries `mati_vault.json`, so backup/restore and
#     the thumb-drive install would write the descriptor into CACHE_DIR while
#     this module read somewhere else entirely — a portable install that looks
#     fine and is unopenable;
#   - `python3 -m Mati.sandbox` could not produce an unencrypted scratch copy
#     at all.
#
# On an install with no MATI_HOME set, CACHE_DIR *is* ROOT/.cache.
VAULT_PATH = os.path.join(C.CACHE_DIR, "mati_vault.json")
# The path as shipped. `_keychain_is_safe` compares against this so a
# redirected VAULT_PATH cannot reach the real keychain account.
_DEFAULT_VAULT_PATH = VAULT_PATH

# ⚠️⚠️ A DATA FORMAT MARKER, NOT A PRODUCT NAME — AND IT MUST NEVER BE
# RENAMED. Every sealed body on disk begins with this string, so changing it
# does not re-tag anything: it makes `is_encrypted()` answer False for the
# whole store, `open_body()` pass ciphertext through as if it were plaintext,
# and `ollin1:U2FsdGVkX1...` land in the prompt and in front of the person.
#
# It is pinned for the same reason `KEYCHAIN_SERVICE` is: nothing a user sees
# contains it, and the cost of changing it is their memories turning to
# gibberish.
#
# ⚠️ READ ACCEPTS EVERY HISTORICAL SPELLING, WRITE USES ONE. A store sealed
# under an older build must keep opening; `PREFIX` is what new writes get.
PREFIX = "ollin1:"                    # marks an encrypted body — DO NOT RENAME
KNOWN_PREFIXES = ("ollin1:", "olin1:", "mati1:")
PBKDF2_ITERS = 600_000                 # for the recovery passphrase only

# Short, unambiguous words. Deliberately not a dictionary file — a recovery
# phrase you cannot regenerate is worse than one with a smaller word list, and
# 5 words from 128 plus a number is ~37 bits, ample against an offline attacker
# who must also get the wrapped blob off the machine first.
_WORDS = """tenoch basin quetzal ember jade obsidian codex river maize copal
serpent turquoise cactus eagle drum feather lake mirror smoke flint reed jaguar
temple ash amber cinder plume gourd hearth ibis kernel lantern marsh nectar
onyx pollen quarry ridge salt thorn umber vine willow xerus yucca zenith arch
bell chalk dune elm fern glyph husk inlet kiln loam moss nook opal pine quill
rush stone tide urn vale wick yarn zephyr anvil brine cedar delta echo frost
grove haze iris juniper kelp lichen mica north oak petal quartz root sage
tundra under vault warp axis yield zone amber birch clay dusk east fjord gorge
hill isle jetty knoll ledge mesa nadir oasis peak quay reef spur trail upland
vista wold crest""".split()


class VaultError(RuntimeError):
    pass


class Locked(VaultError):
    """No key in memory. Unlock first."""


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

def _openssl(args, data, password):
    """Run openssl with the password on fd 3 — never in argv, which `ps` shows."""
    r_fd, w_fd = os.pipe()
    try:
        # `pass_fds` keeps the descriptor open in the child but does NOT
        # renumber it, so the argument has to name the fd we actually got.
        # A hardcoded `fd:3` silently refers to nothing.
        os.write(w_fd, password.encode() + b"\n")
        os.close(w_fd)
        w_fd = None
        proc = subprocess.run(
            [OPENSSL, "enc", *args, "-pbkdf2", "-iter", "1000",
             "-pass", f"fd:{r_fd}"],
            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            pass_fds=(r_fd,))
    finally:
        if w_fd is not None:
            os.close(w_fd)
        os.close(r_fd)
    if proc.returncode != 0:
        raise VaultError(f"openssl failed: {proc.stderr.decode()[:200]}")
    return proc.stdout


def _subkey(data_key, label):
    return hmac.new(data_key, label, hashlib.sha256).digest()


# ⚠️⚠️ FORMAT CONSTANTS, NOT PRODUCT NAMES — THE SAME RULE AS `PREFIX`. The
# subkeys are HMAC(data_key, label), so the label is part of the key
# derivation: change one letter and the right key derives the wrong subkeys,
# every MAC fails, and the error says "wrong key" — about the right key, with
# nothing wrong except this string.
#
# ⭐ WRITE ONE, READ ALL — same rule as `KNOWN_PREFIXES`. A store sealed under
# any historical build must keep opening. The MAC is checked under each family
# in turn; that costs nothing in security, because every family's subkey is
# still derived from the data key, so a forgery needs the key under any label.
LABEL_FAMILY = "ollin"                 # what new writes get — DO NOT RENAME
KNOWN_LABEL_FAMILIES = ("ollin", "olin", "mati")


def _subkeys(data_key, family):
    return (_subkey(data_key, f"{family}-body-enc".encode()).hex(),
            _subkey(data_key, f"{family}-body-mac".encode()))


def encrypt(data_key, text):
    """Encrypt-then-MAC. Returns a self-contained string safe to store."""
    enc, mac = _subkeys(data_key, LABEL_FAMILY)
    blob = _openssl(["-aes-256-cbc"], text.encode("utf-8"), enc)
    tag = hmac.new(mac, blob, hashlib.sha256).digest()
    return PREFIX + base64.b64encode(blob).decode() + ":" + \
        base64.b64encode(tag).decode()


def decrypt(data_key, stored):
    """Inverse of `encrypt`. Verifies the MAC BEFORE decrypting."""
    if not is_encrypted(stored):
        return stored                          # not yet migrated
    try:
        # ⚠️ STRIP THE PREFIX THAT IS ACTUALLY THERE, not `len(PREFIX)`.
        # The historical spellings are different lengths — `ollin1:` is 7,
        # `olin1:` and `mati1:` are 6 — so a fixed strip eats a character of
        # base64 from one of them and raises "corrupt ciphertext" on a body
        # that is perfectly intact.
        _p = next(p for p in KNOWN_PREFIXES if stored.startswith(p))
        blob_b64, tag_b64 = stored[len(_p):].split(":", 1)
        blob = base64.b64decode(blob_b64)
        tag = base64.b64decode(tag_b64)
    except (ValueError, base64.binascii.Error) as e:
        raise VaultError(f"corrupt ciphertext: {e}") from e
    for family in KNOWN_LABEL_FAMILIES:
        enc, mac = _subkeys(data_key, family)
        if hmac.compare_digest(hmac.new(mac, blob, hashlib.sha256).digest(),
                               tag):
            return _openssl(["-aes-256-cbc", "-d"], blob, enc).decode("utf-8")
    raise VaultError("ciphertext failed its integrity check — wrong key, "
                     "or the stored note was tampered with")


def is_encrypted(text):
    # ⚠️ ANY KNOWN SPELLING. Checking only the current one is exactly how
    # 20,444 sealed bodies read as plaintext.
    return isinstance(text, str) and text.startswith(KNOWN_PREFIXES)


# ─────────────────────────────────────────────────────────────────────────────
# Where the key lives
# ─────────────────────────────────────────────────────────────────────────────

def _keychain_ids():
    """(service, account) for THIS vault. Reads the vault's own record.

    ⚠️⚠️ ONE KEYCHAIN ENTRY PER INSTALL. `KEYCHAIN_SERVICE`/`KEYCHAIN_ACCOUNT`
    are module constants, identical in every copy of Mati; read from those, a
    person who set up a second copy would silently replace the first one's
    data key, and the first vault would no longer open — by a route
    `_keychain_is_safe` cannot see, since for a second install VAULT_PATH and
    its own default match.

    ⭐ THE VAULT FILE RECORDS ITS OWN service/account, and this reads the
    record.

    ⚠️ LEGACY INSTALLS KEEP WORKING. A vault written before per-install
    accounts has `{"service": "mati-vault", "account": "data-key"}` in it, so
    the record and the constants agree and nothing moves. A vault with no
    `keychain` block at all is passphrase-only and never reaches here.
    """
    try:
        doc = _read_vault()
    except Exception:
        doc = None
    # ⚠️ AN EXPLICIT OVERRIDE BEATS THE RECORD. A caller that has deliberately
    # pointed the constants somewhere — every test harness does — means it. If
    # the doc won unconditionally, redirecting the constants would look like
    # isolation and silently operate on whatever vault happened to be at
    # VAULT_PATH. That is how the real key got deleted.
    if (KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT) != _DEFAULT_KEYCHAIN:
        return KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT
    kc = (doc or {}).get("keychain") or {}
    return (kc.get("service") or KEYCHAIN_SERVICE,
            kc.get("account") or KEYCHAIN_ACCOUNT)


def _new_keychain_account():
    """A per-install account name for a NEW vault.

    ⚠️ RANDOM, NOT DERIVED FROM THE PATH. A path hash would change when the
    install is moved or renamed, orphaning a key that is still perfectly good;
    a random name is written into the vault file and travels with it.
    """
    import secrets
    return f"{KEYCHAIN_ACCOUNT}-{secrets.token_hex(6)}"


def _keychain_set(hex_key):
    # ⚠️ Refuses when VAULT_PATH has been redirected — see `_keychain_is_safe`.
    if not _keychain_is_safe():
        return False
    service, account = _keychain_ids()
    subprocess.run(["security", "add-generic-password",
                    "-s", service, "-a", account,
                    "-w", hex_key, "-U"],
                   check=True, capture_output=True)


def _keychain_get():
    service, account = _keychain_ids()
    p = subprocess.run(["security", "find-generic-password",
                        "-s", service, "-a", account, "-w"],
                       capture_output=True)
    return p.stdout.decode().strip() if p.returncode == 0 else None


def _keychain_delete():
    """⚠️ GUARDED, like `_keychain_set` — and it matters more here, because
    this is the DESTRUCTIVE one. A cleanup that means the TEST account can
    resolve to the REAL one once ids come from the vault document and the
    caller has already restored VAULT_PATH. The vault FILE would still wrap the
    key, so nothing would be lost, but Mati would no longer open itself.
    """
    if not _keychain_is_safe():
        return False
    service, account = _keychain_ids()
    subprocess.run(["security", "delete-generic-password",
                    "-s", service, "-a", account],
                   capture_output=True)


def touch_id(reason="unlock Mati's memories", timeout=60):
    """Prompt for Touch ID. True if the user authenticated.

    Loaded dynamically so Mati needs no PyObjC framework wrapper package —
    only the core `objc` bridge, which macOS already has.
    """
    try:
        import objc
    except ImportError:
        return False
    try:
        ns = {}
        objc.loadBundle(
            "LocalAuthentication", ns,
            bundle_path="/System/Library/Frameworks/LocalAuthentication.framework")
        objc.registerMetaDataForSelector(
            b"LAContext", b"evaluatePolicy:localizedReason:reply:",
            {"arguments": {4: {"callable": {
                "retval": {"type": b"v"},
                "arguments": {0: {"type": b"^v"}, 1: {"type": b"Z"},
                              2: {"type": b"@"}}}}}})
        ctx = ns["LAContext"].alloc().init()
        done, out = threading.Event(), {}

        def reply(ok, err):
            out["ok"] = bool(ok)
            done.set()

        ctx.evaluatePolicy_localizedReason_reply_(1, reason, reply)
        return bool(out.get("ok")) if done.wait(timeout) else False
    except Exception:
        return False


def biometrics_available():
    try:
        import objc
        ns = {}
        objc.loadBundle(
            "LocalAuthentication", ns,
            bundle_path="/System/Library/Frameworks/LocalAuthentication.framework")
        ctx = ns["LAContext"].alloc().init()
        r = ctx.canEvaluatePolicy_error_(1, None)
        return bool(r[0]) if isinstance(r, (tuple, list)) else bool(r)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# The vault file — the recovery wrap, and a canary
# ─────────────────────────────────────────────────────────────────────────────

# ⚠️ A FORMAT CONSTANT — the plaintext sealed into every vault document at
# setup. Renamed with the product, it made every correct key "not open this
# vault". Write one, accept every historical spelling.
CANARY = "ollin-vault-canary"          # DO NOT RENAME
KNOWN_CANARIES = ("ollin-vault-canary", "olin-vault-canary",
                  "mati-vault-canary")


def _opens_canary(key, doc):
    """Does `key` open this document's canary? Never raises."""
    try:
        return decrypt(key, doc["canary"]) in KNOWN_CANARIES
    except Exception:
        return False


def _read_vault():
    if not os.path.exists(VAULT_PATH):
        return None
    with open(VAULT_PATH) as f:
        return json.load(f)


def _write_vault(doc):
    tmp = VAULT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, VAULT_PATH)


def _wrap_with_passphrase(data_key, passphrase, salt):
    kek = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt,
                              PBKDF2_ITERS, dklen=32)
    return base64.b64encode(
        _openssl(["-aes-256-cbc"], data_key, kek.hex())).decode()


def _unwrap_with_passphrase(wrapped, passphrase, salt):
    kek = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt,
                              PBKDF2_ITERS, dklen=32)
    return _openssl(["-aes-256-cbc", "-d"],
                    base64.b64decode(wrapped), kek.hex())


def make_passphrase():
    return "-".join(secrets.choice(_WORDS) for _ in range(5)) + \
        f"-{secrets.randbelow(90) + 10}"


def is_set_up():
    return _read_vault() is not None


# ─────────────────────────────────────────────────────────────────────────────
# The unlocked key, held only in memory
# ─────────────────────────────────────────────────────────────────────────────

_key = None


def unlock(passphrase=None, reason="unlock Mati's memories", quiet=False):
    """Get the data key into memory. Touch ID first, passphrase as the fallback."""
    global _key
    if _key is not None:
        return _key
    doc = _read_vault()
    if doc is None:
        raise VaultError("no vault — run: python3 -m Mati.vault --setup")

    if passphrase:
        key = _unwrap_with_passphrase(doc["wrapped"], passphrase,
                                      base64.b64decode(doc["salt"]))
    elif "keychain" not in doc:
        # ⚠️ A PASSPHRASE-ONLY VAULT, AND SAYING SO BEATS A TOUCH ID PROMPT
        # THAT CANNOT WORK. This vault was deliberately created with no
        # keychain copy (see `setup(passphrase_only=True)`), so there is
        # nothing for biometrics to unwrap. Prompting anyway and failing would
        # read as a broken Touch ID rather than as the choice it was.
        raise Locked("this vault is passphrase-only — there is no key in the "
                     "keychain by design. Unlock with:\n"
                     "         python3 -m Mati.vault --unlock")
    else:
        # ⚠️ THE FINGERPRINT IS ASKED FOR ONLY WHERE SETUP PROMISED IT. A
        # vault made on a Mac with no Touch ID records `touch_id: False`, and
        # prompting there could only ever fail — leaving Mati locked on every
        # start with a key sitting in the keychain it was allowed to read.
        # A vault with no record predates the choice and keeps the prompt.
        if doc["keychain"].get("touch_id", True) and not touch_id(reason):
            raise Locked("Touch ID declined or unavailable. Unlock with the "
                         "recovery passphrase instead.")
        hex_key = _keychain_get()
        if not hex_key:
            raise Locked("the keychain no longer holds Mati's key. Use the "
                         "recovery passphrase: python3 -m Mati.vault --unlock")
        key = bytes.fromhex(hex_key)

    if not _opens_canary(key, doc):
        raise VaultError("that key does not open this vault")
    _key = key
    if not quiet:
        print("  [vault] unlocked", file=sys.stderr)
        # A passphrase unlock on a machine whose keychain has no copy is the
        # signature of an install that has just been carried here. Say so
        # once, rather than letting the user conclude Touch ID is broken.
        # ⚠️ NOT FOR A PASSPHRASE VAULT. Having no keychain copy is what that
        # mode IS; suggesting `--adopt` there would talk someone into undoing
        # the choice they made at setup.
        if passphrase and "keychain" in doc and not adopted():
            print("  [vault] this Mac's keychain does not hold the key — "
                  "Mati cannot open itself here until you run:\n"
                  "         python3 -m Mati.vault --adopt", file=sys.stderr)
    return _key


def unlock_method():
    """How THIS vault opens: 'touch_id', 'keychain', 'passphrase', or None.

    None means no vault. Read from the vault file, never from the machine: a
    Touch ID vault carried to a Mac without a sensor is still a Touch ID
    vault, and saying otherwise would hide why it will not open.
    """
    try:
        doc = _read_vault()
    except Exception:
        doc = None
    if doc is None:
        return None
    kc = doc.get("keychain")
    if not kc:
        return "passphrase"
    return "touch_id" if kc.get("touch_id", True) else "keychain"


def vault_id():
    """A short, non-secret name for this vault, or None.

    Derived from the salt, which is public. It exists so that something about
    to hand a passphrase to a running Mati can check that Mati is serving
    THIS vault and not another install on the same port.
    """
    try:
        doc = _read_vault()
    except Exception:
        doc = None
    if not doc or not doc.get("salt"):
        return None
    return hashlib.sha256(doc["salt"].encode()).hexdigest()[:12]


def adopted():
    """Does this machine's keychain hold a key that OPENS THIS VAULT?

    ⚠️ VALIDITY, NOT PRESENCE. A keychain can hold a key that cannot open the
    vault; checking only that one is there would make `--adopt` print "this
    Mac's keychain already holds the key" and exit 0 without doing anything —
    a repair command reporting success while leaving the fault in place.

    A wrong key is not the same as a key. Verifying costs one decrypt of the
    canary that `unlock` performs anyway.
    """
    hex_key = _keychain_get()
    if not hex_key:
        return False
    doc = _read_vault()
    if doc is None:
        return False
    try:
        return _opens_canary(bytes.fromhex(hex_key), doc)
    except Exception:
        return False        # unreadable, wrong length, wrong key — all "no"


# ⚠️ A FORMAT CONSTANT: it tags a file the person keeps OFF this machine,
# possibly for years. A rename would make every kit already written refuse to
# restore. Write one, accept every historical spelling.
KIT_HEADER = "ollin-recovery-kit/1"    # DO NOT RENAME
KNOWN_KIT_HEADERS = ("ollin-recovery-kit/1", "olin-recovery-kit/1",
                     "mati-recovery-kit/1")


def recovery_kit(path):
    """Write the key to a file you keep ELSEWHERE. Returns the path.

    ⚠️⚠️ WHY THIS EXISTS: the keychain copy and the passphrase wrap are
    independent in principle and can be lost together in practice — a wiped
    keychain and a passphrase that was never written down. The vault file
    survives that; the KEY does not, and without it the memories are gone.

    ⚠️ A THIRD WRAP WOULD NOT HELP. A wrap under a third thing the person has
    to remember fails the same way the passphrase does.

    ⭐ SO THIS STORES THE KEY ITSELF, not a wrap of it. That is the whole
    point and the whole cost: **this file IS the vault.** Anyone holding it
    reads every memory without a passphrase and without Touch ID. It is worth
    having only if it lives somewhere the machine is not — a drive in a
    drawer, a password manager, paper.

    ⚠️ IT REFUSES TO WRITE BESIDE THE VAULT. A key in `.cache/` next to the
    thing it opens is not a backup, it is the encryption cancelling itself.
    """
    import json as _json
    import stat as _stat

    k = key()                       # ⚠️ requires an unlocked vault: authorised
    dest = os.path.abspath(os.path.expanduser(path))
    # ⚠️ BOTH THE LIVE VAULT AND THE DEFAULT. Guarding only
    # `_DEFAULT_VAULT_PATH` would protect the shipped `.cache` and happily write
    # a key beside a REDIRECTED vault — which is every test, and any second
    # install.
    guards = {os.path.abspath(os.path.dirname(VAULT_PATH)),
              os.path.abspath(os.path.dirname(_DEFAULT_VAULT_PATH))}
    if any(dest.startswith(g + os.sep) or os.path.dirname(dest) == g
           for g in guards):
        guard = os.path.dirname(os.path.abspath(VAULT_PATH))
        raise VaultError(
            f"refusing to write the key into {guard} — it sits beside the "
            f"vault it opens. Choose a path on a drive you remove, or in a "
            f"password manager.")

    doc = _read_vault() or {}
    body = {
        "kind": KIT_HEADER,
        "data_key_hex": k.hex(),
        "canary": doc.get("canary"),
        "note": ("This file IS Mati's vault. Anyone holding it can read every "
                 "memory. Keep it off the machine Mati runs on. To use it: "
                 "python3 -m Mati.vault --restore-kit <this file>"),
    }
    # ⚠️ 0600 BEFORE THE BYTES, not after. Creating world-readable and
    # tightening afterwards leaves a window, and this is key material.
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        _json.dump(body, f, indent=2)
    os.chmod(dest, _stat.S_IRUSR | _stat.S_IWUSR)
    return dest


def restore_kit(path):
    """Load the key from a recovery kit and teach it to this keychain.

    ⚠️ VERIFIED AGAINST THE CANARY BEFORE IT IS TRUSTED. A kit from a
    different vault would otherwise be adopted silently and every read would
    return noise.
    """
    import json as _json
    global _key

    src = os.path.abspath(os.path.expanduser(path))
    with open(src, encoding="utf-8") as f:
        body = _json.load(f)
    if body.get("kind") not in KNOWN_KIT_HEADERS:
        raise VaultError(f"{src} is not an Mati recovery kit")
    k = bytes.fromhex(body["data_key_hex"])

    doc = _read_vault()
    if not doc:
        raise VaultError("there is no vault here for this kit to open")
    # ⚠️ A WRONG KEY RAISES RATHER THAN RETURNING THE WRONG TEXT, so comparing
    # the result was not enough — the real failure arrived as a decryption
    # error and escaped as something the caller could not interpret. Both
    # outcomes mean the same thing and both say so.
    if not _opens_canary(k, doc):
        raise VaultError(
            "this kit does not open THIS vault — it is probably from another "
            "install. Nothing was changed.")
    _key = k
    return True


def adopt():
    """Write the open key into this machine's keychain. For a transfer.

    Mati is meant to be carried — the whole install is a few megabytes and
    moves on a drive. But **only the passphrase wrap travels**: it lives in
    `mati_vault.json`, while the keychain copy stays behind on the old Mac.

    So on a destination machine the passphrase works and Mati cannot open
    itself, for ever — the person retypes the passphrase at every launch and
    reasonably concludes the feature is broken. This is the one step that
    finishes a transfer.

    Deliberately NOT automatic on a passphrase unlock. Writing the key into a
    new machine's keychain is a durable grant of access to that machine, and
    should be something the person chose rather than something that happened
    because they typed a passphrase once — on a borrowed computer, or to read
    something back off a drive, that default would be wrong.
    """
    if _key is None:
        raise Locked("unlock first — there is no key to adopt")
    # ⚠️ VERIFY THE WRITE. A keychain write can be refused or silently dropped,
    # and reporting success over a keychain that has not changed would tell the
    # person Mati will open itself here when it will not.
    _keychain_set(_key.hex())
    if not adopted():
        raise VaultError(
            "the key was written to the keychain but does not open the vault "
            "— nothing has changed. Check `security find-generic-password -s "
            f"{KEYCHAIN_SERVICE} -a {KEYCHAIN_ACCOUNT}`")
    return True


def lock():
    """Drop the key — and the plaintext it was used to produce.

    ⚠️ `bridge` caches decrypted bodies so lexical search does not respawn
    openssl once per note. Dropping the key while that cache still held
    plaintext would make `lock()` a lie.
    """
    global _key
    _key = None
    try:
        from . import bridge
        bridge.forget_bodies()
    except Exception:
        pass                      # never let locking fail on a cache


def key():
    if _key is None:
        raise Locked("Mati's memories are locked")
    return _key


def unlocked():
    return _key is not None


def seal(text):
    """Encrypt a body for storage — no-op when no vault is set up."""
    if not is_set_up():
        return text
    return encrypt(key(), text)


def open_body(stored):
    """Decrypt a stored body. Passes plaintext through unchanged."""
    if not is_encrypted(stored):
        return stored
    return decrypt(key(), stored)


# ─────────────────────────────────────────────────────────────────────────────
# Setup and migration
# ─────────────────────────────────────────────────────────────────────────────

def _keychain_is_safe():
    """Is this process pointed at the real vault, or a redirected one?

    ⚠️ A REDIRECTED VAULT MUST NOT REACH THE REAL KEYCHAIN ACCOUNT. A script
    that points `VAULT_PATH` at a temp file and calls `setup()` would write its
    new data key to the REAL account if the account name followed nothing but
    a module constant — and the real vault would then fail its canary check,
    silently, until some unrelated command tried to unlock it. Nothing would be
    lost (the vault FILE still holds the passphrase-wrapped key, and
    `--unlock` plus `--adopt` restores it), but nothing would say so either.

    This makes that discipline the module's rather than the caller's.
    """
    return os.path.abspath(VAULT_PATH) == os.path.abspath(_DEFAULT_VAULT_PATH)


def setup(force=False, passphrase_only=False, presence_check=True):
    """Create the key, wrap it, and encrypt whatever is already stored.

    The two arguments choose one of the three modes in this module's header:

        passphrase_only=True                 'passphrase'
        presence_check=True  (the default)   'touch_id'
        presence_check=False                 'keychain'

    ⚠️ `passphrase_only` WRITES NOTHING TO THE KEYCHAIN, and that is the whole
    of what it offers: the key is nowhere on the machine, so nothing running
    there can read it while Mati is stopped. The price is that someone types
    the passphrase every time Mati starts, and `setup.py` makes them confirm
    they have written it down before continuing.

    ⚠️ `presence_check=False` IS FOR A MAC WITH NO TOUCH ID. A fingerprint
    prompt there can only fail, and Mati would start locked every time with
    its key one line away. The mode is recorded in the vault file, so `unlock`
    knows not to ask.
    """
    global _key
    if is_set_up() and not force:
        raise VaultError("a vault already exists — --force replaces it, which "
                         "makes every existing memory unreadable")
    data_key = secrets.token_bytes(32)
    passphrase = make_passphrase()
    salt = secrets.token_bytes(16)
    _key = data_key
    doc = {
        "version": 1,
        "salt": base64.b64encode(salt).decode(),
        "iters": PBKDF2_ITERS,
        "wrapped": _wrap_with_passphrase(data_key, passphrase, salt),
        "canary": encrypt(data_key, CANARY),
    }
    if not passphrase_only:
        # ⚠️ ITS OWN ACCOUNT, so a second install cannot overwrite a first
        # one's key. Written into the vault BEFORE `_keychain_set` below, or
        # the write would land on the account this vault no longer claims.
        doc["keychain"] = {"service": KEYCHAIN_SERVICE,
                           "account": _new_keychain_account(),
                           "touch_id": bool(presence_check)}
    _write_vault(doc)
    if not passphrase_only:
        _keychain_set(data_key.hex())
    return passphrase


# ─────────────────────────────────────────────────────────────────────────────
# WHICH COLUMNS HOLD SOMETHING A PERSON WROTE OR READ
#
# ⚠️ PROTECTION IS A PROPERTY OF THE STORE, NOT A HABIT OF ONE MODULE. The vault
# is built around `notes`, but other tables in the SAME FILE hold just as much
# — the book text waiting to be studied, what Mati means to say to the person,
# its own questions, the passages it declined to learn. Every table added later
# brings its own writer, and a writer that does not know about the vault does
# not call it.
#
# So the two sets are DECLARED, and the tests fail if a text column appears in
# neither. A new table cannot opt out by omission; it has to be classified, in
# the open, by someone who thought about it.
SEALED_COLUMNS = (
    # ── classified columns ──────────────────────────────
    # ⚠️ EVERY ONE OF THESE WAS CHECKED BY WRITING A SENTINEL THROUGH THE REAL
    # API AND GREPPING THE RAW FILE, not by reading the code for a `seal()`
    # call. That matters: a first pass cleared `watchlist.item` as sealed
    # because `_key()` lowercases, the sentinel was uppercase, and the byte
    # search missed it. The column is plaintext. A declaration built on a
    # mis-measurement is worse than no declaration.
    ("day_log", "text"),              # the transcript — what was actually said
    ("spending", "merchant"),         # where the money went
    ("spending", "item"),             # what was bought
    ("spending", "note"),
    ("watchlist", "note"),            # sealed
    ("price_seen", "merchant"),       # sealed — where they shop
    ("radio_presets", "name"),        # sealed — what they listen to
    ("radio_presets", "url"),         # sealed
    # ⚠️ A DRAFT IS THE PERSON'S OWN WRITING — anything they supplied is
    # identifying, and prose they wrote is the clearest case.
    ("drafts", "body"),
    # ⚠️ THE BRIEF IS THE PERSON'S OWN WORDS ABOUT THEIR OWN WORK. "3000
    # words on the divorce letter" names a subject as plainly as any episode
    # does, and `why` is their reason for abandoning something.
    ("targets", "what"),
    ("targets", "basis"),
    ("targets", "why"),
    # ⚠️ AN ANNOTATION QUOTES THE PERSON'S PROSE in its anchor and is
    # Mati's own observation in its note — both sealed, for the two
    # separate reasons the rule already covers.
    ("draft_notes", "anchor"), ("draft_notes", "note"),
    ("notes", "body"), ("notes", "verbatim"),
    ("to_say", "body"), ("to_say", "topic"),
    ("study_queue", "body"),
    ("inquiry_topics", "topic"), ("inquiry_topics", "next_q"),
    ("refusals", "body"),
    # Columns with no rows yet are classified too: a scan of a live store
    # cannot see them, and the classification guard can.
    ("to_say", "detail"),
    ("predictions", "claim"), ("predictions", "basis"),
    ("predictions", "correction"),
    ("library", "note"),
    # The person's own words, split into the turns they were typed in. Every
    # bit as sensitive as `notes.body` and sealed for exactly the same reason —
    # a new column holding user text in the clear is the finding that
    # was already recorded once in a security sweep.
    ("notes", "parts"),
    # Mati's own condensation of its own verbatim. Derived from the
    # person's words, so it is exactly as sensitive as what it condenses.
    ("notes", "digest"),
    # A feed item's playable file. Part of what the person holds.
    ("notes", "media"),
    # A meeting title and place are as personal as a note body.
    ("agenda", "title"), ("agenda", "location"),
    # What someone listens to is as personal as a note body.
    ("played", "name"), ("played", "detail"),
    # ⚠️ THE PERSON'S OWN WORDS. `rapport.question` stores settled exchanges
    # learned from the record — 'hello', 'i am <name>' — verbatim.
    ("rapport", "question"),
    # Empty in the live store today, and designed to hold words lifted from
    # memories — so it is sealed BEFORE it has content rather than after.
    ("segue_offers", "terms"),
)

# Deliberately in the clear, each for a reason that has to survive review.
CLEAR_COLUMNS = {
    # ── classified columns ──────────────────────────────
    # ⚠️⚠️ THREE OF THESE ARE A REAL EXCEPTION TO THE RULE, NOT AN APPLICATION
    # OF IT. The rule: anything that contains information provided by the user
    # is identifying data, and is marked and secured. `budgets.name`,
    # `watchlist.item` and `price_seen.item` ARE supplied by the person —
    # "grocery", "coffee", "book" — and they are plaintext anyway, because
    # each is a lookup key:
    #
    #   budgets.name      UNIQUE INDEX, and `spending.budget` joins to it
    #   watchlist.item    `ON CONFLICT(item)` upsert; `WHERE item=?` in
    #                     `unwatch` and `history`
    #   price_seen.item   INDEX (item, at); `watching()` passes its item
    #                     straight into `cheapest()`, so the two columns are
    #                     ONE key space and must match each other
    #
    # `vault.seal` is nondeterministic, so sealing any of them makes every
    # write a new row and every lookup a miss. The honest fix is a keyed
    # deterministic index plus a sealed display column — a schema migration,
    # not a declaration.
    #
    # ⭐ DECIDED: ship as-is, knowingly. Someone who can already read the raw
    # database learns the envelope names and the shopping list — and nothing
    # else. Amounts, merchants, notes and every transcript are sealed, and a
    # test proves that on each run rather than asserting it here. What would
    # reopen it is a column that is BOTH a lookup key and genuinely sensitive.
    ("budgets", "name"): "⚠️ user-supplied, plain: UNIQUE index + the join key "
                         "spending.budget matches on — see the note above",
    ("watchlist", "item"): "⚠️ user-supplied, plain: the ON CONFLICT upsert key "
                           "and the WHERE key for unwatch/history",
    ("price_seen", "item"): "⚠️ user-supplied, plain: one key space with "
                            "watchlist.item, indexed (item, at)",

    ("budgets", "period"): "month/week/once — a fixed vocabulary",
    ("budgets", "created"): "timestamp",
    ("budgets", "withdrawn"): "timestamp",
    ("spending", "budget"): "which envelope; the indexed join to budgets.name",
    ("spending", "at"): "timestamp — kept clear so totals work without the key",
    ("spending", "corrected"): "timestamp",
    ("spending", "recorded"): "timestamp",
    ("watchlist", "added"): "timestamp",
    ("watchlist", "dropped"): "timestamp — withdrawal is a date, not content",
    ("price_seen", "source"): "the name of a FEED (\"flipp\"), not of a person",
    ("price_seen", "at"): "timestamp",
    ("price_seen", "mentioned"): "timestamp",
    ("radio_presets", "codec"): "MP3/AAC — a fact about the stream",
    ("radio_presets", "country"): "from the station directory, not from them",
    ("radio_presets", "saved_at"): "timestamp",
    ("day_log", "at"): "timestamp; the (day, at) index orders the transcript",
    ("day_log", "day"): "the day key — WHERE day=? on every read",
    ("day_log", "who"): "them/mati — two values, verified in the live table",
    ("day_log", "note"): "a machine label; the live table holds only 'replay'",
    ("day_log", "ref"): "an internal row reference, never prose",
    # ⚠️ A DRAFT'S TITLE IS PLAIN BECAUSE IT IS THE LOOKUP KEY — the unique
    # index and every WHERE clause use it, and sealed values do not compare.
    # Same reason `notes.source` is plain. The BODY, which is the person's
    # actual writing, is sealed.
    # ⚠️ A DECLARATION IS A CLAIM. A column declared sealed that the code does
    # not actually seal would turn the tests green over plaintext, which is
    # precisely the error this file exists to prevent.
    ("drafts", "title"): "the lookup key; the unique index compares it",
    ("targets", "draft"): "the draft's title — the same lookup key as "
                          "drafts.title, and compared against it",
    ("targets", "measure"): "words/passages/sessions/done — an enum",
    ("targets", "due"): "a date the person gave; no content",
    ("targets", "source"): "user/inferred — provenance, not content",
    ("targets", "status"): "open/met/superseded/withdrawn — a state",
    ("targets", "created"): "timestamp",
    ("targets", "touched"): "timestamp",
    ("draft_notes", "title"): "which draft it belongs to; the lookup key",
    ("draft_notes", "created"): "timestamp",
    ("drafts", "status"): "open/done — a state, not content",
    ("drafts", "created"): "timestamp",
    ("drafts", "touched"): "timestamp; the pointer orders by it",
    ("notes", "source"): "provenance kind; recall filters on it without a key",
    ("notes", "wall_created"): "timestamp; the whole schedule reads it",
    ("notes", "speaker"): "a first name, and attribution must survive a lock",
    ("notes", "cut_at"): "how many words the reply reached — a count, not text",
    ("notes", "media_kind"): "audio/video — so 'do you have any video on X' is answerable without the key",
    ("played", "at"): "timestamp; 'that station from Saturday night' is answered by it",
    ("played", "kind"): "station/track — a category, not content",
    ("agenda", "starts"): "timestamp; the agenda is ordered and grouped by it",
    ("agenda", "ends"): "timestamp",
    ("agenda", "rrule"): "a recurrence rule — structure, not content",
    ("agenda", "calendar"): "which calendar it came from, like study_queue.origin",
    ("agenda", "source"): "apple/ics/mati — provenance, same as notes.source",
    ("agenda", "imported"): "timestamp",
    ("agenda", "withdrawn"): "timestamp when it left the source; NULL means live",
    ("agenda", "ext_id"): "the source's own opaque identifier, not text of ours",
    ("notes", "personal_only"): "a 0/1 flag marking MATI'S OWN introspection — likes, self-discoveries, interests, goals. Set on `mine` rows and nothing else. ⭐ `make_ship_field.assert_impersonal` REFUSES to call a store shippable if any of these rows are in it, which is what makes the flag mean something: 'a build is from scratch' is a property of how the build RUNS, not of what it emits. ⚠️ Deliberately NOT filtered on transfer: a copy to the same person's new Mac SHOULD carry them, they are what makes it the same Mati. ⚠️ The flag is equivalent to `source LIKE 'mine%'`; it is kept because an export to ANOTHER person needs it as the precedent for what must not travel, and a test trips if the two ever diverge. Plain, so the flag is readable without the key",
    ("rapport", "kind"): "greeting/introduction — a category, not content",
    ("rapport", "qhash"): "a hash, deliberately — it is the lookup key for the sealed question beside it, and equality search cannot work on salted ciphertext",
    ("rapport", "for_day"): "timestamp",
    ("rapport", "made_at"): "timestamp",
    ("segue_offers", "at"): "timestamp",
    ("segue_offers", "tier"): "which shelf the offer came from, a category",
    ("segue_offers", "arm"): "study arm label, not content",
    ("activity", "at"): "timestamp; the history page groups by it",
    ("activity", "event"): "wake/sleep/dream — a category, not content",
    # ⚠️ The weakest of these. It carries document names ("113 passages of
    # asked about computer science"), which is the same exposure already
    # accepted for `study_queue.origin` — and accepted for the same reason:
    # progress has to be groupable without the key. It must stay a LABEL. The
    # moment anything writes a sentence of the person's here, it belongs above.
    ("activity", "detail"): "a short label — document name or a count",
    # A published author and a publication year, from the printed book. The
    # title beside them is already clear because it is the PRIMARY KEY, and an
    # author is no more private than the title it belongs to.
    ("library", "author"): "a published author's name, like the title above",
    ("library", "published"): "a publication year",
    ("to_say", "kind"): "reminder/finding/question — a category, not content",
    ("study_queue", "origin"): "document name; progress GROUPs BY it",
    ("study_queue", "sha"): "content hash, already one-way",
    ("study_queue", "status"): "queued/learned — a state machine",
    ("study_queue", "added"): "timestamp",
    ("study_queue", "studied"): "timestamp",
    ("study_queue", "detail"): "a similarity number, not text",
    ("inquiry_topics", "added"): "timestamp",
    ("inquiry_topics", "last_asked"): "timestamp",
    ("inquiry_topics", "status"): "open/closed",
    ("refusals", "rule"): "which invariant fired",
    ("refusals", "why"): "the reason, e.g. '1 words'",
    ("refusals", "source"): "provenance, same as notes.source",
    ("refusals", "at"): "timestamp",
    ("recall_hits", "how"): "field/name — which channel found it",
    ("gate_log", "at"): "timestamp",
    ("gate_log", "text_hash"): "a hash, deliberately — see gatelog.py",
    ("to_say", "session"): "which conversation, not what was said",
    ("consent", "key"): "setting name",
    ("consent", "value"): "on/off, a quiet-hours range",
    ("predictions", "kind"): "timing/recurrence",
    ("predictions", "state"): "open/hit/missed/withdrawn",
    ("predictions", "detail"): "how it resolved, a number",
    # ⚠️ `library.title` CANNOT be sealed: it is the PRIMARY KEY, `add()`
    # depends on ON CONFLICT(title), and each seal produces different
    # ciphertext, so sealing it would turn every re-add into a duplicate.
    # The cost is real and is stated rather than hidden — the titles of what
    # Mati has read are legible in the file.
    ("library", "title"): "PRIMARY KEY; ON CONFLICT needs a stable value",
    ("library", "origin"): "source string, same as notes.source",
    ("library", "status"): "read/reading/suggested/wanted",
    ("library", "suggested_by"): "a first name, same as notes.speaker",
    ("library", "added_at"): "timestamp",
    ("library", "finished_at"): "timestamp",
    ("sqlite_sequence", "name"): "sqlite internal",
    ("sqlite_sequence", "seq"): "sqlite internal",
}


def seal_soft(text):
    """Seal if the key is here; otherwise store as-is for `migrate_tables`.

    ⚠️ `seal()` RAISES when locked, and these writers are not all on paths that
    hold the key — `say_later` fires from the sleep loop, `study` enqueues in
    the background. Hard-failing would turn a locked vault into a crash that
    loses a reminder, which is a worse outcome than the exposure it prevents.

    This is the pattern the module already uses for `notes`: plaintext rows
    accumulate before setup, `is_encrypted` tolerates a mixed store, and
    `migrate` catches up. `migrate_tables` is idempotent and runs at startup
    and after an unlock, so the window is one locked spell, not forever.

    ⚠️ It is a WINDOW, not an exemption. Anything written while locked is in
    the clear until the next migration, and that is the honest cost of not
    dropping the person's data on the floor.
    """
    try:
        return seal(text)
    except Locked:
        return text


def migrate_tables(db_path=None, verbose=True):
    """Seal every SEALED_COLUMNS value that is still in the clear.

    Safe to run twice, like `migrate` — an already-sealed value is skipped, so
    this is idempotent and can be called on every startup without cost.
    """
    import sqlite3
    from . import bridge
    # ⚠️ REFUSE WHEN LOCKED. `seal()` passes plaintext through unchanged when
    # there is no key, so a migration run locked would rewrite every value as
    # itself and report a count — a no-op that looks like a success and leaves
    # the store exactly as exposed as before.
    if not unlocked():
        raise Locked("migrate_tables needs the key — sealing without one "
                     "would rewrite plaintext as plaintext and report success")
    db_path = db_path or bridge.NOTES_DB_PATH
    conn = sqlite3.connect(db_path)
    total = 0
    try:
        have = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for table, col in SEALED_COLUMNS:
            if table not in have:
                continue
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                continue
            key = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")
                   if r[5]] or ["rowid"]
            key = key[0]
            rows = conn.execute(
                f"SELECT {key}, {col} FROM {table} "
                f"WHERE {col} IS NOT NULL AND {col} <> ''").fetchall()
            done = 0
            for rid, val in rows:
                if is_encrypted(val):
                    continue
                conn.execute(f"UPDATE {table} SET {col}=? WHERE {key}=?",
                             (seal(val), rid))
                done += 1
            if done and verbose:
                print(f"  [vault] sealed {done} of {len(rows)} in "
                      f"{table}.{col}", file=sys.stderr)
            total += done
        conn.commit()
    finally:
        conn.close()
    return total


def migrate(db_path=None, verbose=True):
    """Encrypt every plaintext body in place. Safe to run twice."""
    import sqlite3
    from . import bridge
    db_path = db_path or bridge.NOTES_DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        bridge._ensure_schema(conn)
        rows = conn.execute("SELECT note_id, body FROM notes").fetchall()
        done = 0
        for note_id, body in rows:
            if is_encrypted(body):
                continue
            conn.execute("UPDATE notes SET body=? WHERE note_id=?",
                         (seal(body), note_id))
            done += 1
        conn.commit()
    finally:
        conn.close()
    if verbose:
        print(f"  [vault] encrypted {done} of {len(rows)} bodies "
              f"({len(rows)-done} already were)", file=sys.stderr)
    return done


def harden_permissions():
    """Owner-only on the files that hold memory. Free, and the floor."""
    from . import bridge
    changed = []
    for path in (bridge.NOTES_DB_PATH, bridge.NOTES_THETA0_PATH,
                 bridge.address_theta0_path(), VAULT_PATH):
        if path and os.path.exists(path):
            mode = os.stat(path).st_mode & 0o777
            if mode != 0o600:
                os.chmod(path, 0o600)
                changed.append((os.path.basename(path), oct(mode)))
    return changed


def unlock_running(passphrase=None, timeout=90):
    """Ask the Mati running on this machine to open its vault. Its answer.

    Unlocking in a process of your own opens nothing for the one that is
    actually answering, so this is how a locked Mati gets its memories back
    without being restarted. With no passphrase it retries the vault's own
    way — the keychain, after a fingerprint where the vault calls for one.

    ⚠️ ONLY TO A MATI SERVING THIS VAULT. The port is per machine, not per
    install: a second copy could be the one listening, and a passphrase is not
    something to hand to whatever answers. `vault_id` is compared first, and
    nothing is sent on a mismatch.

    Returns `running` and `same_vault`, then whatever `/unlock` replied.
    """
    import urllib.error
    import urllib.request
    base = f"http://{C.SERVER_HOST}:{C.SERVER_PORT}"
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as r:
            health = json.loads(r.read().decode())
    except Exception:
        return {"running": False}
    if not vault_id() or health.get("vault_id") != vault_id():
        return {"running": True, "same_vault": False}
    req = urllib.request.Request(
        base + "/unlock",
        data=json.dumps({"passphrase": passphrase} if passphrase else {}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            reply = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            reply = json.loads(e.read().decode())
        except Exception:
            reply = {"unlocked": False, "error": f"HTTP {e.code}"}
    except Exception as e:
        reply = {"unlocked": False, "error": str(e)}
    return {"running": True, "same_vault": True, **reply}


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Mati's memory vault")
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="with --setup: replace the key, losing every memory")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--unlock", action="store_true",
                    help="open the vault, and the Mati that is running")
    ap.add_argument("--export", metavar="PATH",
                    help="write decrypted memories to a file you choose")
    ap.add_argument("--recovery-kit", metavar="PATH",
                    help="write the key to a file you keep ELSEWHERE — the "
                         "only thing that survives losing both the keychain "
                         "and the passphrase")
    ap.add_argument("--restore-kit", metavar="PATH",
                    help="open this vault with a recovery kit and put the key "
                         "back in the keychain")
    ap.add_argument("--adopt", action="store_true",
                    help="after carrying Mati here: teach THIS Mac's keychain "
                         "the key, so Touch ID works again")
    args = ap.parse_args()

    if args.recovery_kit:
        if not is_set_up():
            print("  there is no vault here to make a kit for"); return 1
        if not unlocked():
            try:
                unlock(reason="write a recovery kit")
            except Exception as e:
                print(f"  the vault must be open to copy its key: {e}"); return 1
        try:
            dest = recovery_kit(args.recovery_kit)
        except Exception as e:
            print(f"  {e}"); return 1
        print(f"\n  written: {dest}  (mode 0600)")
        print("\n  ⚠️ THIS FILE IS THE VAULT. Anyone holding it reads every")
        print("  memory — no passphrase, no Touch ID. It is worth having only")
        print("  if it lives somewhere this machine is not: a drive you")
        print("  remove, a password manager, paper in a drawer.")
        print("\n  Move it off this machine now, then delete the copy here.")
        print(f"\n  To use it later:  python3 -m Mati.vault --restore-kit <path>")
        return 0

    if args.restore_kit:
        try:
            restore_kit(args.restore_kit)
        except Exception as e:
            print(f"  {e}"); return 1
        print("  the kit opened this vault")
        try:
            adopt()
        except Exception as e:
            print(f"  opened, but the keychain was not updated: {e}")
            print("  the memories are readable in this process; Touch ID is not fixed")
            return 1
        print("  adopted — Touch ID works again on this Mac")
        return 0

    if args.adopt:
        import getpass
        if adopted():
            print("  this Mac's keychain already holds the key")
            return 0
        print("  Mati was carried here, so Touch ID has nothing to unlock "
              "with yet.\n  Enter the recovery passphrase to teach this Mac "
              "the key.\n")
        try:
            unlock(passphrase=getpass.getpass("  passphrase: ").strip(),
                   quiet=True)
        except Exception as e:
            # ⚠️ SAY WHICH STEP FAILED. A wrong passphrase and a failed
            # keychain write are different problems with different fixes, and
            # this printed neither.
            print(f"\n  that passphrase did not open the vault: {e}")
            return 1
        try:
            adopt()
        except Exception as e:
            print(f"\n  unlocked, but the keychain was not updated: {e}")
            return 1
        print("  adopted — Touch ID will work on this Mac from now on")
        return 0

    if args.status or not any((args.setup, args.unlock, args.export, args.adopt)):
        doc = _read_vault()
        mode = unlock_method()
        print(f"  vault        : {'set up' if doc else 'not set up'}")
        if mode:
            print(f"  opens with   : " + {
                "touch_id": "the keychain, after a fingerprint",
                "keychain": "the keychain, by itself when Mati starts",
                "passphrase": "the passphrase, typed each time Mati starts",
            }[mode])
        print(f"  biometrics   : {'available' if biometrics_available() else 'no'}")
        if mode == "passphrase":
            print("  keychain key : none, by design")
        else:
            held = bool(_keychain_get())
            print(f"  keychain key : {'present' if held else 'absent'}"
                  + ("" if held or not doc else
                     "  <- carried here? run --adopt"))
        if doc:
            import sqlite3
            from . import bridge
            conn = sqlite3.connect(bridge.NOTES_DB_PATH)
            bodies = [r[0] for r in conn.execute("SELECT body FROM notes")]
            conn.close()
            enc = sum(1 for b in bodies if is_encrypted(b))
            print(f"  memories     : {enc}/{len(bodies)} encrypted")
        for name, was in harden_permissions():
            print(f"  tightened    : {name} {was} -> 0o600")
        return 0

    if args.setup:
        # ⚠️ ASK THE MACHINE. A Touch ID vault on a Mac with no sensor can
        # never open itself; `python3 -m Mati.setup` offers the full choice.
        touch = biometrics_available()
        phrase = setup(force=args.force, presence_check=touch)
        n = migrate()
        harden_permissions()
        print("\n" + "=" * 62)
        print("  Mati's memories are now encrypted.")
        print("\n  RECOVERY PASSPHRASE — write this down now:\n")
        print(f"      {phrase}\n")
        print("  " + ("Touch ID unlocks Mati day to day." if touch else
                     "Mati opens its own key from your keychain when it starts.")
              + " This passphrase is the")
        print("  ONLY way back if the keychain is lost — a wiped Mac or a new")
        print("  machine. It is not stored anywhere you can read it again,")
        print("  and nothing can recover it.")
        print("=" * 62)
        print(f"\n  {n} existing memories encrypted.")
        return 0

    if args.unlock:
        import getpass
        mode = unlock_method()
        if mode is None:
            print("  there is no vault here to unlock"); return 1
        ask = ("recovery passphrase" if mode == "passphrase" else
               "recovery passphrase (or just return, to have Mati open "
               "its own key)")
        phrase = getpass.getpass(f"  {ask}: ").strip()
        if phrase:
            # ⚠️ CHECKED HERE FIRST, so a mistyped passphrase is never sent
            # anywhere and never counts against the running Mati's limit.
            try:
                unlock(passphrase=phrase, quiet=True)
            except Exception as e:
                print(f"  that passphrase did not open the vault: {e}")
                return 1
        elif mode == "passphrase":
            print("  this vault opens only with its passphrase"); return 1
        out = unlock_running(phrase or None)
        if not out.get("running"):
            print("  " + ("the passphrase is right, but " if phrase else "")
                  + f"no Mati is running on port {C.SERVER_PORT} — start it, "
                  "then run this again")
            return 0 if phrase else 1
        if not out.get("same_vault"):
            print(f"  a Mati is running on port {C.SERVER_PORT}, but it is a "
                  "different install — nothing was sent to it")
            return 1
        if out.get("unlocked"):
            print("  Mati is unlocked" + (" — it already was" if out.get("already")
                                          else " and can read its memories now"))
            return 0
        print(f"  Mati stayed locked: {out.get('error') or 'no reason given'}")
        return 1

    if args.export:
        import sqlite3
        from . import bridge
        unlock(reason="export Mati's memories in the clear")
        conn = sqlite3.connect(bridge.NOTES_DB_PATH)
        rows = conn.execute(
            "SELECT note_id, source, body FROM notes ORDER BY note_id").fetchall()
        conn.close()
        with open(args.export, "w") as f:
            for note_id, source, body in rows:
                f.write(f"--- #{note_id} [{source}] ---\n{open_body(body)}\n\n")
        os.chmod(args.export, 0o600)
        print(f"  {len(rows)} memories written to {args.export} IN THE CLEAR "
              f"(mode 0600)")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
