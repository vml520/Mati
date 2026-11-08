"""
Mati's memories, encrypted at rest, unlocked by your fingerprint.

    python3 -m Mati.vault --setup       # once: make a key, print recovery phrase
    python3 -m Mati.vault --status
    python3 -m Mati.vault --lock        # forget the key held in memory
    python3 -m Mati.vault --export out.txt

WHY NOT JUST PROMPT BEFORE OPENING MATI. Because that protects nothing. The
episodes live in `.cache/mati_notes.db`, a plain SQLite file that was
world-readable (`-rw-r--r--`) until this existed — anyone with a shell reads
every one of them without Mati running. And Mati ships open source, so a
prompt in Python is a prompt anyone can delete.

So Touch ID is not a gate here. It releases a KEY. The enforcement lives in the
T2 Secure Enclave and the login keychain, not in this file: without the key the
bodies are ciphertext, and editing this source does not change that.

WHAT IS AND IS NOT ENCRYPTED. Note bodies. Not the addresses.

    note_id | source | feat (plain) | body (ciphertext)

`recall` ranks on `feat`, the trajectory-feature address, and only needs the
body text for the handful of notes it actually surfaces. So the field keeps
working untouched, retrieval never needs the key, and the content is
unreadable at rest. Encrypting the whole database instead would have meant
SQLCipher (a dependency Mati does not have) or decrypting to a temp file —
which leaves plaintext on disk while running and after a crash.

THREAT MODEL, stated plainly. This protects against another account on the
machine, a stolen or imaged disk, and Time Machine backups. It does NOT protect
against you: you can always unlock, and anything you can unlock, malware
running as you can unlock too. That is the correct model for a personal AI, and
overstating it would be worse than not having it.

RECOVERY. The key is wrapped twice — once into the keychain behind Touch ID,
once with a passphrase printed at setup. A wiped Mac, a new machine, or a reset
Touch ID loses the keychain copy; the passphrase is the only way back. Nothing
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
# ⚠️⚠️ NOT RENAMED WITH THE PRODUCT, AND THAT IS DELIBERATE. When Mati
# became Mati on  every other "mati" went; this string stayed.
#
# The data key lives in the macOS keychain UNDER THIS NAME. Changing it does
# not move the key — it orphans it, and the only person who would notice is
# someone who already had a vault. This project has destroyed that key TWICE
# and recovered it by luck both times (`recovery_kit()` exists because of it).
#
# ⭐ A KEYCHAIN IS NOT A PATH: there is no rename, only a write under a new
# name and an entry nobody looks for under the old one. Nothing a user ever
# sees contains this string, so consistency here buys nothing and risks
# everything. `test_vault` pins it.
KEYCHAIN_SERVICE = "ollin-vault"
KEYCHAIN_ACCOUNT = "data-key"
# ⚠️ What the constants were at import. Anything else means a caller
# redirected them on purpose, and that intent outranks the vault's record.
_DEFAULT_KEYCHAIN = (KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT)
# ⚠️ C.CACHE_DIR, NOT C.ROOT/.cache. This was hardcoded to the repo's own
# .cache and so ignored MATI_HOME, unlike every other path in the package.
# Three things broke, found once the vault was actually enabled:
#
#   - a sandbox or clean room saw the REAL vault as set up and tried to seal
#     with a key its own store had no descriptor for — `test_coldstart` died
#     on `Locked` while writing its first episode;
#   - `transfer.manifest()` carries `mati_vault.json`, so backup/restore and
#     the thumb-drive install wrote the descriptor into CACHE_DIR while this
#     module went on reading somewhere else entirely — the portable install
#     would have looked fine and been unopenable;
#   - `python3 -m Mati.sandbox` could not produce an unencrypted scratch copy
#     at all.
#
# On an install with no MATI_HOME set, CACHE_DIR *is* ROOT/.cache, so this
# changes nothing about where an existing vault lives.
VAULT_PATH = os.path.join(C.CACHE_DIR, "mati_vault.json")
# The path as shipped. `_keychain_is_safe` compares against this so a
# redirected VAULT_PATH cannot reach the real keychain account.
_DEFAULT_VAULT_PATH = VAULT_PATH

PREFIX = "mati1:"                     # marks an encrypted body
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


def encrypt(data_key, text):
    """Encrypt-then-MAC. Returns a self-contained string safe to store."""
    enc = _subkey(data_key, b"mati-body-enc").hex()
    mac = _subkey(data_key, b"mati-body-mac")
    blob = _openssl(["-aes-256-cbc"], text.encode("utf-8"), enc)
    tag = hmac.new(mac, blob, hashlib.sha256).digest()
    return PREFIX + base64.b64encode(blob).decode() + ":" + \
        base64.b64encode(tag).decode()


def decrypt(data_key, stored):
    """Inverse of `encrypt`. Verifies the MAC BEFORE decrypting."""
    if not is_encrypted(stored):
        return stored                          # not yet migrated
    try:
        blob_b64, tag_b64 = stored[len(PREFIX):].split(":", 1)
        blob = base64.b64decode(blob_b64)
        tag = base64.b64decode(tag_b64)
    except (ValueError, base64.binascii.Error) as e:
        raise VaultError(f"corrupt ciphertext: {e}") from e
    mac = _subkey(data_key, b"mati-body-mac")
    if not hmac.compare_digest(hmac.new(mac, blob, hashlib.sha256).digest(), tag):
        raise VaultError("ciphertext failed its integrity check — wrong key, "
                         "or the stored note was tampered with")
    enc = _subkey(data_key, b"mati-body-enc").hex()
    return _openssl(["-aes-256-cbc", "-d"], blob, enc).decode("utf-8")


def is_encrypted(text):
    return isinstance(text, str) and text.startswith(PREFIX)


# ─────────────────────────────────────────────────────────────────────────────
# Where the key lives
# ─────────────────────────────────────────────────────────────────────────────

def _keychain_ids():
    """(service, account) for THIS vault. Reads the vault's own record.

    ⚠️⚠️ TWO INSTALLS USED TO SHARE ONE KEYCHAIN ENTRY, AND THE SECOND
    OVERWROTE THE FIRST. `KEYCHAIN_SERVICE`/`KEYCHAIN_ACCOUNT` are module
    constants, identical in every copy of Mati — so a person with a working
    install who set up a second copy silently replaced the first one's data
    key under `mati-vault/data-key`. Touch ID then produced a key that could
    not open the first vault, which is exactly the incident this module's
    `_keychain_is_safe` note describes, arriving by a route that guard cannot
    see: it compares VAULT_PATH against THIS install's own default, and for a
    second install those match.

    ⭐ THE VAULT FILE ALREADY RECORDED ITS OWN service/account — the format
    had the answer and the accessors ignored it, reading the constants
    instead. This reads the record.

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
    """⚠️ GUARDED, AND IT WAS NOT. `_keychain_set` refused when VAULT_PATH was
    redirected and `_keychain_delete` — the DESTRUCTIVE one — had no guard at
    all. That is exactly backwards, and it cost a real data key: a cleanup
    line reading `_keychain_delete()  # the TEST account, never the real one`
    resolved to the REAL account once ids came from the vault doc, because by
    then the caller had already restored VAULT_PATH. Nothing was lost — the
    vault FILE still wrapped the key — but Touch ID stopped working and the
    comment asserting safety was no longer true.
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

CANARY = "mati-vault-canary"


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
        if not touch_id(reason):
            raise Locked("Touch ID declined or unavailable. Unlock with the "
                         "recovery passphrase instead.")
        hex_key = _keychain_get()
        if not hex_key:
            raise Locked("the keychain no longer holds Mati's key. Use the "
                         "recovery passphrase: python3 -m Mati.vault --unlock")
        key = bytes.fromhex(hex_key)

    if decrypt(key, doc["canary"]) != CANARY:
        raise VaultError("that key does not open this vault")
    _key = key
    if not quiet:
        print("  [vault] unlocked", file=sys.stderr)
        # A passphrase unlock on a machine whose keychain has no copy is the
        # signature of an install that has just been carried here. Say so
        # once, rather than letting the user conclude Touch ID is broken.
        if passphrase and not adopted():
            print("  [vault] this Mac's keychain does not hold the key — "
                  "Touch ID will not work here until you run:\n"
                  "         python3 -m Mati.vault --adopt", file=sys.stderr)
    return _key


def adopted():
    """Does this machine's keychain hold a key that OPENS THIS VAULT?

    ⚠️ THIS CHECKED PRESENCE, NOT VALIDITY, AND THAT MADE `--adopt` REFUSE THE
    ONE CASE IT EXISTS FOR. Found : the keychain held a key that
    could not open the vault, `adopted()` saw a key and returned True, and
    `--adopt` printed "this Mac's keychain already holds the key" and exited 0
    without doing anything. The repair command reported success while leaving
    the fault in place, and Touch ID went on failing.

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
        return decrypt(bytes.fromhex(hex_key), doc["canary"]) == CANARY
    except Exception:
        return False        # unreadable, wrong length, wrong key — all "no"


KIT_HEADER = "mati-recovery-kit/1"


def recovery_kit(path):
    """Write the key to a file you keep ELSEWHERE. Returns the path.

    ⚠️⚠️ WHY THIS EXISTS: THE KEY HAS BEEN DESTROYED TWICE AND RECOVERED BY
    LUCK BOTH TIMES — once because a process happened to still be running with
    it in memory, once because the passphrase had been printed to a screen and
    that output happened to persist.
    The vault file was intact on both occasions; the memories were never at
    risk. The KEY was.

    ⚠️ THE EXISTING "TWO WRAPS" ARE NOT TWO CHANCES. The keychain wrap and the
    passphrase wrap are independent in principle and were lost together in
    practice, because the passphrase was never written down. A third wrap
    under a third thing the person has to remember would fail the same way.

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
    # ⚠️ BOTH THE LIVE VAULT AND THE DEFAULT. The first version guarded only
    # `_DEFAULT_VAULT_PATH`, so it protected the shipped `.cache` and happily
    # wrote a key beside a REDIRECTED vault — which is every test, and any
    # second install. Using a default where the live value was meant is the
    # same mistake that destroyed the key twice.
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
    return noise — the failure `adopted()` used to permit, arriving from a
    file instead of a keychain.
    """
    import json as _json
    global _key

    src = os.path.abspath(os.path.expanduser(path))
    with open(src, encoding="utf-8") as f:
        body = _json.load(f)
    if body.get("kind") != KIT_HEADER:
        raise VaultError(f"{src} is not an Mati recovery kit")
    k = bytes.fromhex(body["data_key_hex"])

    doc = _read_vault()
    if not doc:
        raise VaultError("there is no vault here for this kit to open")
    # ⚠️ A WRONG KEY RAISES RATHER THAN RETURNING THE WRONG TEXT, so comparing
    # the result was not enough — the real failure arrived as a decryption
    # error and escaped as something the caller could not interpret. Both
    # outcomes mean the same thing and both say so.
    try:
        matched = decrypt(k, doc["canary"]) == CANARY
    except Exception:
        matched = False
    if not matched:
        raise VaultError(
            "this kit does not open THIS vault — it is probably from another "
            "install. Nothing was changed.")
    _key = k
    return True


def adopt():
    """Re-wrap the open key into this machine's keychain. For a transfer.

    Mati is meant to be carried — the whole install is a few megabytes and
    moves on a drive. But the key is wrapped twice and **only one wrap
    travels**: the passphrase wrap lives in `mati_vault.json`, while the
    Touch ID wrap lives in this Mac's Secure Enclave and stays behind.

    So on a destination machine the passphrase works and Touch ID does not,
    for ever — the user retypes a twelve-word passphrase at every launch and
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
    # ⚠️ VERIFY THE WRITE. This returned True unconditionally, so a keychain
    # write that was refused or silently dropped still reported success — and
    # the caller printed "Touch ID will work on this Mac from now on" over a
    # keychain that had not changed. Third time this week a function has
    # reported an outcome it never checked.
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

    ⚠️ A DIAGNOSTIC ONCE OVERWROTE THE REAL KEYCHAIN KEY WITH A THROWAWAY. A
    diagnostic script redirected `VAULT_PATH` to a temp file and called
    `setup()` — which writes the new data key to the REAL keychain account,
    because the account name is a module constant and does not follow
    VAULT_PATH. Touch ID then produced a key that could not open the real
    vault, and `unlock()` failed its canary check.

    Nothing was lost: the vault FILE still holds the passphrase-wrapped key,
    and `--unlock` plus `--adopt` restores it. But the failure was silent at
    the moment it happened and only surfaced minutes later, on an unrelated
    command.

    `test_vault` had this right all along — it swaps KEYCHAIN_ACCOUNT to
    "data-key-TEST" and deletes only that. This makes the discipline the
    module's rather than the caller's.
    """
    return os.path.abspath(VAULT_PATH) == os.path.abspath(_DEFAULT_VAULT_PATH)


def setup(force=False, passphrase_only=False):
    """Create the key, wrap it, and encrypt whatever is already stored.

    ⚠️ `passphrase_only=True` IS FOR MACS WITHOUT TOUCH ID, AND IT IS NOT A
    DEGRADED MODE — it is the only honest one there. Until  setup
    SKIPPED ENCRYPTION ENTIRELY on such a machine and printed a note about it,
    so those installs kept every memory in the clear while the README's
    privacy wording read exactly the same as everywhere else.

    ⚠️ AND IT MUST NOT WRITE THE KEY TO THE KEYCHAIN. The keychain entry is
    what Touch ID gates; with no biometrics it is readable by anything running
    in the login session, so storing the key there would make the encryption
    theatre — a vault whose key sits beside it. Passphrase-only means the
    passphrase is genuinely the only way in, which is why `setup.py` makes the
    person confirm they have written it down before continuing.
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
                           "account": _new_keychain_account()}
    _write_vault(doc)
    if not passphrase_only:
        _keychain_set(data_key.hex())
    return passphrase


# ─────────────────────────────────────────────────────────────────────────────
# WHICH COLUMNS HOLD SOMETHING A PERSON WROTE OR READ
#
# ⚠️ THE VAULT PROTECTED ONE TABLE AND EVERYTHING ELSE WENT ROUND IT. Found
# found by chasing a `test_vault` failure that had been dismissed as noise
# three times. `notes.body` and `notes.verbatim` were sealed; four sibling
# tables in the SAME FILE were not:
#
#     study_queue.body      2,776 rows   551,988 characters of book text
#     to_say.body/topic     1,153 rows   what Mati means to SAY to the person
#     inquiry_topics        9 rows       its own questions
#     refusals.body         16 rows      passages it declined to learn
#
# The cause was structural: the vault was built around `notes`, and every table
# added afterwards brought its own writer and simply did not call it.
# Protection was a habit of one module rather than a property of the store, and
# a habit cannot be relied on by a table that does not know it exists.
#
# So the two sets are DECLARED, and `test_vault` fails if a text column appears
# in neither. A new table cannot opt out by omission any more; it has to be
# classified, in the open, by someone who thought about it.
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
    ("price_seen", "merchant"),       # sealed  — where they shop
    ("radio_presets", "name"),        # sealed  — what they listen to
    ("radio_presets", "url"),         # sealed
    # ⚠️ A DRAFT IS THE PERSON'S OWN WRITING — the rule says anything
    # they supplied is identifying, and prose they wrote is the
    # clearest case. Declared when the table was created, rather than
    # becoming the seventh unclassified one.
    ("drafts", "body"),
    # ⚠️ THE BRIEF IS THE PERSON'S OWN WORDS ABOUT THEIR OWN WORK. "3000
    # words on the divorce letter" names a subject as plainly as any episode
    # does, and `why` is their reason for abandoning something. `test_vault`
    # caught these unclassified the day the module landed, which is the only
    # reason they are here rather than in plaintext on disk.
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
    # Found by the classification guard, which sees columns that have no rows
    # yet and so were invisible to a scan of the live store.
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
    # ⚠️ FOUND BY THE CLASSIFICATION GUARD, AND IT IS THE PERSON'S
    # OWN WORDS. `rapport.question` stores settled exchanges learned from the
    # record — 'hello', 'i am <name>' — verbatim and, until now, in the clear.
    # Exactly the finding a security sweep recorded for the
    # sibling tables; `rapport` was built after that sweep and inherited none of
    # it. The guard caught it, nobody did.
    ("rapport", "question"),
    # Empty in the live store today, and designed to hold words lifted from
    # memories — so it is sealed BEFORE it has content rather than after.
    ("segue_offers", "terms"),
)

# Deliberately in the clear, each for a reason that has to survive review.
CLEAR_COLUMNS = {
    # ── classified columns ──────────────────────────────
    # ⚠️⚠️ THREE OF THESE ARE A REAL EXCEPTION TO THE RULE, NOT AN APPLICATION
    # OF IT. The rule is the user's: "anything that contains information provided
    # by the user is identifying data, and should be marked and secured."
    # `budgets.name`, `watchlist.item` and `price_seen.item` ARE supplied by
    # the person — "grocery", "coffee", "book" are in the live table — and
    # they are plaintext anyway, because each is a lookup key:
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
    # ⭐ DECIDED by the user: ship as-is, knowingly. Someone who can
    # already read the raw database learns the envelope names and the
    # shopping list — and nothing else. Amounts, merchants, notes and every
    # transcript are sealed, and a test proves that on each run rather than
    # asserting it here. Settled, not an open question. What would reopen it
    # is a column that
    # is BOTH a lookup key and genuinely sensitive.
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
    # ⚠️ THIS WAS DECLARED SEALED BEFORE IT WAS, which is precisely the
    # error this file exists to prevent: a declaration is a claim, and a
    # claim that outruns the code turns `test_vault` green over plaintext.
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
    ("notes", "personal_only"): "a 0/1 flag marking MATI'S OWN introspection — likes, self-discoveries, interests, goals. Set on `mine` rows and nothing else. ⭐ IT HAS A READER AS OF : `make_ship_field.assert_impersonal` REFUSES to call a store shippable if any of these rows are in it, which is what makes the flag mean something. That was previously true only by accident: 'a build is from scratch' is a property of how the build RUNS, not of what it emits. ⚠️ Deliberately NOT filtered on transfer: a copy to the same person's new Mac SHOULD carry them, they are what makes it the same Mati. ⚠️ Today the flag is exactly equivalent to `source = 'mine'` (measured: 40 of 6669, zero asymmetry); it is kept because a planned export to ANOTHER person needs it as the precedent for what must not travel, and a test trips if the two ever diverge. Plain, so the flag is readable without the key",
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
    the background. Making them hard-fail turned a locked vault into a crash
    that lost a reminder, which is a worse outcome than the exposure it was
    meant to prevent.

    This is the pattern the module already uses for `notes`: plaintext rows
    accumulate before setup, `is_encrypted` tolerates a mixed store, and
    `migrate` catches up. `migrate_tables` is idempotent and runs at startup,
    so the window is one process lifetime, not forever.

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


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Mati's memory vault")
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="with --setup: replace the key, losing every memory")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--unlock", action="store_true",
                    help="unlock with the recovery passphrase")
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
        print(f"  vault        : {'set up' if doc else 'not set up'}")
        print(f"  biometrics   : {'available' if biometrics_available() else 'no'}")
        print(f"  keychain key : {'present' if _keychain_get() else 'absent'}"
              + ("" if _keychain_get() or not doc else
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
        phrase = setup(force=args.force)
        n = migrate()
        harden_permissions()
        print("\n" + "=" * 62)
        print("  Mati's memories are now encrypted.")
        print("\n  RECOVERY PASSPHRASE — write this down now:\n")
        print(f"      {phrase}\n")
        print("  Touch ID unlocks Mati day to day. This passphrase is the")
        print("  ONLY way back if the keychain is lost — a wiped Mac, a new")
        print("  machine, a reset Touch ID. It is not stored anywhere you can")
        print("  read it again, and nothing can recover it.")
        print("=" * 62)
        print(f"\n  {n} existing memories encrypted.")
        return 0

    if args.unlock:
        import getpass
        unlock(passphrase=getpass.getpass("recovery passphrase: "))
        print("  unlocked (this process only)")
        return 0

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
