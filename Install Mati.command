#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Install Mati
#
#  Double-click this file to install Mati, or run it in Terminal. It sets up
#  everything Mati needs on this Mac, and running it again later updates Mati
#  without touching your memories.
#
#  What it does, in order:
#    1. checks that this Mac can run Mati
#    2. installs Apple's Command Line Tools if they are missing (they include
#       Python, which Mati is written in)
#    3. installs the few Python packages Mati uses
#    4. installs Ollama, the free program that runs the language model
#    5. downloads the language model (about 2 GB)
#    6. puts Mati in a folder called "Mati" in your home folder
#    7. runs Mati's first-time setup, which asks a few questions
#    8. adds a "Mati" app to the Applications folder in your home folder
#
#  It never asks for an administrator password and never uses `sudo`. Nothing
#  about you is sent anywhere: the downloads are Apple's tools, Python
#  packages, Ollama, the model, and Mati itself.
# ─────────────────────────────────────────────────────────────────────────────

set -u

MATI_ZIP="${MATI_ZIP_URL:-https://github.com/vml520/Mati/archive/refs/heads/main.zip}"
OLLAMA_ZIP="https://ollama.com/download/Ollama-darwin.zip"
OLLAMA_API="http://127.0.0.1:11434"
PY=/usr/bin/python3
PACKAGES="numpy tiktoken pyobjc-core pyobjc-framework-Cocoa pyobjc-framework-NaturalLanguage"

# Where things go. These can be changed for testing; the defaults are what
# the README describes.
DEST="${MATI_DEST:-$HOME/Mati}"
APPS="${MATI_APPS:-$HOME/Applications}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)"
TMP="$(mktemp -d -t mati-install)"
trap 'rm -rf "$TMP"' EXIT

bold=$'\033[1m'; dim=$'\033[2m'; off=$'\033[0m'
STEP=0
step()  { STEP=$((STEP + 1)); printf '\n%s%s. %s%s\n' "$bold" "$STEP" "$1" "$off"; }
say()   { printf '   %s\n' "$@"; }
ok()    { printf '   ✓ %s\n' "$1"; }
pause() { if [ -t 0 ]; then printf '\n%s' "${1:-Press Return to close this window.}"; read -r _; fi; }
fail()  {
    printf '\n   %sMati could not be installed.%s\n' "$bold" "$off"
    for line in "$@"; do printf '   %s\n' "$line"; done
    printf '\n   Nothing you had before has been changed. When the problem is fixed,\n'
    printf '   run this installer again — it carries on from where it stopped.\n'
    pause
    exit 1
}

ollama_up() { curl -s -m 3 -o /dev/null -w '%{http_code}' "$OLLAMA_API/api/version" 2>/dev/null | grep -q 200; }

clear 2>/dev/null
printf '%sMati installer%s\n' "$bold" "$off"
say "This sets up Mati on this Mac. It takes 10–30 minutes, most of it" \
    "downloading. You can leave it running; it will ask when it needs you."

# ── 1. the Mac ──────────────────────────────────────────────────────────────
step "Checking this Mac"
[ "$(uname -s)" = "Darwin" ] || fail "Mati runs on macOS only."
ram_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
free_gb=$(df -g "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')
say "macOS $(sw_vers -productVersion 2>/dev/null), $(uname -m), ${ram_gb} GB of memory, ${free_gb:-?} GB free"
if [ "${free_gb:-99}" -lt 5 ]; then
    fail "Mati needs about 5 GB of free disk space for the model and its tools." \
         "This Mac has ${free_gb} GB free."
fi
MODEL="llama3.2:3b"
if [ "$ram_gb" -lt 8 ]; then
    MODEL="llama3.2:1b"
    say "⚠️  Less than 8 GB of memory: Mati will use the smaller 1B model, and"
    say "    replies will still be slow. 8 GB is the real minimum."
fi
ok "this Mac can run Mati"

# ── 2. Python ───────────────────────────────────────────────────────────────
step "Python (Apple's Command Line Tools)"
if ! xcode-select -p >/dev/null 2>&1; then
    say "Mati needs Apple's free Command Line Tools, which include Python."
    say "A window is opening — click ${bold}Install${off}, then ${bold}Agree${off}. This takes"
    say "5–20 minutes. This installer waits and carries on by itself."
    xcode-select --install >/dev/null 2>&1
    waited=0
    until xcode-select -p >/dev/null 2>&1; do
        sleep 10; waited=$((waited + 10))
        if [ "$waited" -ge 3600 ]; then
            fail "The Command Line Tools did not finish installing within an hour." \
                 "Install them from the window that appeared (or run" \
                 "  xcode-select --install" \
                 "in Terminal), then run this installer again."
        fi
    done
fi
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null \
    || fail "Python 3.9 or newer is needed, and $PY is not it."
ok "Python $("$PY" -c 'import platform; print(platform.python_version())')"

# ── 3. Python packages ──────────────────────────────────────────────────────
step "Python packages Mati uses"
if "$PY" -c 'import numpy, tiktoken, objc, Foundation, NaturalLanguage' 2>/dev/null; then
    ok "already installed"
else
    say "Installing: $PACKAGES"
    "$PY" -m pip install --user --disable-pip-version-check --no-warn-script-location -q $PACKAGES \
        || fail "The Python packages would not install. Check the internet" \
                "connection and run this installer again."
    "$PY" -c 'import numpy, tiktoken, objc, Foundation, NaturalLanguage' 2>/dev/null \
        || fail "The Python packages installed but will not load."
    ok "installed"
fi

# ── 4. Ollama ───────────────────────────────────────────────────────────────
step "Ollama, which runs the language model"
find_ollama() {
    for d in /Applications "$HOME/Applications"; do
        [ -d "$d/Ollama.app" ] && { echo "$d/Ollama.app"; return 0; }
    done
    return 1
}
if ollama_up; then
    ok "Ollama is running"
else
    app="$(find_ollama)"
    if [ -z "$app" ] && command -v ollama >/dev/null 2>&1; then
        say "Starting Ollama…"
        nohup ollama serve >/dev/null 2>&1 </dev/null &
    else
        if [ -z "$app" ]; then
            say "Downloading Ollama (about 230 MB)…"
            curl -fL --progress-bar -o "$TMP/Ollama.zip" "$OLLAMA_ZIP" \
                || fail "Ollama would not download. Check the internet connection."
            ditto -x -k "$TMP/Ollama.zip" "$TMP/ollama" \
                || fail "The Ollama download could not be opened."
            target=/Applications
            [ -w "$target" ] || { target="$HOME/Applications"; mkdir -p "$target"; }
            mv "$TMP/ollama/Ollama.app" "$target/" \
                || fail "Ollama could not be moved into $target."
            app="$target/Ollama.app"
            ok "Ollama installed in $target"
        fi
        say "Starting Ollama. If it shows its own welcome window, you can close it."
        open "$app"
    fi
    waited=0
    until ollama_up; do
        sleep 3; waited=$((waited + 3))
        [ "$waited" -ge 120 ] && fail "Ollama did not start. Open Ollama from Applications," \
                                      "then run this installer again."
    done
    ok "Ollama is running"
fi

# ── 5. the language model ───────────────────────────────────────────────────
step "The language model ($MODEL)"
if curl -s -m 10 "$OLLAMA_API/api/tags" | grep -q "\"$MODEL\""; then
    ok "already downloaded"
else
    say "Downloading. This is the big one — about 2 GB for the 3B model."
    MODEL="$MODEL" OLLAMA_API="$OLLAMA_API" "$PY" - <<'PULL' || fail "The model did not download. Check the internet connection."
import json, os, sys, urllib.request
model, api = os.environ["MODEL"], os.environ["OLLAMA_API"]
req = urllib.request.Request(
    api + "/api/pull",
    data=json.dumps({"model": model, "name": model, "stream": True}).encode(),
    headers={"Content-Type": "application/json"})
last = ""
with urllib.request.urlopen(req, timeout=7200) as r:
    for raw in r:
        d = json.loads(raw or b"{}")
        if d.get("error"):
            print("\n   " + d["error"]); sys.exit(1)
        status, total, done = d.get("status", ""), d.get("total"), d.get("completed")
        if total and done:
            sys.stdout.write("\r   %-28s %5.1f%% of %.1f GB" % (
                status[:28], 100.0 * done / total, total / 1e9))
        elif status != last:
            sys.stdout.write("\n   " + status)
        last = status
        sys.stdout.flush()
print()
PULL
    ok "downloaded"
fi

# ── 6. Mati ─────────────────────────────────────────────────────────────────
step "Mati itself"
FIRST=1
[ -d "$DEST/Mati" ] && FIRST=0
if [ -f "$HERE/Mati/server.py" ]; then
    SRC="$HERE"                        # run from inside a downloaded Mati folder
else
    say "Downloading Mati…"
    curl -fsSL -o "$TMP/mati.zip" "$MATI_ZIP" || fail "Mati would not download."
    ditto -x -k "$TMP/mati.zip" "$TMP/src" || fail "The Mati download could not be opened."
    SRC="$(ls -d "$TMP"/src/*/ 2>/dev/null | head -1)"
    SRC="${SRC%/}"
    [ -f "$SRC/Mati/server.py" ] || fail "The Mati download did not contain Mati."
fi
if [ "$(cd "$SRC" && pwd)" = "$(mkdir -p "$DEST" && cd "$DEST" && pwd)" ]; then
    ok "running from $DEST itself — nothing to copy"
elif [ -d "$DEST/.git" ]; then
    say "$DEST is a git checkout, so it is left alone. Update it with git pull."
else
    # ⚠️ YOUR MEMORIES LIVE IN $DEST/.cache AND ARE NEVER OVERWRITTEN. An update
    # replaces Mati's code; from .cache it only adds files that are missing.
    # Stop the Mati that is running from THIS folder, so it can be updated —
    # and only that one: whatever else is listening on the port is not ours.
    if [ "$FIRST" = 0 ] && [ -x "$DEST/installer/start-mati.sh" ] \
            && [ "$("$DEST/installer/start-mati.sh" --status)" = "running" ]; then
        say "Mati is running — stopping it so it can be updated…"
        "$DEST/installer/start-mati.sh" --stop
    fi
    SRC="$SRC" DEST="$DEST" "$PY" - <<'COPY' || fail "Mati's files could not be copied into $DEST."
import os, shutil
src, dest = os.environ["SRC"], os.environ["DEST"]
skip = {".git", ".cache", "__pycache__", ".DS_Store"}
for name in os.listdir(src):
    if name in skip:
        continue
    a, b = os.path.join(src, name), os.path.join(dest, name)
    if os.path.isdir(a):
        shutil.copytree(a, b, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    else:
        shutil.copy2(a, b)
cache_src, cache_dest = os.path.join(src, ".cache"), os.path.join(dest, ".cache")
os.makedirs(cache_dest, exist_ok=True)
if os.path.isdir(cache_src):
    for name in os.listdir(cache_src):
        b = os.path.join(cache_dest, name)
        if not os.path.exists(b):
            shutil.copy2(os.path.join(cache_src, name), b)
COPY
    chmod +x "$DEST/Install Mati.command" "$DEST/installer/start-mati.sh" 2>/dev/null
    if [ "$FIRST" = 1 ]; then ok "installed in $DEST"; else ok "updated in $DEST — your memories are untouched"; fi
fi

if [ "$MODEL" != "llama3.2:3b" ]; then
    (cd "$DEST" && "$PY" -c "from Mati import config as C; C.save_settings(local_model='$MODEL')") \
        || say "⚠️  Could not record the smaller model; Mati may ask for llama3.2:3b."
fi

# ── 7. first-time setup ─────────────────────────────────────────────────────
step "First-time setup"
if [ "$FIRST" = 0 ]; then
    ok "already set up — skipped (to run it again: cd ~/Mati && python3 -m Mati.setup)"
elif [ -n "${MATI_SKIP_SETUP:-}" ]; then
    ok "skipped"
else
    say "Mati will now check this Mac, then ask a few questions:"
    say "  • whether to lock your memories (with Touch ID, if this Mac has it)"
    say "  • what to call you, and a little about you — all optional but the name"
    say ""
    say "${bold}If you lock your memories, it prints a RECOVERY PASSPHRASE.${off}"
    say "${bold}Write it down and keep it somewhere safe.${off} It is the only way back"
    say "into your memories if this Mac's keychain is ever lost."
    pause "Press Return to begin setup. "
    (cd "$DEST" && "$PY" -m Mati.setup) \
        || say "⚠️  Setup did not finish. You can run it any time:  cd ~/Mati && python3 -m Mati.setup"
fi

# ── 8. the Mati app ─────────────────────────────────────────────────────────
step "The Mati app"
mkdir -p "$APPS"
START="$DEST/installer/start-mati.sh"
q_start="${START//\\/\\\\}"; q_start="${q_start//\"/\\\"}"
cat > "$TMP/Mati.applescript" <<APPLESCRIPT
on run
	set s to quoted form of "$q_start"
	set matiStatus to do shell script s & " --status"
	if matiStatus is "busy" then
		display dialog "Another program is using Mati's port, so Mati cannot start. Quit that program, then open Mati again." buttons {"OK"} default button "OK" with icon caution with title "Mati"
		return
	end if
	if matiStatus is "running" then
		set r to display dialog "Mati is running." buttons {"Stop Mati", "Open Mati"} default button "Open Mati" with title "Mati"
		if button returned of r is "Stop Mati" then
			display notification "Stopping Mati — it saves first, which can take a couple of minutes." with title "Mati"
			do shell script s & " --stop"
			display notification "Mati has stopped." with title "Mati"
		else
			do shell script s & " --open"
		end if
		return
	end if
	display notification "Starting Mati — the window opens when it is ready." with title "Mati"
	try
		do shell script s & " --start"
	on error e
		display dialog "Mati could not start." & return & return & e buttons {"OK"} default button "OK" with icon caution with title "Mati"
	end try
end run
APPLESCRIPT
rm -rf "$APPS/Mati.app"
osacompile -o "$APPS/Mati.app" "$TMP/Mati.applescript" >/dev/null \
    || fail "The Mati app could not be created in $APPS."
icon="$DEST/app/icons/icon-1024.png"
if [ -f "$icon" ]; then
    set_icon() {
        local set="$TMP/Mati.iconset"; mkdir -p "$set"
        for s in 16 32 128 256 512; do
            sips -z $s $s "$icon" --out "$set/icon_${s}x${s}.png" >/dev/null 2>&1 || return 1
            sips -z $((s * 2)) $((s * 2)) "$icon" --out "$set/icon_${s}x${s}@2x.png" >/dev/null 2>&1 || return 1
        done
        iconutil -c icns "$set" -o "$APPS/Mati.app/Contents/Resources/applet.icns" 2>/dev/null
    }
    set_icon || true
    touch "$APPS/Mati.app"
fi
ok "Mati is in $APPS — drag it to the Dock to keep it handy"

printf '\n%sMati is installed.%s\n' "$bold" "$off"
say "Open ${bold}Mati${off} from the Applications folder in your home folder. The first"
say "start takes a minute or two; after that, Mati's window opens in your browser."
say "You can delete the folder you downloaded — Mati now lives in $DEST."
if [ -z "${MATI_NO_OPEN:-}" ]; then
    open "$APPS/Mati.app"
fi
pause
