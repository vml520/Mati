#!/bin/bash
# Start Mati, stop it, or open its window. The Mati app runs this.
#
#   start-mati.sh --start    start Ollama and Mati if needed, then open the window
#   start-mati.sh --open     open the window
#   start-mati.sh --stop     stop Mati (it saves first)
#   start-mati.sh --status   prints "running" or "stopped"
#
# ⚠️ EVERY BACKGROUND PROCESS IS FULLY DETACHED (stdin, stdout and stderr all
# redirected). The app runs this through AppleScript's `do shell script`, which
# waits until every inherited output is closed — a server left holding one
# would keep the app waiting for as long as Mati runs.

set -u
MATI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${MATI_PORT:-8756}"
URL="http://127.0.0.1:$PORT/"
OLLAMA_API="http://127.0.0.1:11434"
PY=/usr/bin/python3
LOG="${MATI_HOME:-$MATI_DIR/.cache}/mati.log"

running()   { curl -s -m 2 -o /dev/null -w '%{http_code}' "${URL}health" 2>/dev/null | grep -q 200; }

# ⚠️ THE PORT MAY BE SOMEONE ELSE'S. Another program — or another Mati, from a
# different folder — can be listening on it, and stopping or opening that would
# be acting on the wrong thing. The server is ours only if the process
# listening on the port is running from this folder.
listener_dir() {
    local pid
    pid="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1)"
    [ -n "$pid" ] || return 1
    lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1
}
ours() { [ "$(listener_dir)" = "$(cd "$MATI_DIR" && pwd -P)" ]; }
busy() { [ -n "$(listener_dir)" ] && ! ours; }
ollama_up() { curl -s -m 2 -o /dev/null -w '%{http_code}' "$OLLAMA_API/api/version" 2>/dev/null | grep -q 200; }

start_ollama() {
    ollama_up && return 0
    local d
    for d in /Applications "$HOME/Applications"; do
        if [ -d "$d/Ollama.app" ]; then open -g "$d/Ollama.app"; break; fi
    done
    if ! ollama_up && command -v ollama >/dev/null 2>&1; then
        nohup ollama serve >/dev/null 2>&1 </dev/null &
    fi
    local waited=0
    until ollama_up; do
        sleep 2; waited=$((waited + 2))
        if [ "$waited" -ge 90 ]; then
            echo "Ollama is not running. Open Ollama from the Applications folder, then open Mati again."
            return 1
        fi
    done
}

case "${1:---start}" in
    --status)
        if ours && running; then echo running
        elif busy; then echo busy
        else echo stopped; fi ;;
    --open)
        open "$URL" ;;
    --stop)
        ours || exit 0
        # Saves and folds the store into a copyable state first, which can take
        # a little while on a large store.
        curl -s -m 900 -X POST "${URL}stop" >/dev/null 2>&1
        waited=0
        while running && [ "$waited" -lt 120 ]; do sleep 2; waited=$((waited + 2)); done ;;
    --start)
        if busy; then
            echo "Another program is using Mati's port ($PORT). Quit it, then open Mati again."
            exit 1
        fi
        start_ollama || exit 1
        if ! ours; then
            mkdir -p "$(dirname "$LOG")"
            cd "$MATI_DIR" || exit 1
            nohup "$PY" -m Mati.server >>"$LOG" 2>&1 </dev/null &
            # A first start loads the field and warms the model: 10–30 seconds
            # on most Macs, up to a few minutes on the slowest. Touch ID, if the
            # memories are locked with it, asks during this wait.
            waited=0
            until ours && running; do
                sleep 2; waited=$((waited + 2))
                if [ "$waited" -ge 300 ]; then
                    echo "Mati did not start within five minutes. What it said is in $LOG"
                    exit 1
                fi
            done
        fi
        open "$URL" ;;
    *)
        echo "usage: start-mati.sh [--start | --open | --stop | --status]"; exit 2 ;;
esac
