"""Mati's field engine — the phase-field substrate its memory is built on.

Mati uses the field as MEMORY: addressing, coherence and imprinting. It never
reads the trained readout `W`, which is what using the field as a language
model would need.

⚠️ THIS PACKAGE IS AUTHORITATIVE AND SELF-CONTAINED. It began as a shared
study codebase and was split off so the field could change without anything
else moving under it. Imports are package-relative — no `sys.path`
manipulation — and nothing outside Mati imports it.
"""
