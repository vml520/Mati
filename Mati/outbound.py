"""
Fetching an address that came from outside, without reaching this machine.

    from . import outbound
    with outbound.urlopen(req, timeout=10) as r: ...

⚠️ THE CHECK IS ON THE ADDRESS CONNECTED TO, NOT ON HOW THE URL IS SPELT.
`world.is_public_url` used to read the host as a string, and a string has too
many spellings: `2130706433`, `0x7f000001`, `017700000001`, `0`,
`[::ffff:127.0.0.1]` and `localhost.` all reach this machine, and a public
name can simply resolve to 127.0.0.1. Measured against the old check, 8 of 11
such spellings got through. So the name is resolved, EVERY address it returns
must be out in the world, and the socket connects to one of those checked
addresses — not to a second lookup that could answer differently.

⚠️ AND ON EVERY REDIRECT HOP. A redirect opens a new connection through the
same handlers, so `https://a.example/` answering `302 → http://127.0.0.1/` is
refused when that connection is made — before the request is sent. Checking
the final URL after the fetch, as before, only discarded an answer whose
request had already gone out.

⚠️ ONLY http AND https. The opener is built by hand rather than with
`build_opener`, which would add FTP and file handlers: a redirect to `ftp:`
would otherwise be followed, unguarded.

What is still out of reach here: media players (AVPlayer, Music, QuickTime)
fetch for themselves, so `check_url` is asked before a URL is handed to one —
and a redirect the player follows after that is the player's own.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request


class Refused(OSError):
    """The address is on this machine or its network, not out in the world."""


def _is_world(ip):
    """Is this an address out on the internet? `ip` is an ipaddress object."""
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_global) and not ip.is_multicast


def literal_ip(host):
    """The IP a host string spells, in any of its forms, or None for a name.

    ⚠️ `inet_aton` IS THE ONE THAT READS THE ODD FORMS — decimal, hex, octal,
    and shortened ones like `127.1` — exactly as the resolver will. A host
    that is only digits, dots and hex is an address, however it is written.
    """
    h = (host or "").strip().strip("[]").split("%")[0]
    if not h:
        return None
    try:
        return ipaddress.ip_address(h)
    except ValueError:
        pass
    try:
        return ipaddress.IPv4Address(socket.inet_aton(h))
    except OSError:
        return None


def host_refused(host):
    """Why `host` is plainly not out in the world, without a lookup — or None.

    Catches every spelling of a local address and the local names. A public
    NAME passes here and is checked again when it is resolved.
    """
    h = (host or "").strip().lower().strip("[]").rstrip(".")
    if not h:
        return "that has no host"
    ip = literal_ip(h)
    if ip is not None:
        return None if _is_world(ip) else (
            "that address is on this machine, not the world")
    if (h == "localhost" or h.endswith(".localhost") or h.endswith(".local")
            or h.endswith(".internal") or h.endswith(".lan")
            or h.endswith(".home.arpa") or "." not in h):
        return "that address is on this machine, not the world"
    return None


def resolve_public(host, port):
    """[(family, sockaddr)] for `host`, every one out in the world — or raise.

    ⚠️ ONE LOCAL ANSWER REFUSES THE NAME. A name that resolves to a public
    address and 127.0.0.1 together is not "mostly public".
    """
    why = host_refused(host)
    if why:
        raise Refused(why)
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    out = []
    for family, _type, _proto, _name, sockaddr in infos:
        if not _is_world(ipaddress.ip_address(sockaddr[0].split("%")[0])):
            raise Refused(f"{host} resolves to this machine or its network")
        out.append((family, sockaddr))
    if not out:
        raise Refused(f"{host} did not resolve")
    return out


def _connect_public(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
                    source_address=None):
    """`socket.create_connection`, to a checked address only."""
    host, port = address
    err = None
    for family, sockaddr in resolve_public(host, port):
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as e:
            err = e
            sock.close()
    raise err or Refused(f"could not connect to {host}")


class _HTTPConnection(http.client.HTTPConnection):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._create_connection = _connect_public


class _HTTPSConnection(http.client.HTTPSConnection):
    # ⚠️ TLS STILL VERIFIES THE NAME. Only the socket's address is chosen
    # here; `HTTPSConnection.connect` wraps it with `server_hostname`.
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._create_connection = _connect_public


class _HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_HTTPConnection, req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_HTTPSConnection, req, context=self._context,
                            check_hostname=self._check_hostname)


def _opener():
    o = urllib.request.OpenerDirector()
    for h in (_HTTPHandler(), _HTTPSHandler(),
              urllib.request.HTTPDefaultErrorHandler(),
              urllib.request.HTTPRedirectHandler(),
              urllib.request.HTTPErrorProcessor(),
              urllib.request.UnknownHandler()):
        o.add_handler(h)
    return o


_OPENER = _opener()


def urlopen(url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
    """`urllib.request.urlopen`, refusing any address not out in the world.

    A refusal arrives as `urllib.error.URLError` whose `reason` is `Refused`,
    the same shape as any other failure to connect.
    """
    target = url.full_url if isinstance(url, urllib.request.Request) else url
    scheme = urllib.parse.urlparse(target).scheme.lower()
    if scheme not in ("http", "https"):
        raise urllib.error.URLError(Refused("only http and https addresses"))
    return _OPENER.open(url, data=data, timeout=timeout)


def check_url(url):
    """(ok, why) for a URL about to be handed to something that fetches it
    itself — a media player. Resolves the name; refuses anything local."""
    u = urllib.parse.urlparse(url or "")
    if u.scheme.lower() not in ("http", "https") or not u.hostname:
        return False, "that has to be an http or https address"
    try:
        resolve_public(u.hostname, u.port or (443 if u.scheme == "https"
                                              else 80))
    except Refused as e:
        return False, str(e)
    except OSError as e:
        return False, f"that address could not be found ({type(e).__name__})"
    return True, "ok"
