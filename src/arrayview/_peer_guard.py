"""Refuse connections from other users on the same machine.

The server binds loopback, but on a shared host every user can reach
loopback ports, and nothing in a request says who sent it.  On Linux the
kernel does know: the client end of a loopback TCP connection is a socket
owned by the uid that opened it.  We look that socket up by its exact
4-tuple (sock_diag netlink, the mechanism ``ss`` uses; /proc/net/tcp as a
fallback) and refuse the request when it belongs to a different, non-root
user.

Every legitimate path ends in a process owned by the server's user: the
CLI, the VS Code opener, VS Code's Remote-SSH/tunnel port forwarders, an
``ssh -L``/``-R`` session, a local browser, a Jupyter proxy.  Another user's
VS Code forward (or their ArrayView, or anything else) does not.

Not enforced when the owner cannot be determined (non-Linux, a peer outside
this network namespace such as a container port mapping, a non-loopback
client): behaviour there is unchanged.  ``ARRAYVIEW_ALLOW_OTHER_USERS=1``
disables the check for deliberate shared setups.

``GET /ping`` stays open: other users' launchers and openers read the
server's uid and hostname from it to decide not to use this server.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import struct
import sys

_OPEN_PATHS = frozenset({"/ping"})
_DENY_MESSAGE = b"This ArrayView belongs to another user on this machine."

# sock_diag constants (linux/sock_diag.h, linux/inet_diag.h)
_NETLINK_SOCK_DIAG = 4
_SOCK_DIAG_BY_FAMILY = 20
_NLM_F_REQUEST = 1
_NLMSG_ERROR = 2
_NLMSG_DONE = 3
_ALL_TCP_STATES = 0xFFF
_NOCOOKIE = 0xFFFFFFFF


def _enabled() -> bool:
    return sys.platform.startswith("linux") and os.environ.get(
        "ARRAYVIEW_ALLOW_OTHER_USERS", ""
    ).strip().lower() not in {"1", "true", "yes", "on"}


def _addr_words(ip: ipaddress._BaseAddress) -> bytes:
    raw = ip.packed
    return raw + b"\0" * (16 - len(raw))


def _diag_uid(family: int, src, dst) -> int | None:
    """uid owning the TCP socket whose local end is *src* and remote *dst*."""
    src_ip = ipaddress.ip_address(src[0])
    dst_ip = ipaddress.ip_address(dst[0])
    req = struct.pack(
        "=BBBBI", family, socket.IPPROTO_TCP, 0, 0, _ALL_TCP_STATES
    ) + struct.pack(">HH", src[1], dst[1]) + _addr_words(src_ip) + _addr_words(
        dst_ip
    ) + struct.pack("=III", 0, _NOCOOKIE, _NOCOOKIE)
    header = struct.pack(
        "=IHHII", 16 + len(req), _SOCK_DIAG_BY_FAMILY, _NLM_F_REQUEST, 1, 0
    )
    with socket.socket(
        socket.AF_NETLINK, socket.SOCK_DGRAM, _NETLINK_SOCK_DIAG
    ) as nl:
        nl.settimeout(0.5)
        nl.sendto(header + req, (0, 0))
        data = nl.recv(65536)
    offset = 0
    while offset + 16 <= len(data):
        length, msg_type = struct.unpack_from("=IH", data, offset)
        if length < 16:
            break
        if msg_type in (_NLMSG_DONE, _NLMSG_ERROR):
            return None
        if msg_type == _SOCK_DIAG_BY_FAMILY:
            body = offset + 16
            # inet_diag_msg: 4 bytes, id (48), expires, rqueue, wqueue, uid, inode
            uid = struct.unpack_from("=I", data, body + 4 + 48 + 12)[0]
            return int(uid)
        offset += (length + 3) & ~3
    return None


def _proc_addr(text: str) -> tuple[ipaddress._BaseAddress, int]:
    host, port = text.split(":")
    raw = bytes.fromhex(host)
    # /proc prints each 32-bit word in host (little-endian) byte order.
    words = b"".join(raw[i : i + 4][::-1] for i in range(0, len(raw), 4))
    return ipaddress.ip_address(words), int(port, 16)


def _proc_uid(src, dst) -> int | None:
    want_src = (ipaddress.ip_address(src[0]), int(src[1]))
    want_dst = (ipaddress.ip_address(dst[0]), int(dst[1]))
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table, encoding="ascii") as handle:
                next(handle, None)
                for line in handle:
                    fields = line.split()
                    if len(fields) < 8:
                        continue
                    if (
                        _proc_addr(fields[1]) == want_src
                        and _proc_addr(fields[2]) == want_dst
                    ):
                        return int(fields[7])
        except (OSError, ValueError):
            continue
    return None


def _normalise(host: str) -> str:
    ip = ipaddress.ip_address(host.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return str(ip.ipv4_mapped)
    return str(ip)


def peer_uid(client, server) -> int | None:
    """Owner uid of the client end of this loopback connection, if knowable."""
    if not client or not server:
        return None
    try:
        client_host = ipaddress.ip_address(client[0].split("%", 1)[0])
        if not client_host.is_loopback and not (
            isinstance(client_host, ipaddress.IPv6Address)
            and client_host.ipv4_mapped is not None
            and client_host.ipv4_mapped.is_loopback
        ):
            return None
        src = (client[0].split("%", 1)[0], int(client[1]))
        dst = (server[0].split("%", 1)[0], int(server[1]))
    except (ValueError, TypeError, IndexError):
        return None
    family = socket.AF_INET6 if ":" in src[0] else socket.AF_INET
    try:
        uid = _diag_uid(family, src, dst)
    except OSError:
        uid = None
    if uid is None and family == socket.AF_INET6:
        # Dual-stack listener: the client may have connected over IPv4.
        try:
            uid = _diag_uid(
                socket.AF_INET, (_normalise(src[0]), src[1]),
                (_normalise(dst[0]), dst[1]),
            )
        except (OSError, ValueError):
            uid = None
    if uid is None:
        try:
            uid = _proc_uid(
                (_normalise(src[0]), src[1]), (_normalise(dst[0]), dst[1])
            )
        except ValueError:
            uid = None
    return uid


def is_foreign_peer(client, server) -> bool:
    if not _enabled():
        return False
    uid = peer_uid(client, server)
    return uid is not None and uid != 0 and uid != os.geteuid()


class PeerOwnerGuard:
    """ASGI middleware: refuse HTTP and WebSocket from other local users."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        kind = scope.get("type")
        if kind not in ("http", "websocket") or scope.get("path") in _OPEN_PATHS:
            await self.app(scope, receive, send)
            return
        if not is_foreign_peer(scope.get("client"), scope.get("server")):
            await self.app(scope, receive, send)
            return
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 4403})
            return
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(_DENY_MESSAGE)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": _DENY_MESSAGE})
