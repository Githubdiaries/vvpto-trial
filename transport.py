"""
transport.py - sends vein masks from the Pi to the BeagleBone over Ethernet.

Wire format (one message per frame, over a single TCP connection):

    4 bytes  magic  b"VEIN"
    4 bytes  frame sequence number     (big-endian)
    2 bytes  width
    2 bytes  height
    W*H bytes raw 8-bit mask           (255 = vein)

A 320x240 mask is 77 KB, so 30 fps is only ~2.3 MB/s: trivial for Ethernet and
no compression CPU needed on either board.

Both ends keep only the NEWEST frame. If a board is briefly slow, old frames
are dropped instead of queueing up, so the projected image never lags behind.
"""
import socket
import struct
import threading
import time
from typing import NamedTuple, Optional

import numpy as np

_MAGIC = b"VEIN"
_HEADER = struct.Struct("!4sIHH")


class ReceivedFrame(NamedTuple):
    seq: int
    mask: np.ndarray        # uint8, H x W
    received_at: float      # time.time() when it arrived


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    """Read exactly n bytes or raise ConnectionError if the peer closes."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed the connection")
        got += k
    return buf


# ==========================================================================
# Pi side
# ==========================================================================
class FrameSender:
    """
    Background thread that connects to the BeagleBone (retrying forever, so
    start order does not matter) and sends the latest submitted mask.
    """

    def __init__(self, host: str, port: int) -> None:
        self._addr = (host, port)
        self._cond = threading.Condition()
        self._latest: Optional[np.ndarray] = None
        self._seq = 0
        self._running = True
        self.connected = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, mask: np.ndarray) -> None:
        """Called from the main loop. Never blocks; replaces any unsent frame."""
        with self._cond:
            self._latest = mask
            self._seq += 1
            self._cond.notify()

    def close(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify()
        self._thread.join(timeout=2)

    # ---- internals ------------------------------------------------------
    def _connect(self) -> Optional[socket.socket]:
        while self._running:
            try:
                sock = socket.create_connection(self._addr, timeout=2.0)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(1.0)       # a stuck send must not hang forever
                self.connected = True
                return sock
            except OSError:
                self.connected = False
                time.sleep(1.0)
        return None

    def _run(self) -> None:
        sock = self._connect()
        while self._running and sock is not None:
            with self._cond:
                while self._latest is None and self._running:
                    self._cond.wait(timeout=0.5)
                mask, seq = self._latest, self._seq
                self._latest = None
            if mask is None:
                continue
            h, w = mask.shape
            try:
                sock.sendall(_HEADER.pack(_MAGIC, seq, w, h) + mask.tobytes())
            except OSError:                # link dropped: reconnect
                sock.close()
                self.connected = False
                sock = self._connect()
        if sock is not None:
            sock.close()


# ==========================================================================
# BeagleBone side
# ==========================================================================
class FrameReceiver:
    """
    Background thread that accepts the Pi's connection and keeps the newest
    received mask. The display loop just polls latest().
    """

    def __init__(self, port: int, bind: str = "0.0.0.0") -> None:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((bind, port))
        self._srv.listen(1)
        self._srv.settimeout(1.0)          # so close() can stop the thread

        self._lock = threading.Lock()
        self._latest: Optional[ReceivedFrame] = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def latest(self) -> Optional[ReceivedFrame]:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
        self._srv.close()

    # ---- internals ------------------------------------------------------
    def _run(self) -> None:
        while self._running:
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.settimeout(5.0)
            try:
                while self._running:
                    magic, seq, w, h = _HEADER.unpack(_recv_exact(conn, _HEADER.size))
                    if magic != _MAGIC:
                        raise ValueError("bad magic, stream out of sync")
                    payload = _recv_exact(conn, w * h)
                    mask = np.frombuffer(payload, np.uint8).reshape(h, w)
                    with self._lock:
                        self._latest = ReceivedFrame(seq, mask, time.time())
            except (OSError, ConnectionError, ValueError):
                pass                       # drop connection, wait for the next
            finally:
                conn.close()
