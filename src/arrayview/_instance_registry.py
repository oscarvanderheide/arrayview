"""Small, dependency-free registry for ArrayView server processes.

This module deliberately contains no server or launcher integration.  It is safe
to import on CLI fast paths.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import ctypes
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Iterator
import uuid


REGISTRY_SCHEMA = 1


@dataclass(frozen=True)
class InstanceRecord:
    instance_id: str
    pid: int
    process_start: str
    port: int
    protocol_version: str
    package_version: str
    owner_mode: str
    started_at: float
    last_seen_at: float
    control_token: str
    log_path: str

    @classmethod
    def create(
        cls, *, port: int, protocol_version: str, package_version: str,
        owner_mode: str, log_path: str, pid: int | None = None,
    ) -> "InstanceRecord":
        pid = os.getpid() if pid is None else pid
        identity = process_start_identity(pid)
        if identity is None:
            raise ProcessLookupError(pid)
        now = time.time()
        return cls(str(uuid.uuid4()), pid, identity, port, protocol_version,
                   package_version, owner_mode, now, now,
                   uuid.uuid4().hex + uuid.uuid4().hex, log_path)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "InstanceRecord":
        if value.get("schema") != REGISTRY_SCHEMA:
            raise ValueError("unsupported instance registry schema")
        fields = {key: value[key] for key in cls.__dataclass_fields__}
        return cls(**fields)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"schema": REGISTRY_SCHEMA, **asdict(self)}


def runtime_directory() -> Path:
    override = os.environ.get("ARRAYVIEW_RUNTIME_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_RUNTIME_DIR")
    if base:
        return Path(base) / "arrayview"
    user = getpass.getuser().encode("utf-8", "replace")
    suffix = hashlib.sha256(user).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"arrayview-{suffix}"


def process_start_identity(pid: int) -> str | None:
    """Return an OS process birth identity, or ``None`` when PID is not alive."""
    if pid <= 0:
        return None
    if os.name == "nt":
        return _windows_process_start(pid)
    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        # comm may contain spaces and parentheses; fields after its final ')' are stable.
        return "linux:" + proc_stat.read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, PermissionError, IndexError, OSError):
        pass
    if sys_platform() == "darwin":
        try:
            out = subprocess.run(
                ["ps", "-p", str(pid), "-o", "lstart="], capture_output=True,
                text=True, timeout=2, check=False,
            ).stdout.strip()
            return "darwin:" + out if out else None
        except (OSError, subprocess.SubprocessError):
            return None
    try:
        os.kill(pid, 0)
    except PermissionError:
        return f"pid-only:{pid}"
    except (ProcessLookupError, OSError):
        return None
    return f"pid-only:{pid}"


def sys_platform() -> str:
    import sys
    return sys.platform


def _windows_process_start(pid: int) -> str | None:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    creation = ctypes.c_ulonglong()
    exit_time = ctypes.c_ulonglong()
    kernel = ctypes.c_ulonglong()
    user = ctypes.c_ulonglong()
    try:
        ok = kernel32.GetProcessTimes(handle, ctypes.byref(creation),
                                      ctypes.byref(exit_time), ctypes.byref(kernel),
                                      ctypes.byref(user))
        return f"windows:{creation.value}" if ok else None
    finally:
        kernel32.CloseHandle(handle)


class InstanceRegistry:
    def __init__(self, directory: Path | str | None = None):
        self.directory = Path(directory) if directory is not None else runtime_directory()
        self.records = self.directory / "instances"
        self.lock_path = self.directory / "startup.lock"

    def _prepare(self) -> None:
        self.records.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.directory, 0o700)
            os.chmod(self.records, 0o700)
        except OSError:
            pass

    def write(self, record: InstanceRecord) -> Path:
        self._prepare()
        destination = self.records / f"{record.instance_id}.json"
        fd, temporary = tempfile.mkstemp(prefix=".record-", dir=self.records)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(record.to_dict(), stream, separators=(",", ":"), sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return destination

    def remove(self, instance_id: str) -> bool:
        try:
            (self.records / f"{instance_id}.json").unlink()
            return True
        except FileNotFoundError:
            return False

    def discover(self, *, clean_stale: bool = False) -> list[InstanceRecord]:
        found: list[InstanceRecord] = []
        if not self.records.is_dir():
            return found
        for path in self.records.glob("*.json"):
            try:
                record = InstanceRecord.from_dict(json.loads(path.read_text()))
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                if clean_stale:
                    path.unlink(missing_ok=True)
                continue
            if is_stale(record):
                if clean_stale:
                    path.unlink(missing_ok=True)
                continue
            found.append(record)
        return sorted(found, key=lambda item: item.started_at)

    @contextmanager
    def startup_lock(self, *, timeout: float = 10.0,
                     poll_interval: float = 0.02) -> Iterator[None]:
        self._prepare()
        deadline = time.monotonic() + timeout
        identity = process_start_identity(os.getpid()) or "unknown"
        lock_token = uuid.uuid4().hex
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "process_start": identity,
                "token": lock_token,
            }
        )
        while True:
            try:
                self.lock_path.mkdir()
                (self.lock_path / "owner.json").write_text(payload)
                break
            except FileExistsError:
                if self._break_stale_lock():
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for ArrayView startup lock")
                time.sleep(poll_interval)
        try:
            yield
        finally:
            try:
                owner_path = self.lock_path / "owner.json"
                owner = json.loads(owner_path.read_text())
                if owner.get("token") == lock_token:
                    owner_path.unlink(missing_ok=True)
                    self.lock_path.rmdir()
            except (OSError, json.JSONDecodeError):
                pass

    def _break_stale_lock(self) -> bool:
        try:
            owner = json.loads((self.lock_path / "owner.json").read_text())
            live = process_start_identity(int(owner["pid"]))
            stale = live != owner["process_start"]
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            # A creator may not yet have written owner.json. Only reap an old lock.
            try:
                stale = time.time() - self.lock_path.stat().st_mtime > 5
            except OSError:
                return True
        if not stale:
            return False
        try:
            (self.lock_path / "owner.json").unlink(missing_ok=True)
            self.lock_path.rmdir()
            return True
        except OSError:
            return False


def is_stale(record: InstanceRecord) -> bool:
    return process_start_identity(record.pid) != record.process_start
