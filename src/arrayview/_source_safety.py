"""Network-source preflight and staging without touching the target in-process.

The public launcher imports this module before it performs ``stat``-like calls on
an explicit source.  Linux mount classification is derived from procfs, so the
target path itself is not resolved, opened, or inspected by the launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


_NETWORK_FILESYSTEMS = frozenset(
    {
        "9p",
        "afs",
        "ceph",
        "cifs",
        "coda",
        "fuse.curlftpfs",
        "fuse.davfs",
        "fuse.gcsfuse",
        "fuse.rclone",
        "fuse.s3fs",
        "fuse.sshfs",
        "glusterfs",
        "lustre",
        "nfs",
        "nfs4",
        "smb3",
    }
)
_KNOWN_LOCAL_STAGE_FILESYSTEMS = frozenset(
    {
        "btrfs",
        "ext2",
        "ext3",
        "ext4",
        "f2fs",
        "ramfs",
        "tmpfs",
        "xfs",
        "zfs",
    }
)
_SELF_CONTAINED_FILE_EXTENSIONS = (
    ".h5",
    ".hdf5",
    ".mat",
    ".nii",
    ".nii.gz",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".tif",
    ".tiff",
    ".zarr",
    ".zarr.zip",
)
_NETWORK_STAGE_EXTENSIONS = (
    ".nii",
    ".nii.gz",
    ".npy",
    ".npz",
)
_DEFAULT_STAGE_TIMEOUT_SECONDS = 30.0
_DEFAULT_QUARANTINE_SECONDS = 60.0


class UnsafeSourceError(RuntimeError):
    """An explicit source cannot safely be accessed by this launch."""


@dataclass(frozen=True)
class MountRecord:
    mount_id: int
    parent_id: int
    mountpoint: str
    filesystem: str
    source: str
    super_options: str


@dataclass(frozen=True)
class PreparedSource:
    original_path: str
    launch_path: str
    staging_dir: str | None
    mount: MountRecord | None

    @property
    def is_network(self) -> bool:
        return self.mount is not None and self.mount.filesystem in _NETWORK_FILESYSTEMS


def lexical_abspath(path: str) -> str:
    """Return an absolute path without resolving symlinks or inspecting it."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    pwd = os.environ.get("PWD")
    if pwd and os.path.isabs(pwd):
        return os.path.normpath(os.path.join(pwd, path))
    return os.path.abspath(path)


def _unescape_mount_field(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def parse_mountinfo(text: str) -> list[MountRecord]:
    records: list[MountRecord] = []
    for line in text.splitlines():
        try:
            left, right = line.split(" - ", 1)
            left_fields = left.split()
            right_fields = right.split()
            records.append(
                MountRecord(
                    mount_id=int(left_fields[0]),
                    parent_id=int(left_fields[1]),
                    mountpoint=_unescape_mount_field(left_fields[4]),
                    filesystem=right_fields[0].lower(),
                    source=_unescape_mount_field(right_fields[1]),
                    super_options=" ".join(right_fields[2:]),
                )
            )
        except (IndexError, ValueError):
            continue
    return records


def _mountinfo_path() -> str:
    return os.environ.get("_ARRAYVIEW_TEST_MOUNTINFO", "/proc/self/mountinfo")


def _cifs_debug_path() -> str:
    return os.environ.get("_ARRAYVIEW_TEST_CIFS_DEBUG", "/proc/fs/cifs/DebugData")


def read_mountinfo() -> list[MountRecord]:
    if not sys.platform.startswith("linux"):
        return []
    try:
        with open(_mountinfo_path(), encoding="utf-8", errors="replace") as stream:
            return parse_mountinfo(stream.read())
    except OSError:
        return []


def mount_for_path(path: str, records: list[MountRecord] | None = None) -> MountRecord | None:
    """Return the visible longest-prefix mount without touching *path*."""
    absolute = lexical_abspath(path)
    candidates = []
    for record in records if records is not None else read_mountinfo():
        mountpoint = os.path.normpath(record.mountpoint)
        if absolute == mountpoint or absolute.startswith(mountpoint.rstrip("/") + "/"):
            candidates.append(record)
    if not candidates:
        return None
    longest = max(len(os.path.normpath(item.mountpoint)) for item in candidates)
    same_point = [
        item for item in candidates if len(os.path.normpath(item.mountpoint)) == longest
    ]
    if len(same_point) == 1:
        return same_point[0]
    # For stacked mounts, the visible record is the one that is not the parent
    # of another candidate at the same mountpoint.
    parent_ids = {item.parent_id for item in same_point}
    visible = [item for item in same_point if item.mount_id not in parent_ids]
    return max(visible or same_point, key=lambda item: item.mount_id)


def _mount_addresses(mount: MountRecord) -> set[str]:
    values = set()
    for match in re.finditer(r"(?:^|,)addr=([^, ]+)", mount.super_options):
        values.add(match.group(1).strip("[]").lower())
    return values


def cifs_connection_state(mount: MountRecord, debug_text: str | None = None) -> str:
    """Return ``healthy``, ``disconnected``, or ``unknown`` for a CIFS mount."""
    if mount.filesystem not in {"cifs", "smb3"}:
        return "unknown"
    if debug_text is None:
        try:
            with open(_cifs_debug_path(), encoding="utf-8", errors="replace") as stream:
                debug_text = stream.read()
        except OSError:
            return "unknown"

    addresses = _mount_addresses(mount)
    source_key = mount.source.replace("/", "\\").lower().rstrip("\\")
    blocks = re.split(r"(?=\n\d+\) ConnectionId:)", "\n" + debug_text)
    exact_matches: list[tuple[str, list[str]]] = []
    exact_address_matches: list[tuple[str, list[str]]] = []
    address_matches: list[tuple[str, list[str]]] = []
    for block in blocks:
        lines = block.splitlines()
        shares_at = next(
            (index for index, line in enumerate(lines) if line.strip().lower() == "shares:"),
            None,
        )
        share_stanzas: list[list[str]] = []
        if shares_at is not None:
            current: list[str] = []
            for line in lines[shares_at + 1 :]:
                if line.strip().lower() in {
                    "extra channels:",
                    "servers:",
                    "sessions:",
                }:
                    break
                if re.match(r"^\s*\d+\)\s", line):
                    if current:
                        share_stanzas.append(current)
                    current = [line]
                elif current:
                    current.append(line)
            if current:
                share_stanzas.append(current)

        def _matches_source(stanza: list[str]) -> bool:
            if not source_key:
                return False
            normalized = "\n".join(stanza).lower().replace("/", "\\")
            start = 0
            while (found := normalized.find(source_key, start)) >= 0:
                end = found + len(source_key)
                if end == len(normalized) or normalized[end] in {"\\", " ", "\t", "\n"}:
                    return True
                start = found + 1
            return False

        relevant_stanzas = [stanza for stanza in share_stanzas if _matches_source(stanza)]
        block_addresses = {
            value.strip("[]").lower()
            for value in re.findall(r"\bAddress:\s*([^\s]+)", block, re.IGNORECASE)
        }
        session_addresses: set[str] = set()
        for line in lines:
            if not re.search(r"(?:SMB session status|Session Status):", line, re.I):
                continue
            for value in re.findall(r"\b(?:Address|Name):\s*([^\s]+)", line, re.I):
                session_addresses.add(value.strip("[]").lower())
        match = (block, ["\n".join(stanza) for stanza in relevant_stanzas])
        if relevant_stanzas:
            exact_matches.append(match)
            if addresses and addresses.intersection(session_addresses):
                exact_address_matches.append(match)
        elif addresses and addresses.intersection(block_addresses):
            address_matches.append(match)
    matches = exact_address_matches or exact_matches or address_matches
    if not matches:
        return "unknown"

    states = []
    for block, share_lines in matches:
        primary_tcp = re.search(r"TCP status:\s*(\d+)", block, re.IGNORECASE)
        tcp_numeric = [int(primary_tcp.group(1))] if primary_tcp else []
        connection_numeric = [
            int(value)
            for value in re.findall(
                r"(?:SMB session status|Session Status):\s*(\d+)",
                block,
                flags=re.IGNORECASE,
            )
        ]
        share_bad = any(
            "DISCONNECTED" in line.upper()
            or "RECONNECTING" in line.upper()
            or re.search(r"\bStatus:\s*[23]\b", line, re.IGNORECASE)
            for line in share_lines
        )
        share_good = bool(share_lines) and all(
            re.search(r"\bStatus:\s*1\b", line, re.IGNORECASE)
            for line in share_lines
        )
        if share_bad or any(value in {2, 3} for value in tcp_numeric) or (
            connection_numeric and all(value in {2, 3} for value in connection_numeric)
        ):
            states.append("disconnected")
        elif (
            tcp_numeric
            and all(value == 1 for value in tcp_numeric)
            and connection_numeric
            and all(value == 1 for value in connection_numeric)
            and share_good
        ):
            states.append("healthy")
        else:
            states.append("unknown")
    if states and all(state == "disconnected" for state in states):
        return "disconnected"
    if states and all(state == "healthy" for state in states):
        return "healthy"
    return "unknown"


def _quarantine_path(mount: MountRecord) -> Path:
    identity = f"{mount.filesystem}\0{mount.mountpoint}\0{mount.source}".encode()
    token = hashlib.sha256(identity).hexdigest()[:20]
    return _safe_storage_root() / "mount-guards" / token


def _safe_storage_root() -> Path:
    """Return an ArrayView-owned root on a mount known not to be network-backed."""
    override = os.environ.get("_ARRAYVIEW_TEST_SAFE_SOURCE_ROOT")
    if override:
        root = Path(override)
    else:
        records = read_mountinfo()
        if not records:
            raise UnsafeSourceError(
                "Linux mount information is unavailable, so ArrayView cannot "
                "choose safe local storage for a network source."
            )
        candidates = [
            os.environ.get("XDG_RUNTIME_DIR"),
            "/dev/shm",
            "/var/tmp",
            "/tmp",
        ]
        root = None
        for candidate in candidates:
            if not candidate or not os.path.isabs(candidate):
                continue
            suffix = str(os.getuid()) if hasattr(os, "getuid") else str(os.getpid())
            proposed = Path(candidate) / f"arrayview-source-{suffix}"
            mount = mount_for_path(str(proposed), records)
            if mount is not None and mount.filesystem in _KNOWN_LOCAL_STAGE_FILESYSTEMS:
                root = proposed
                break
        if root is None:
            raise UnsafeSourceError(
                "No known-local staging directory is available for this network source."
            )
    if override:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        return root

    base = root.parent
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        base_fd = os.open(base, flags)
        try:
            try:
                os.mkdir(root.name, mode=0o700, dir_fd=base_fd)
            except FileExistsError:
                pass
            root_fd = os.open(root.name, flags, dir_fd=base_fd)
            try:
                stat_result = os.fstat(root_fd)
                if hasattr(os, "getuid") and stat_result.st_uid != os.getuid():
                    raise UnsafeSourceError(
                        "The local ArrayView staging root has another owner."
                    )
                os.fchmod(root_fd, 0o700)
            finally:
                os.close(root_fd)
        finally:
            os.close(base_fd)
    except OSError as exc:
        raise UnsafeSourceError(
            "The selected local staging root is unsafe or unavailable."
        ) from exc
    return root


def validated_staging_directory(filepath: str, claimed: str | None) -> str | None:
    """Accept cleanup authority only for a direct child of our private root."""
    if not claimed:
        return None
    root = lexical_abspath(str(_safe_storage_root() / "staging"))
    directory = lexical_abspath(str(claimed))
    file_parent = os.path.dirname(lexical_abspath(filepath))
    if os.path.dirname(directory) != root or file_parent != directory:
        return None
    return directory


def _ensure_private_directory(path: Path, *, exist_ok: bool) -> None:
    """Create/open one child without following a symlink at that child."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(path.parent, flags)
    except OSError as exc:
        raise UnsafeSourceError(
            f"ArrayView private directory parent {str(path.parent)!r} is unsafe."
        ) from exc
    try:
        try:
            os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            if not exist_ok:
                raise
        try:
            child_fd = os.open(path.name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise UnsafeSourceError(
                f"ArrayView private directory {str(path)!r} is unsafe."
            ) from exc
        try:
            stat_result = os.fstat(child_fd)
            if hasattr(os, "getuid") and stat_result.st_uid != os.getuid():
                raise UnsafeSourceError(
                    f"ArrayView private directory {str(path)!r} has another owner."
                )
            os.fchmod(child_fd, 0o700)
        finally:
            os.close(child_fd)
    finally:
        os.close(parent_fd)


def cleanup_staging_directory(directory: str) -> bool:
    root = lexical_abspath(str(_safe_storage_root() / "staging"))
    candidate = lexical_abspath(directory)
    if os.path.dirname(candidate) != root:
        return False
    shutil.rmtree(candidate, ignore_errors=True)
    return True


def direct_network_mount(path: str) -> MountRecord | None:
    mount = mount_for_path(path)
    if mount is not None and mount.filesystem in _NETWORK_FILESYSTEMS:
        return mount
    return None


def network_mount_below(path: str) -> MountRecord | None:
    """Return a known network submount a recursive scan could descend into."""
    root = lexical_abspath(path).rstrip("/") + "/"
    for mount in read_mountinfo():
        point = os.path.normpath(mount.mountpoint)
        if point.startswith(root) and mount.filesystem in _NETWORK_FILESYSTEMS:
            return mount
    return None


def scan_root_before_magic(pattern: str) -> str:
    """Return the nearest complete directory ancestor before glob syntax."""
    absolute = lexical_abspath(pattern)
    components = Path(absolute).parts
    safe: list[str] = []
    for component in components:
        if any(marker in component for marker in ("*", "?", "[")):
            break
        safe.append(component)
    if len(safe) == len(components):
        return absolute
    return os.path.join(*safe) if safe else os.path.sep


def source_may_require_sibling_discovery(path: str) -> bool:
    """Whether loading may inspect the source's parent for related files."""
    lower = lexical_abspath(path).lower()
    return lower.endswith(".dcm") or not lower.endswith(_SELF_CONTAINED_FILE_EXTENSIONS)


def discovery_network_mount(path: str) -> MountRecord | None:
    """Return a network mount that loading *path* could discover indirectly."""
    absolute = lexical_abspath(path)
    nested = network_mount_below(absolute)
    if nested is not None:
        return nested
    if source_may_require_sibling_discovery(absolute):
        return network_mount_below(os.path.dirname(absolute))
    return None


def _check_quarantine(mount: MountRecord) -> None:
    guard = _quarantine_path(mount)
    if guard.is_symlink():
        raise UnsafeSourceError(
            f"Network mount {mount.mountpoint!r} has an unsafe ArrayView source guard."
        )
    try:
        state = json.loads((guard / "state.json").read_text())
    except FileNotFoundError:
        if not guard.exists():
            return
        age = time.time() - guard.stat().st_mtime
        if age >= _DEFAULT_QUARANTINE_SECONDS:
            shutil.rmtree(guard, ignore_errors=True)
            return
        raise UnsafeSourceError(
            f"Network mount {mount.mountpoint!r} has an incomplete ArrayView "
            "source guard; retry after the quarantine interval."
        )
    except (OSError, ValueError, TypeError):
        try:
            age = time.time() - guard.stat().st_mtime
        except OSError:
            age = 0
        if age >= _DEFAULT_QUARANTINE_SECONDS:
            shutil.rmtree(guard, ignore_errors=True)
            return
        raise UnsafeSourceError(
            f"Network mount {mount.mountpoint!r} has an unreadable ArrayView "
            "source guard; retry after the quarantine interval."
        )
    from ._instance_registry import process_start_identity

    pid = int(state.get("pid", 0))
    saved_start = state.get("process_start")
    current_start = process_start_identity(pid) if pid > 0 else None
    live = saved_start is not None and current_start is not None and current_start == saved_start
    status = str(state.get("status", "in_progress"))
    age = time.time() - float(state.get("updated_at", time.time()))
    if live:
        raise UnsafeSourceError(
            f"Network mount {mount.mountpoint!r} is quarantined after a timed-out "
            "source access. No further helper will be started until the existing "
            "helper exits after the mount recovers or is detached."
        )
    if status in {"reaped", "unreaped", "in_progress"} and age < _DEFAULT_QUARANTINE_SECONDS:
        raise UnsafeSourceError(
            f"Network mount {mount.mountpoint!r} is temporarily quarantined after "
            "a timed-out source access."
        )
    staging_dir = state.get("staging_dir")
    if isinstance(staging_dir, str):
        cleanup_staging_directory(staging_dir)
    shutil.rmtree(guard, ignore_errors=True)


def _acquire_mount_guard(mount: MountRecord) -> Path:
    guard = _quarantine_path(mount)
    _ensure_private_directory(guard.parent, exist_ok=True)
    for _ in range(2):
        _check_quarantine(mount)
        try:
            _ensure_private_directory(guard, exist_ok=False)
        except FileExistsError:
            continue
        try:
            _write_guard_state(guard, os.getpid(), "in_progress")
        except BaseException:
            shutil.rmtree(guard, ignore_errors=True)
            raise
        return guard
    raise UnsafeSourceError(
        f"Another ArrayView source operation is already active for network mount "
        f"{mount.mountpoint!r}."
    )


def _write_guard_state(
    guard: Path,
    pid: int,
    status: str,
    *,
    staging_dir: str | None = None,
) -> None:
    from ._instance_registry import process_start_identity

    payload = {
        "pid": pid,
        "process_start": process_start_identity(pid),
        "status": status,
        "updated_at": time.time(),
        "staging_dir": staging_dir,
    }
    temporary = guard / f".state-{os.getpid()}-{time.monotonic_ns()}.tmp"
    try:
        temporary.write_text(json.dumps(payload))
        os.replace(temporary, guard / "state.json")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _stop_helper(process: subprocess.Popen) -> bool:
    """Best-effort bounded stop. False means the kernel did not reap the child."""
    try:
        process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=0.5)
        return True
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=0.5)
        return True
    except subprocess.TimeoutExpired:
        return False


def _stage_network_file(path: str, mount: MountRecord, timeout: float) -> PreparedSource:
    guard = _acquire_mount_guard(mount)
    staging_dir = None
    try:
        staging_root = _safe_storage_root() / "staging"
        _ensure_private_directory(staging_root, exist_ok=True)
        import uuid

        staging_path = staging_root / uuid.uuid4().hex
        staging_dir = str(staging_path)
        _write_guard_state(
            guard,
            os.getpid(),
            "in_progress",
            staging_dir=staging_dir,
        )
        _ensure_private_directory(staging_path, exist_ok=False)
    except BaseException:
        if staging_dir is not None:
            cleanup_staging_directory(staging_dir)
        shutil.rmtree(guard, ignore_errors=True)
        raise
    filename = os.path.basename(path) or "source"
    destination = os.path.join(staging_dir, filename)
    command = [sys.executable, "-m", "arrayview._source_safety", "--copy", path, destination]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            **({"start_new_session": True} if os.name != "nt" else {}),
        )
        _write_guard_state(
            guard,
            process.pid,
            "in_progress",
            staging_dir=staging_dir,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            reaped = _stop_helper(process)
            try:
                _write_guard_state(
                    guard,
                    process.pid,
                    "reaped" if reaped else "unreaped",
                    staging_dir=staging_dir,
                )
            except OSError:
                pass
            cleanup_staging_directory(staging_dir)
            limitation = (
                " The helper could not be reaped because the kernel is still waiting "
                "on the mount; it can only exit after the mount recovers or is detached."
                if not reaped
                else ""
            )
            raise UnsafeSourceError(
                f"Timed out after {timeout:g}s while staging a file from network mount "
                f"{mount.mountpoint!r}.{limitation}"
            )
    except UnsafeSourceError:
        raise
    except BaseException as exc:
        if "process" in locals():
            reaped = _stop_helper(process)
            try:
                _write_guard_state(
                    guard,
                    process.pid,
                    "reaped" if reaped else "unreaped",
                    staging_dir=staging_dir,
                )
            except OSError:
                pass
        else:
            shutil.rmtree(guard, ignore_errors=True)
        cleanup_staging_directory(staging_dir)
        if isinstance(exc, Exception):
            raise UnsafeSourceError(
                f"Could not start the bounded helper for network mount "
                f"{mount.mountpoint!r}: {exc}"
            ) from exc
        raise
    if returncode != 0:
        cleanup_staging_directory(staging_dir)
        shutil.rmtree(guard, ignore_errors=True)
        raise UnsafeSourceError(
            f"Could not stage the source from network mount {mount.mountpoint!r}. "
            "Only regular self-contained files are supported safely; copy the "
            "source to local storage first for directory or related-file inputs."
        )
    shutil.rmtree(guard, ignore_errors=True)
    return PreparedSource(path, destination, staging_dir, mount)


def prepare_source(path: str, *, timeout: float | None = None) -> PreparedSource:
    """Preflight and, for a network file, make a bounded local snapshot."""
    absolute = lexical_abspath(path)
    mount = mount_for_path(absolute)
    if os.environ.get("ARRAYVIEW_SKIP_SOURCE_STAGING") == "1":
        # Opt out of network-source staging entirely: use the source path
        # directly, exactly like a local file. This restores the pre-staging
        # behavior (fast open, but a dropped network mount while the file is
        # memory-mapped can wedge the process in a way not even `kill -9`
        # recovers from — see docs/loading.md).
        return PreparedSource(absolute, absolute, None, mount)
    if mount is None or mount.filesystem not in _NETWORK_FILESYSTEMS:
        unsafe_discovery = discovery_network_mount(absolute)
        if unsafe_discovery is not None:
            raise UnsafeSourceError(
                f"Source or related-file discovery would enter network mount "
                f"{unsafe_discovery.mountpoint!r}. Copy the collection or DICOM "
                "series to local storage first."
            )
        return PreparedSource(absolute, absolute, None, mount)

    lower = absolute.lower()
    if source_may_require_sibling_discovery(absolute):
        raise UnsafeSourceError(
            "Network-mounted DICOM files require sibling discovery. Copy the series "
            "directory to local storage before launching ArrayView."
        )
    if not lower.endswith(_NETWORK_STAGE_EXTENSIONS):
        raise UnsafeSourceError(
            "This network-mounted format may reference other files while loading. "
            "Copy it to local storage before launching ArrayView."
        )

    if mount.filesystem in {"cifs", "smb3"}:
        state = cifs_connection_state(mount)
        if state == "disconnected":
            raise UnsafeSourceError(
                f"Network mount {mount.mountpoint!r} is disconnected or reconnecting. "
                "ArrayView did not access the source. Recover or detach the mount, "
                "then retry."
            )
    seconds = (
        float(os.environ.get("ARRAYVIEW_SOURCE_TIMEOUT_SECONDS", _DEFAULT_STAGE_TIMEOUT_SECONDS))
        if timeout is None
        else float(timeout)
    )
    return _stage_network_file(absolute, mount, max(0.1, seconds))


def cleanup_prepared_sources(sources: list[PreparedSource]) -> None:
    for directory in dict.fromkeys(source.staging_dir for source in sources if source.staging_dir):
        cleanup_staging_directory(directory)


def _copy_main(source: str, destination: str) -> int:
    try:
        test_sleep = os.environ.get("_ARRAYVIEW_TEST_COPY_SLEEP_SECONDS")
        if test_sleep:
            time.sleep(float(test_sleep))
        if not os.path.isfile(source):
            return 3
        shutil.copyfile(source, destination)
        return 0
    except Exception:
        return 2


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--copy":
        raise SystemExit(_copy_main(sys.argv[2], sys.argv[3]))
    raise SystemExit(2)
