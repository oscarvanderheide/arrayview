import asyncio
import io
import os

import numpy as np
from fastapi import File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from arrayview._io import _SUPPORTED_EXTS, _peek_file_shape, load_data
from arrayview._session import SESSIONS, Session


def register_loading_routes(app, *, notify_shells, setup_rgb) -> None:
    @app.get("/listfiles")
    def list_files(directory: str = ""):
        """List supported array files recursively (depth <= 4, max 300 files)."""
        target = os.path.abspath(directory) if directory else os.getcwd()
        max_files = 300
        max_depth = 4
        results = []
        try:
            for root, dirs, files in os.walk(target):
                rel_root = os.path.relpath(root, target)
                depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                if depth >= max_depth:
                    dirs.clear()
                for fname in sorted(files):
                    name_lower = fname.lower()
                    ext = (
                        ".nii.gz"
                        if name_lower.endswith(".nii.gz")
                        else os.path.splitext(name_lower)[1]
                    )
                    if ext not in _SUPPORTED_EXTS:
                        continue
                    fpath = os.path.join(root, fname)
                    rel = os.path.relpath(fpath, target)
                    name = fname if root == target else rel
                    try:
                        file_size = os.path.getsize(fpath)
                    except OSError:
                        continue
                    shape = _peek_file_shape(fpath, ext)
                    results.append(
                        {
                            "name": name,
                            "path": fpath,
                            "size": file_size,
                            "shape": shape,
                        }
                    )
                    if len(results) >= max_files:
                        break
                if len(results) >= max_files:
                    break
        except Exception as e:
            return {"error": str(e)}
        return results

    @app.get("/sessions")
    def get_sessions():
        """Return active sessions with metadata for the picker sidebar."""
        result = []
        for session in SESSIONS.values():
            dtype_str = str(getattr(session.data, "dtype", "unknown"))
            result.append(
                {
                    "sid": session.sid,
                    "name": session.name,
                    "shape": [int(x) for x in session.shape],
                    "filepath": session.filepath,
                    "dtype": dtype_str,
                    "estimated_mem": session._estimated_mem,
                }
            )
        return result

    @app.post("/load")
    async def load_file(request: Request):
        """Load a file into a new session. Optionally notify webview shells."""
        body = await request.json()
        filepath = str(body["filepath"])
        name = str(body.get("name") or os.path.basename(filepath))
        notify = bool(body.get("notify", False))
        abs_path = os.path.abspath(filepath)
        for existing in SESSIONS.values():
            if existing.filepath and os.path.abspath(existing.filepath) == abs_path:
                return {"sid": existing.sid, "name": existing.name, "notified": False}
        if not os.environ.get("ARRAYVIEW_SKIP_RAM_GUARD"):
            from ._io import FULL_LOAD_EXTS

            ext = os.path.splitext(filepath)[1].lower()
            if filepath.lower().endswith(".nii.gz"):
                ext = ".nii.gz"
            if ext in FULL_LOAD_EXTS:
                try:
                    import psutil

                    file_size = os.path.getsize(abs_path)
                    available = psutil.virtual_memory().available
                    if file_size > available:
                        return JSONResponse(
                            {
                                "error": "insufficient_memory",
                                "estimated_bytes": file_size,
                                "available_bytes": available,
                                "filename": os.path.basename(filepath),
                            },
                            status_code=507,
                        )
                except ImportError:
                    pass
        try:
            from ._io import load_data_with_meta

            data, spatial_meta = await asyncio.to_thread(load_data_with_meta, filepath)
        except Exception as e:
            return {"error": str(e)}
        session = await asyncio.to_thread(Session, data, filepath=filepath, name=name)
        if spatial_meta is not None:
            session.spatial_meta = spatial_meta
            session.original_volume = data
        if body.get("rgb"):
            try:
                await asyncio.to_thread(setup_rgb, session)
            except ValueError as e:
                return {"error": str(e)}
        SESSIONS[session.sid] = session
        notified = False
        if notify:
            tab_url = None
            if body.get("compare_sids"):
                tab_url = (
                    f"/?sid={session.sid}"
                    f"&compare_sid={body['compare_sid']}"
                    f"&compare_sids={body['compare_sids']}"
                )
            notified = await notify_shells(session.sid, name, url=tab_url, wait=False)
        return {"sid": session.sid, "name": name, "notified": notified}

    @app.get("/fs/list")
    def fs_list(
        path: str | None = None,
        base_sid: str | None = None,
        mode: str | None = None,
    ):
        """List directory entries for the filesystem picker."""
        home = os.path.realpath(os.path.expanduser("~"))
        target = os.path.realpath(path) if path else home
        if not target.startswith(home):
            target = home
        if not os.path.isdir(target):
            target = home

        base_shape: tuple | None = None
        if base_sid and mode in ("overlay", "vectorfield"):
            session = SESSIONS.get(base_sid)
            if session is not None:
                base_shape = tuple(int(x) for x in session.shape)

        entries: list[dict] = []
        try:
            with os.scandir(target) as it:
                for entry in it:
                    try:
                        if entry.name.startswith("."):
                            continue
                        is_dir = entry.is_dir(follow_symlinks=False)
                        if is_dir:
                            entries.append(
                                {
                                    "name": entry.name,
                                    "path": os.path.join(target, entry.name),
                                    "is_dir": True,
                                    "size": None,
                                }
                            )
                            continue
                        lower = entry.name.lower()
                        ext = (
                            ".nii.gz"
                            if lower.endswith(".nii.gz")
                            else ".zarr.zip"
                            if lower.endswith(".zarr.zip")
                            else os.path.splitext(lower)[1]
                        )
                        if ext not in _SUPPORTED_EXTS:
                            continue
                        full = os.path.join(target, entry.name)
                        shape = None
                        if base_shape is not None:
                            shape = _peek_shape(full)
                            if not _shape_compatible(base_shape, shape, mode or ""):
                                continue
                        st = entry.stat(follow_symlinks=False)
                        entries.append(
                            {
                                "name": entry.name,
                                "path": full,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "shape": list(shape) if shape else None,
                            }
                        )
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError) as e:
            return {"error": str(e)}

        entries.sort(key=lambda item: (not item["is_dir"], item["name"].lower()))

        parent = os.path.dirname(target) if target != home else None
        if parent and not parent.startswith(home):
            parent = None

        return {"cwd": target, "parent": parent, "home": home, "entries": entries}

    @app.post("/load-upload")
    async def load_upload(file: UploadFile = File(...)):
        """Accept a drag-and-dropped .npy or .mat file and create a new session."""
        import tempfile

        filename = file.filename or "array"
        ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
        if ext not in (".npy", ".mat"):
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {ext or '(none)'}"
            )

        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            data = await asyncio.to_thread(load_data, tmp_path)
        except Exception as e:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        session = await asyncio.to_thread(Session, data, name=filename)
        SESSIONS[session.sid] = session
        await notify_shells(session.sid, filename, wait=False)
        resp_shape = [
            int(s)
            for s in (
                session.spatial_shape if session.rgb_axis is not None else session.shape
            )
        ]
        return {"sid": session.sid, "name": filename, "shape": resp_shape}

    @app.post("/load_bytes")
    async def load_bytes_endpoint(request: Request):
        """Accept a base64-encoded .npy array from a remote machine."""
        import base64

        body = await request.json()
        data_b64 = body.get("data_b64", "")
        name = str(body.get("name") or "array")
        rgb = bool(body.get("rgb", False))

        try:
            raw = base64.b64decode(data_b64)
            arr = np.load(io.BytesIO(raw))
        except Exception as e:
            return {"error": f"Failed to decode array: {e}"}

        session = await asyncio.to_thread(Session, arr, name=name)
        if rgb:
            try:
                await asyncio.to_thread(setup_rgb, session)
            except ValueError as e:
                return {"error": str(e)}
        SESSIONS[session.sid] = session

        import arrayview._session as _session_mod
        from arrayview._vscode import _open_via_signal_file

        port = _session_mod.SERVER_PORT or 8000
        url = f"http://localhost:{port}/?sid={session.sid}"
        _open_via_signal_file(url)
        return {"sid": session.sid, "url": url}


def _peek_shape(path: str) -> tuple | None:
    """Cheaply read the shape of an array file without loading its data."""
    lower = path.lower()
    try:
        if lower.endswith(".npy"):
            with open(path, "rb") as f:
                try:
                    version = np.lib.format.read_magic(f)
                    if version == (1, 0):
                        shape, _, _ = np.lib.format.read_array_header_1_0(f)
                    elif version == (2, 0):
                        shape, _, _ = np.lib.format.read_array_header_2_0(f)
                    else:
                        return None
                    return tuple(int(s) for s in shape)
                except Exception:
                    return None
        if lower.endswith(".nii") or lower.endswith(".nii.gz"):
            try:
                import nibabel as nib

                img = nib.load(path, mmap=True)
                return tuple(int(s) for s in img.shape)
            except Exception:
                return None
        if lower.endswith(".h5") or lower.endswith(".hdf5"):
            try:
                import h5py

                with h5py.File(path, "r") as f:
                    keys = list(f.keys())
                    if len(keys) == 1:
                        return tuple(int(s) for s in f[keys[0]].shape)
            except Exception:
                return None
        if lower.endswith(".zarr"):
            try:
                import json

                zarr_meta = os.path.join(path, ".zarray")
                if os.path.isfile(zarr_meta):
                    with open(zarr_meta) as f:
                        return tuple(int(s) for s in json.load(f).get("shape", []))
            except Exception:
                return None
    except Exception:
        return None
    return None


def _shape_compatible(base: tuple, cand: tuple, mode: str) -> bool:
    """Check whether a candidate shape is compatible with the base shape."""
    if not base or not cand:
        return True
    if mode == "overlay":
        return tuple(base) == tuple(cand)
    if mode == "vectorfield":
        if len(cand) != len(base) + 1:
            return False
        for drop_idx in range(len(cand)):
            if cand[drop_idx] == 3 and tuple(
                cand[:drop_idx] + cand[drop_idx + 1 :]
            ) == tuple(base):
                return True
        return False
    return True
