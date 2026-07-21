"""Staged drag-and-drop imports."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from fastapi import File, Form, HTTPException, Request, UploadFile

from arrayview._session import SESSIONS, Session


DROP_TTL_SECONDS = 15 * 60
_UPLOAD_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _DropSeries:
    selector: str
    uid: str
    series_number: str
    modality: str
    count: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _DropEntry:
    drop_id: str
    root: str
    source: str
    kind: str
    name: str
    shape: tuple[int, ...] | None
    dtype: str | None
    created_at: float
    series: tuple[_DropSeries, ...] = ()


def _drop_state(app):
    if not hasattr(app.state, "drop_imports"):
        app.state.drop_imports = {}
        app.state.drop_imports_lock = threading.Lock()
    return app.state.drop_imports, app.state.drop_imports_lock


def _safe_relative_path(raw: str, fallback: str) -> PurePosixPath:
    value = (raw or fallback).replace("\\", "/")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise HTTPException(status_code=400, detail=f"Unsafe relative path: {value!r}")
    return path


def _display_name(paths: list[PurePosixPath]) -> str:
    if len(paths) == 1 and len(paths[0].parts) == 1:
        return paths[0].name
    roots = {path.parts[0] for path in paths if len(path.parts) > 1}
    return next(iter(roots)) if len(roots) == 1 else "dropped folder"


def _actions(shape: tuple[int, ...] | None, base_sid: str | None) -> dict:
    if shape is None:
        reason = "Select a DICOM series."
        return {
            action: {"enabled": False, "reason": reason}
            for action in ("open", "compare", "overlay")
        }
    actions = {"open": {"enabled": True, "reason": None}}
    base = SESSIONS.get(base_sid) if base_sid else None
    if not base_sid:
        enabled = False
        reason = "No base array is open."
    elif base is None:
        enabled = False
        reason = "Base session not found."
    elif tuple(int(size) for size in base.shape) != shape:
        enabled = False
        reason = f"Requires exact shape {list(base.shape)}."
    else:
        enabled = True
        reason = None
    actions["compare"] = {"enabled": enabled, "reason": reason}
    actions["overlay"] = {"enabled": enabled, "reason": reason}
    return actions


def _load_candidate(entry: _DropEntry, *, select=None):
    from arrayview._io import default_array_key, load_data_with_meta, load_dir_collection

    if entry.kind == "folder":
        pattern = os.path.join(entry.source, "**", "*")
        data, spatial_meta, _overlays, _summary = load_dir_collection([pattern])
        return data, spatial_meta
    key = default_array_key(entry.source)
    return load_data_with_meta(entry.source, key=key, select=select)


def _inspect_candidate(root: str, paths: list[PurePosixPath]):
    from arrayview._dicom import discover_dicom_series, is_dicom_source

    is_folder = len(paths) > 1 or any(len(path.parts) > 1 for path in paths)
    source = root if is_folder else os.path.join(root, *paths[0].parts)
    series = discover_dicom_series(source) if is_dicom_source(source) else []
    if series:
        from arrayview._io import load_data_with_meta

        inspected_series = []
        first_data = None
        for item in series:
            item_data, _item_meta = load_data_with_meta(source, select=item["uid"])
            if first_data is None:
                first_data = item_data
            inspected_series.append(
                _DropSeries(
                    selector=uuid.uuid4().hex[:12],
                    uid=item["uid"],
                    series_number=item["series_number"],
                    modality=item["modality"],
                    count=item["count"],
                    shape=tuple(int(size) for size in item_data.shape),
                    dtype=str(item_data.dtype),
                )
            )
        if len(inspected_series) == 1:
            return "dicom", source, first_data, tuple(inspected_series)
        return "dicom", source, None, tuple(inspected_series)

    kind = "folder" if is_folder else "file"
    provisional = _DropEntry(
        drop_id="",
        root=root,
        source=source,
        kind=kind,
        name="",
        shape=(),
        dtype="",
        created_at=0,
    )
    data, _meta = _load_candidate(provisional)
    return kind, source, data, ()


def _remove_entry(entry: _DropEntry) -> None:
    shutil.rmtree(entry.root, ignore_errors=True)


async def _expire_drop(app, drop_id: str, created_at: float) -> None:
    await asyncio.sleep(DROP_TTL_SECONDS)
    imports, lock = _drop_state(app)
    with lock:
        entry = imports.get(drop_id)
        if entry is None or entry.created_at != created_at:
            return
        imports.pop(drop_id, None)
    await asyncio.to_thread(_remove_entry, entry)


def register_drop_routes(app) -> None:
    @app.post("/drop/inspect")
    async def inspect_drop(
        files: list[UploadFile] = File(...),
        relative_paths: list[str] | None = Form(None),
        base_sid: str | None = Form(None),
    ):
        if not files:
            raise HTTPException(status_code=400, detail="At least one file is required.")
        if relative_paths is not None and len(relative_paths) != len(files):
            raise HTTPException(
                status_code=400,
                detail="relative_paths must contain one value for each file.",
            )

        root = tempfile.mkdtemp(prefix="arrayview-drop-")
        paths: list[PurePosixPath] = []
        seen: set[str] = set()
        try:
            for index, upload in enumerate(files):
                fallback = os.path.basename(upload.filename or f"upload-{index}")
                raw = relative_paths[index] if relative_paths is not None else fallback
                relative = _safe_relative_path(raw, fallback)
                key = relative.as_posix()
                duplicate_key = key.casefold()
                if duplicate_key in seen:
                    raise HTTPException(
                        status_code=400, detail=f"Duplicate relative path: {key!r}"
                    )
                seen.add(duplicate_key)
                target = Path(root).joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as stream:
                    while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
                        stream.write(chunk)
                paths.append(relative)

            kind, source, data, series = await asyncio.to_thread(
                _inspect_candidate, root, paths
            )
            shape = tuple(int(size) for size in data.shape) if data is not None else None
            dtype = str(data.dtype) if data is not None else None
            drop_id = uuid.uuid4().hex
            created_at = time.monotonic()
            entry = _DropEntry(
                drop_id=drop_id,
                root=root,
                source=source,
                kind=kind,
                name=_display_name(paths),
                shape=shape,
                dtype=dtype,
                created_at=created_at,
                series=series,
            )
            imports, lock = _drop_state(app)
            with lock:
                imports[drop_id] = entry
            asyncio.create_task(_expire_drop(app, drop_id, created_at))
            return {
                "drop_id": drop_id,
                "kind": kind,
                "name": entry.name,
                "shape": list(shape) if shape is not None else None,
                "dtype": dtype,
                "series": [
                    {
                        "selector": item.selector,
                        "series_number": item.series_number,
                        "modality": item.modality,
                        "count": item.count,
                        "shape": list(item.shape),
                        "dtype": item.dtype,
                        "actions": _actions(item.shape, base_sid),
                    }
                    for item in series
                ],
                "actions": _actions(shape, base_sid),
            }
        except HTTPException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(root, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/drop/commit")
    async def commit_drop(request: Request):
        body = await request.json()
        drop_id = str(body.get("drop_id") or "")
        action = str(body.get("action") or "")
        base_sid = str(body.get("base_sid") or "") or None
        if action not in {"compare", "open", "overlay"}:
            raise HTTPException(status_code=400, detail="Invalid drop action.")

        imports, lock = _drop_state(app)
        with lock:
            entry = imports.pop(drop_id, None)
        if entry is None:
            raise HTTPException(status_code=404, detail="Drop not found or expired.")

        selected_series = None
        requested_selector = body.get("series")
        if entry.series:
            if requested_selector is None and len(entry.series) > 1:
                with lock:
                    imports[drop_id] = entry
                raise HTTPException(status_code=409, detail="Select a DICOM series.")
            if requested_selector is None:
                selected_series = entry.series[0]
            else:
                selected_series = next(
                    (item for item in entry.series if item.selector == requested_selector),
                    None,
                )
                if selected_series is None:
                    with lock:
                        imports[drop_id] = entry
                    raise HTTPException(status_code=400, detail="Invalid DICOM series selector.")
        selected_shape = (
            selected_series.shape if selected_series is not None else entry.shape
        )
        action_info = _actions(selected_shape, base_sid)[action]
        if not action_info["enabled"]:
            with lock:
                imports[drop_id] = entry
            raise HTTPException(status_code=409, detail=action_info["reason"])

        select = selected_series.uid if selected_series is not None else None
        try:
            data, spatial_meta = await asyncio.to_thread(
                _load_candidate, entry, select=select
            )
            actual_shape = tuple(int(size) for size in data.shape)
            if actual_shape != selected_shape:
                raise ValueError("The staged array shape changed after inspection.")
            session = await asyncio.to_thread(Session, data, name=entry.name)
            if spatial_meta is not None:
                session.spatial_meta = spatial_meta
                session.original_volume = data
            session._drop_staging_dir = entry.root
            SESSIONS[session.sid] = session
        except Exception as exc:
            _remove_entry(entry)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if action == "compare":
            query = f"sid={base_sid}&compare_sid={session.sid}"
        elif action == "overlay":
            query = f"sid={base_sid}&overlay_sid={session.sid}"
        else:
            query = f"sid={session.sid}"
        return {
            "sid": session.sid,
            "base_sid": base_sid,
            "action": action,
            "name": entry.name,
            "shape": list(actual_shape),
            "dtype": str(data.dtype),
            "url": "/?" + urllib.parse.quote(query, safe="=&,"),
        }

    @app.delete("/drop/{drop_id}")
    async def delete_drop(drop_id: str):
        imports, lock = _drop_state(app)
        with lock:
            entry = imports.pop(drop_id, None)
        if entry is None:
            raise HTTPException(status_code=404, detail="Drop not found or expired.")
        await asyncio.to_thread(_remove_entry, entry)
        return {"drop_id": drop_id, "deleted": True}
