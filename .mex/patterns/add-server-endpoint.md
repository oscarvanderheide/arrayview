---
name: add-server-endpoint
description: Adding a new REST or WebSocket route to the FastAPI server (_server.py) — includes session lookup, render dispatch, and stdio transport parity.
triggers:
  - "endpoint"
  - "route"
  - "FastAPI"
  - "_server.py"
  - "WebSocket"
  - "API"
  - "HTTP"
edges:
  - target: context/architecture.md
    condition: when understanding how _server.py fits in the system
  - target: context/render-pipeline.md
    condition: when the endpoint triggers a render (slice, mosaic, projection)
  - target: context/conventions.md
    condition: for lazy import rules and Verify Checklist
last_updated: 2026-04-15
---

# Add Server Endpoint

## Context

`_server.py` is the FastAPI app. All routes live here. Session lookup is the first thing every route does: `session = SESSIONS.get(sid)` with a 404 if missing.

Heavy render work must go to the render thread via `_render()` from `_session.py` — never block the async event loop with CPU work.

If the feature involves the VS Code direct webview (stdio transport), `_stdio_server.py` must also be updated to handle the new message type. The two servers must stay feature-equivalent.

## Steps

1. **Add the route to `_server.py`:**
   ```python
   @app.get("/my_route/{sid}")
   async def my_route(sid: str, request: Request):
       session = SESSIONS.get(sid)
       if session is None:
           raise HTTPException(status_code=404, detail="Session not found")
       # ... logic ...
       return JSONResponse({"result": ...})
   ```

2. **For render work**, dispatch via the render thread:
   ```python
   loop = asyncio.get_event_loop()
   result = await _render(loop, lambda: my_render_func(session, params))
   ```
   Never call `extract_slice()` or `render_rgba()` directly on the async path.

3. **For WebSocket routes**, follow the existing `/ws/{sid}` pattern:
   - Drain messages in a loop
   - Send binary frames for image data, JSON for metadata
   - Handle `WebSocketDisconnect` gracefully — decrement `_session_mod.VIEWER_SOCKETS`
   - The binary frame format (RGBA bytes) must match what `_viewer.html` expects

4. **For file upload routes**, use `UploadFile` from FastAPI — already imported in `_server.py`.

5. **Wire the frontend:** If the route is called by JavaScript, add the fetch/WebSocket call in the correct section of `_viewer.html`. Follow the dual-write pattern if it updates display state.

6. **Update `_stdio_server.py`** if the feature is needed in VS Code direct webview mode. The stdio server handles messages as JSON objects on stdin — add a new `elif msg_type == "my_type":` branch in the message dispatch loop.

## Gotchas

- **Do not add logic to `_app.py`** — `_app.py` is a backward-compat shim only. All new routes go in `_server.py`.
- **Session lookup first** — every route must validate `sid` before doing anything. A missing session that falls through silently causes `AttributeError` on `session.data`.
- **WebSocket binary protocol is tightly coupled** — the byte layout (offset, header fields, RGBA payload) is shared between `_server.py` and the WS handler in `_viewer.html`. Change one → change both. Mismatch causes the canvas to render garbage or stay blank.
- **Never block the event loop** — even a small `np.array()` call on a large array can block for hundreds of milliseconds. Use `await _render(loop, func)` for all numpy work.
- **`VIEWER_SOCKETS` is a module-level integer in `_session.py`** — increment/decrement via `_session_mod.VIEWER_SOCKETS` (the module reference), not the locally imported name. Same for `VIEWER_SIDS`.
- **`_stdio_server.py` parity** — if your endpoint returns data that the viewer needs in any environment, it must work in the stdio transport too. VS Code tunnel users will silently get a broken feature otherwise.

## Verify

- [ ] Route validates `sid` and returns 404 for unknown sessions
- [ ] CPU/render work dispatched via `await _render(loop, ...)`, not called directly
- [ ] WebSocket binary frame layout matches `_viewer.html` expectations (if applicable)
- [ ] No new logic added to `_app.py`
- [ ] `_stdio_server.py` updated if the feature is needed in VS Code direct webview
- [ ] `uv run pytest tests/test_view_component_integration.py` passes
- [ ] Manual test: `uv run arrayview dev/sample.npy` — new route reachable and returns expected response

## Update Scaffold
- [ ] Update `.mex/ROUTER.md` "Current Project State" if what's working/not built has changed
- [ ] Update any `.mex/context/` files that are now out of date
- [ ] If this is a new task type without a pattern, create one in `.mex/patterns/` and add to `INDEX.md`
