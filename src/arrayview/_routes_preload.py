import threading

from fastapi import Request

from arrayview._render import _run_preload
from arrayview._session import SESSIONS


def register_preload_routes(app) -> None:
    @app.post("/preload/{sid}")
    async def start_preload(sid: str, request: Request):
        session = SESSIONS.get(sid)
        if not session:
            return {"error": "Invalid session"}

        body = await request.json()
        dim_x = int(body["dim_x"])
        dim_y = int(body["dim_y"])
        idx_list = [int(x) for x in body["indices"]]
        colormap = str(body.get("colormap", "gray"))
        dr = int(body.get("dr", 1))
        slice_dim = int(body["slice_dim"])
        dim_z = int(body.get("dim_z", -1))
        complex_mode = int(body.get("complex_mode", 0))
        log_scale = bool(body.get("log_scale", False))

        session.preload_gen += 1
        gen = session.preload_gen
        threading.Thread(
            target=_run_preload,
            args=(
                session,
                gen,
                dim_x,
                dim_y,
                idx_list,
                colormap,
                dr,
                slice_dim,
                dim_z,
                complex_mode,
                log_scale,
            ),
            daemon=True,
        ).start()
        return {"status": "started"}

    @app.get("/preload_status/{sid}")
    def get_preload_status(sid: str):
        session = SESSIONS.get(sid)
        if not session:
            return {"error": "Invalid session"}
        with session.preload_lock:
            return {
                "done": session.preload_done,
                "total": session.preload_total,
                "skipped": session.preload_skipped,
            }