# Instance Previewer Sidebar & RAM Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a visual instance previewer sidebar to the unified picker and a RAM guard that blocks loading full-load formats when available memory is insufficient.

**Architecture:** The picker modal (`#uni-picker-box`) gains a left sidebar panel showing thumbnail+metadata cards for each open session. Thumbnails are fetched via a new `GET /thumbnail/{sid}` endpoint that renders a small JPEG using the session's default view (first two dims, middle slice, gray colormap). The RAM guard adds a pre-load check in `POST /load` that returns HTTP 507 for `.pt`/`.tif`/`.mat` files exceeding available RAM.

**Tech Stack:** Python (FastAPI), vanilla JS, HTML/CSS. `psutil` for RAM checks (already a dependency).

---

### Task 1: RAM Guard — Backend (`_server.py`, `_io.py`)

**Files:**
- Modify: `src/arrayview/_io.py` (add `FULL_LOAD_EXTS` set after `_SUPPORTED_EXTS`, ~line 153)
- Modify: `src/arrayview/_server.py:2071-2106` (add RAM check in `load_file`)

- [ ] **Step 1: Add format classification to `_io.py`**

Add after the `_SUPPORTED_EXTS` frozenset (line 153):

```python
# Formats that load the entire array into RAM (no mmap/lazy access).
# RAM guard only applies to these.
FULL_LOAD_EXTS = frozenset([".pt", ".pth", ".tif", ".tiff", ".mat"])
```

- [ ] **Step 2: Add RAM check to `POST /load` in `_server.py`**

Add this block in `load_file()` right after the dedup check (after the `for existing in SESSIONS` loop, before the `try: data = await ...` line):

```python
    # RAM guard: block full-load formats that would exceed available memory.
    if not os.environ.get("ARRAYVIEW_SKIP_RAM_GUARD"):
        from ._io import FULL_LOAD_EXTS
        ext = os.path.splitext(filepath)[1].lower()
        # Handle double extensions like .nii.gz (not relevant here but consistent)
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
                pass  # psutil not available — skip guard
```

Note: `JSONResponse` is already imported from `fastapi.responses` (line 24).

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `cd /Users/oscar/Projects/packages/python/arrayview && python -m pytest tests/test_api.py -v --timeout=30`
Expected: All pass (RAM guard doesn't trigger for test arrays which are tiny).

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_io.py src/arrayview/_server.py
git commit -m "feat: RAM guard — block full-load formats exceeding available memory"
```

---

### Task 2: RAM Guard — Frontend Dialog (`_viewer.html`)

**Files:**
- Modify: `src/arrayview/_viewer.html` (CSS ~line 331, and `showUnifiedPicker` function ~line 8657)

- [ ] **Step 1: Add CSS for RAM block dialog**

Add after the `.cp-item-cb.checked` block (around line 331):

```css
.ram-block-overlay { position: absolute; inset: 0; background: rgba(0,0,0,0.7); display: flex;
    align-items: center; justify-content: center; border-radius: var(--radius-lg); z-index: 5; }
.ram-block-box { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: 24px; text-align: center; max-width: 320px; }
.ram-block-icon { font-size: 28px; margin-bottom: 8px; }
.ram-block-title { font-size: 13px; color: #f87171; font-weight: 600; }
.ram-block-detail { font-size: 11px; color: var(--muted); margin-top: 10px; line-height: 1.8; font-family: monospace; }
.ram-block-detail strong { color: var(--text); }
.ram-block-hint { margin-top: 12px; padding: 8px 12px; background: var(--surface-2); border-radius: 4px;
    font-size: 9px; color: var(--muted); }
.ram-block-ok { margin-top: 14px; padding: 5px 20px; background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 3px; font-size: 10px; color: var(--text); cursor: pointer; }
.ram-block-ok:hover { border-color: var(--picker-accent, var(--active-dim)); }
```

- [ ] **Step 2: Add RAM block dialog helper function**

Add this function just before `showUnifiedPicker` in `_viewer.html`:

```javascript
function _showRamBlockDialog(parentEl, filename, estimatedBytes, availableBytes) {
    const fmt = (b) => {
        if (b >= 1024 ** 3) return (b / 1024 ** 3).toFixed(1) + ' GB';
        return (b / 1024 ** 2).toFixed(0) + ' MB';
    };
    const overlay = document.createElement('div');
    overlay.className = 'ram-block-overlay';
    overlay.innerHTML = `<div class="ram-block-box">
        <div class="ram-block-icon">⚠</div>
        <div class="ram-block-title">Not enough memory</div>
        <div class="ram-block-detail">
            <strong>${filename}</strong> requires ~${fmt(estimatedBytes)}<br>
            Available RAM: ${fmt(availableBytes)}
        </div>
        <div class="ram-block-hint">Close other instances or free memory to continue</div>
        <button class="ram-block-ok">OK</button>
    </div>`;
    parentEl.appendChild(overlay);
    const btn = overlay.querySelector('.ram-block-ok');
    btn.focus();
    return new Promise(resolve => {
        function dismiss() { overlay.remove(); resolve(); }
        btn.onclick = dismiss;
        overlay.onkeydown = e => {
            if (e.key === 'Escape' || e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); dismiss(); }
        };
    });
}
```

- [ ] **Step 3: Handle `insufficient_memory` error in file-open flow**

In `showUnifiedPicker`, find the two places where `fetch('/load', ...)` is called and the response is checked for `data.error`. There are two: one in `toggleSel` for files (the compare-select path) and one in `openFile` (the direct-open path).

In both places, replace:
```javascript
if (data.error) { showToast(`load error: ${data.error}`); item.classList.remove('cp-item-loading'); return; }
```

with:
```javascript
if (data.error === 'insufficient_memory') {
    item.classList.remove('cp-item-loading');
    await _showRamBlockDialog(
        document.getElementById('uni-picker-box'),
        data.filename, data.estimated_bytes, data.available_bytes
    );
    return;
}
if (data.error) { showToast(`load error: ${data.error}`); item.classList.remove('cp-item-loading'); return; }
```

**Important:** The `toggleSel` and `openFile` inner functions that call `fetch('/load')` use `.then()` chains. Convert the `.then()` callback to `async` so `await` works:

For `toggleSel` in the files section, change:
```javascript
}).then(r => r.json()).then(data => {
```
to:
```javascript
}).then(r => r.json()).then(async data => {
```

Same for `openFile`:
```javascript
}).then(r => r.json()).then(data => {
```
to:
```javascript
}).then(r => r.json()).then(async data => {
```

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: RAM guard frontend — block dialog for insufficient memory"
```

---

### Task 3: Thumbnail Endpoint (`_server.py`)

**Files:**
- Modify: `src/arrayview/_server.py` (add `GET /thumbnail/{sid}` endpoint after `/sessions`)

- [ ] **Step 1: Add thumbnail endpoint**

Add after the `get_sessions` function (after line ~2068):

```python
@app.get("/thumbnail/{sid}")
async def get_thumbnail(sid: str, w: int = 96, h: int = 72):
    """Return a small JPEG thumbnail of the session's current default view."""
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)

    ndim = len(session.shape)
    # Default view: last two dims, middle index for all others
    dim_x = ndim - 1
    dim_y = ndim - 2 if ndim >= 2 else 0
    idx_list = [s // 2 for s in session.shape]

    try:
        rgba = await asyncio.to_thread(
            render_rgba, session, dim_x, dim_y, tuple(idx_list),
            "gray", 1, 0, False, None, None,
        )
    except Exception:
        # Fallback: return a 1x1 gray pixel
        rgba = np.full((1, 1, 4), 128, dtype=np.uint8)

    # Resize to thumbnail dimensions
    Image = _pil_image()
    img = Image.fromarray(rgba[:, :, :3])
    img = img.resize((w, h), Image.NEAREST if max(rgba.shape[:2]) < h else Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    return Response(content=buf.getvalue(), media_type="image/jpeg")
```

- [ ] **Step 2: Verify imports**

`io` is already imported (line 7). PIL is lazy-imported via `_pil_image()` helper (line 184). Both are used in the endpoint code above.

- [ ] **Step 3: Quick manual test**

Start arrayview with a test array and hit the endpoint:
```bash
python -c "import numpy as np; import arrayview; arrayview.view(np.random.rand(100,100))" &
sleep 2
curl -s http://localhost:*/thumbnail/* -o /tmp/thumb.jpg
```

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_server.py
git commit -m "feat: GET /thumbnail/{sid} — small JPEG preview for picker sidebar"
```

---

### Task 4: Extend `/sessions` Response with Metadata (`_server.py`, `_session.py`)

**Files:**
- Modify: `src/arrayview/_server.py:2057-2068` (extend `get_sessions` response)
- Modify: `src/arrayview/_session.py` (add `estimated_memory_bytes` property)

- [ ] **Step 1: Add memory estimation to Session**

Add after `self.MOSAIC_CACHE_BYTES = _MOSAIC_CACHE_BYTES` in `__init__` (line 184):

```python
        self._estimated_mem = self._estimate_memory()
```

Add this method to the Session class (after `__init__`):

```python
    def _estimate_memory(self):
        """Estimate memory footprint in bytes (array data + cache budgets)."""
        itemsize = np.dtype(getattr(self.data, "dtype", np.float32)).itemsize
        data_bytes = int(np.prod(self.shape)) * itemsize
        return data_bytes
```

- [ ] **Step 2: Extend `/sessions` endpoint response**

Replace the `get_sessions` function body:

```python
@app.get("/sessions")
def get_sessions():
    """Returns list of active sessions with metadata for the picker sidebar."""
    result = []
    for s in SESSIONS.values():
        dtype_str = str(getattr(s.data, "dtype", "unknown"))
        result.append({
            "sid": s.sid,
            "name": s.name,
            "shape": [int(x) for x in s.shape],
            "filepath": s.filepath,
            "dtype": dtype_str,
            "estimated_mem": s._estimated_mem,
        })
    return result
```

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_server.py src/arrayview/_session.py
git commit -m "feat: extend /sessions with dtype and memory estimate for sidebar"
```

---

### Task 5: Instance Sidebar — HTML Structure (`_viewer.html`)

**Files:**
- Modify: `src/arrayview/_viewer.html` (HTML markup ~line 1192, CSS ~line 286)

- [ ] **Step 1: Update picker HTML markup**

Replace the `#uni-picker-box` div (lines 1193-1201):

```html
        <div id="uni-picker-box">
            <div id="uni-picker-sidebar"></div>
            <div id="uni-picker-main">
                <div id="uni-picker-header">
                    <span id="uni-picker-mode-pill"></span>
                    <span id="uni-picker-tab-hint">Space to select &middot; Enter to open</span>
                </div>
                <input id="uni-picker-search" type="text" placeholder="filter…" autocomplete="off" spellcheck="false">
                <div id="uni-picker-list"></div>
                <button id="uni-picker-cancel">Cancel</button>
            </div>
        </div>
```

- [ ] **Step 2: Add sidebar CSS**

Add after the existing `#uni-picker-box` rule (line ~291), replacing it:

```css
#uni-picker-box { background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); min-width: 360px; max-width: 720px; width: 60vw;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px;
    display: flex; overflow: hidden; position: relative; }
#uni-picker-box.no-sidebar { max-width: 520px; width: 50vw; }
#uni-picker-sidebar { width: 200px; background: var(--surface-2); border-right: 1px solid var(--border);
    display: flex; flex-direction: column; overflow-y: auto; flex-shrink: 0; }
#uni-picker-sidebar:empty { display: none; }
#uni-picker-main { flex: 1; padding: 16px 20px; display: flex; flex-direction: column; min-width: 0; }
.sidebar-header { padding: 10px 12px 6px; font-size: 10px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
.sidebar-card { display: flex; gap: 8px; padding: 8px 12px; cursor: pointer;
    border-left: 2px solid transparent; }
.sidebar-card:hover { background: var(--surface); }
.sidebar-card.current { background: var(--surface); border-left-color: var(--active-dim); }
.sidebar-card-thumb { width: 48px; height: 36px; border-radius: 3px; overflow: hidden;
    flex-shrink: 0; background: var(--surface); }
.sidebar-card-thumb img { width: 100%; height: 100%; object-fit: cover; }
.sidebar-card-info { min-width: 0; flex: 1; }
.sidebar-card-name { font-size: 10px; color: var(--text); white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; font-weight: 600; }
.sidebar-card-meta { font-size: 8px; color: var(--muted); margin-top: 1px; }
.sidebar-card-current { font-size: 7px; color: var(--active-dim); margin-top: 2px; }
.sidebar-footer { padding: 8px 12px; font-size: 8px; color: var(--muted);
    border-top: 1px solid var(--border); margin-top: auto; }
```

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: picker sidebar HTML structure and CSS"
```

---

### Task 6: Instance Sidebar — JS Logic (`_viewer.html`)

**Files:**
- Modify: `src/arrayview/_viewer.html` (`showUnifiedPicker` function and `_fetchPickerData`)

- [ ] **Step 1: Add sidebar population to `showUnifiedPicker`**

At the top of `showUnifiedPicker`, after the existing element lookups (after `const searchEl = ...`), add:

```javascript
        const sidebarEl = document.getElementById('uni-picker-sidebar');
        const pickerBox = document.getElementById('uni-picker-box');
```

After `searchEl.focus();` and before `populate();`, add the sidebar population logic:

```javascript
        // Populate sidebar with instance previews
        sidebarEl.innerHTML = '';
        const otherSessions = sessions.filter(s => sessions.length > 1);
        if (otherSessions.length > 1) {
            pickerBox.classList.remove('no-sidebar');
            const hdr = document.createElement('div');
            hdr.className = 'sidebar-header';
            hdr.textContent = 'Open Instances';
            sidebarEl.appendChild(hdr);

            let totalMem = 0;
            for (const s of sessions) {
                totalMem += s.estimated_mem || 0;
                const card = document.createElement('div');
                card.className = 'sidebar-card' + (s.sid === sid ? ' current' : '');
                card.tabIndex = 0;

                const thumb = document.createElement('div');
                thumb.className = 'sidebar-card-thumb';
                const img = document.createElement('img');
                img.src = `/thumbnail/${s.sid}?w=96&h=72`;
                img.loading = 'lazy';
                img.alt = s.name;
                thumb.appendChild(img);

                const info = document.createElement('div');
                info.className = 'sidebar-card-info';
                const nameEl = document.createElement('div');
                nameEl.className = 'sidebar-card-name';
                nameEl.textContent = s.name || '(unnamed)';
                info.appendChild(nameEl);

                const shapeStr = Array.isArray(s.shape) ? `(${s.shape.join(', ')})` : '?';
                const meta1 = document.createElement('div');
                meta1.className = 'sidebar-card-meta';
                meta1.textContent = shapeStr;
                info.appendChild(meta1);

                const dtypeSlice = s.dtype || '';
                const meta2 = document.createElement('div');
                meta2.className = 'sidebar-card-meta';
                meta2.textContent = dtypeSlice;
                info.appendChild(meta2);

                if (s.sid === sid) {
                    const cur = document.createElement('div');
                    cur.className = 'sidebar-card-current';
                    cur.textContent = '(current)';
                    info.appendChild(cur);
                }

                card.appendChild(thumb);
                card.appendChild(info);

                card.onclick = () => {
                    if (s.sid === sid) return; // already here
                    finish({ action: 'open', sids: [s.sid], names: [s.name || ''] });
                    if (window.parent !== window && window.parent.__arrayviewShell) {
                        window.parent.activateTab(s.sid);
                    } else {
                        window.location.href = '/?sid=' + s.sid;
                    }
                };
                card.onkeydown = e => {
                    if (e.key === 'Enter') { e.preventDefault(); card.onclick(); }
                };
                sidebarEl.appendChild(card);
            }

            const footer = document.createElement('div');
            footer.className = 'sidebar-footer';
            const memStr = totalMem >= 1024 ** 3
                ? (totalMem / 1024 ** 3).toFixed(1) + ' GB'
                : (totalMem / 1024 ** 2).toFixed(0) + ' MB';
            footer.textContent = `${sessions.length} instances · ${memStr} used`;
            sidebarEl.appendChild(footer);
        } else {
            pickerBox.classList.add('no-sidebar');
            sidebarEl.innerHTML = '';
        }
```

- [ ] **Step 2: Add keyboard navigation for sidebar**

In the `overlay.onkeydown` handler, add Tab key handling to cycle focus between search, sidebar, and file list. Add this at the top of the handler (before the `Escape` check):

```javascript
            if (e.key === 'Tab') {
                e.preventDefault();
                const sidebarCards = [...sidebarEl.querySelectorAll('.sidebar-card')];
                const fileItems = [...listEl.querySelectorAll('.cp-item')];
                const inSidebar = sidebarCards.includes(document.activeElement);
                const inFiles = fileItems.includes(document.activeElement);
                const inSearch = document.activeElement === searchEl;

                if (e.shiftKey) {
                    // Reverse: files → sidebar → search
                    if (inFiles && sidebarCards.length) sidebarCards[0].focus();
                    else if (inSidebar) searchEl.focus();
                    else if (inSearch && fileItems.length) fileItems[0].focus();
                } else {
                    // Forward: search → sidebar → files
                    if (inSearch && sidebarCards.length) sidebarCards[0].focus();
                    else if (inSidebar && fileItems.length) fileItems[0].focus();
                    else searchEl.focus();
                }
                return;
            }
```

- [ ] **Step 3: Update search filtering to include sidebar**

In the `searchEl.oninput` handler, after populating the file list, also filter sidebar cards:

```javascript
        searchEl.oninput = () => {
            if (_searchTimer) clearTimeout(_searchTimer);
            _searchTimer = setTimeout(() => {
                _searchTimer = null;
                const q = searchEl.value.trim().toLowerCase();
                const filtSessions = q ? sessions.filter(s => (s.name || '').toLowerCase().includes(q)) : sessions;
                const filtFiles    = q ? fileEntries.filter(f => f.name.toLowerCase().includes(q)) : fileEntries;
                populate(filtSessions, filtFiles);
                // Filter sidebar cards too
                sidebarEl.querySelectorAll('.sidebar-card').forEach(card => {
                    const name = card.querySelector('.sidebar-card-name')?.textContent?.toLowerCase() || '';
                    card.style.display = (!q || name.includes(q)) ? '' : 'none';
                });
            }, 100);
        };
```

This replaces the existing `searchEl.oninput` handler.

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: instance sidebar JS — thumbnails, click-to-switch, keyboard nav"
```

---

### Task 7: Collateral — Update Help Overlay and README

**Files:**
- Modify: `src/arrayview/_viewer.html` (help overlay, search for `#help-overlay`)
- Modify: `README.md` (if picker/keyboard shortcuts are documented there)

- [ ] **Step 1: Check help overlay for picker documentation**

Search `_viewer.html` for the help overlay content. If the picker shortcut (Cmd/Ctrl/Shift+O) is listed, add a note about the instance sidebar. If there's mention of loaded arrays in the picker description, update it.

- [ ] **Step 2: Update help overlay**

Add to the picker shortcut row: "Opens picker with instance sidebar (when multiple arrays are loaded)"

- [ ] **Step 3: Check README for picker docs**

If `README.md` mentions the picker, add a brief note about the instance sidebar and RAM guard.

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html README.md
git commit -m "docs: update help overlay and README for instance sidebar & RAM guard"
```

---

### Task 8: Integration Smoke Test

**Files:**
- No new files — manual verification

- [ ] **Step 1: Start arrayview with multiple sessions**

```bash
python -c "
import numpy as np
import arrayview
# Open two arrays to trigger sidebar
a = np.random.rand(100, 100)
b = np.random.rand(50, 50, 30).astype(np.float32)
arrayview.view(a, name='test_2d')
arrayview.view(b, name='test_3d')
"
```

- [ ] **Step 2: Verify sidebar appears**

Open the picker (Cmd/Ctrl/Shift+O). Confirm:
- Sidebar shows on the left with two instance cards
- Each card has a thumbnail, name, shape, dtype
- Current instance has yellow border and "(current)" label
- Footer shows "2 instances · X MB used"

- [ ] **Step 3: Verify click-to-switch**

Click the non-current instance card. Confirm the picker closes and the view switches.

- [ ] **Step 4: Verify RAM guard**

Create a large `.pt` file and try to load it when RAM is low (or temporarily mock `psutil.virtual_memory().available` to a small value). Confirm the block dialog appears.

- [ ] **Step 5: Verify single-session behavior**

With only one session open, confirm the picker renders at its original width with no sidebar.

- [ ] **Step 6: Final commit if any fixes needed**

```bash
git add -u
git commit -m "fix: integration fixes for instance sidebar & RAM guard"
```
