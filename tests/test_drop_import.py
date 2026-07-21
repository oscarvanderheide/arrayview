import base64
import io
import os
import time

import numpy as np
import pytest
from playwright.sync_api import expect
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from arrayview._session import SESSIONS


def _npy_upload(name, array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return ("files", (name, buffer.getvalue(), "application/octet-stream"))


def _dicom_bytes(tmp_path, *, series_uid=None, filename="slice.dcm", value=1):
    path = tmp_path / filename
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.Modality = "MR"
    ds.SeriesNumber = 3
    ds.InstanceNumber = 1
    ds.Rows = 2
    ds.Columns = 3
    ds.PixelSpacing = [1, 1]
    ds.SliceThickness = 2
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, 0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.full((2, 3), value, dtype=np.uint16).tobytes()
    ds.save_as(path)
    return path.read_bytes()


def test_inspect_then_commit_creates_session_only_on_commit(client, sid_2d):
    before = set(SESSIONS)
    array = np.zeros(SESSIONS[sid_2d].shape, dtype=np.float32)
    inspected = client.post(
        "/drop/inspect",
        files=[_npy_upload("candidate.npy", array)],
        data={"base_sid": sid_2d},
    )
    inspected.raise_for_status()
    body = inspected.json()

    assert set(SESSIONS) == before
    assert body["kind"] == "file"
    assert body["shape"] == list(array.shape)
    assert body["dtype"] == "float32"
    assert body["actions"] == {
        "open": {"enabled": True, "reason": None},
        "compare": {"enabled": True, "reason": None},
        "overlay": {"enabled": True, "reason": None},
    }

    committed = client.post(
        "/drop/commit",
        json={"drop_id": body["drop_id"], "base_sid": sid_2d, "action": "compare"},
    )
    committed.raise_for_status()
    result = committed.json()
    assert result["sid"] in SESSIONS
    assert result["url"] == f"/?sid={sid_2d}&compare_sid={result['sid']}"

    staging_dir = SESSIONS[result["sid"]]._drop_staging_dir
    assert os.path.isdir(staging_dir)
    client.post(f"/release/{result['sid']}").raise_for_status()
    assert not os.path.exists(staging_dir)


def test_shape_mismatch_disables_compare_and_overlay(client, sid_2d):
    inspected = client.post(
        "/drop/inspect",
        files=[_npy_upload("different.npy", np.zeros((3, 4), dtype=np.uint8))],
        data={"base_sid": sid_2d},
    ).json()

    assert inspected["actions"]["open"]["enabled"] is True
    assert inspected["actions"]["compare"]["enabled"] is False
    assert inspected["actions"]["overlay"]["enabled"] is False
    assert "exact shape" in inspected["actions"]["compare"]["reason"]

    rejected = client.post(
        "/drop/commit",
        json={
            "drop_id": inspected["drop_id"],
            "base_sid": sid_2d,
            "action": "overlay",
        },
    )
    assert rejected.status_code == 409
    client.delete(f"/drop/{inspected['drop_id']}").raise_for_status()


def test_inspect_rejects_traversal_and_duplicate_paths(client):
    files = [
        _npy_upload("a.npy", np.zeros((2, 2))),
        _npy_upload("b.npy", np.zeros((2, 2))),
    ]
    traversal = client.post(
        "/drop/inspect",
        files=files[:1],
        data={"relative_paths": "../escape.npy"},
    )
    assert traversal.status_code == 400
    assert "Unsafe relative path" in traversal.json()["detail"]

    duplicate = client.post(
        "/drop/inspect",
        files=[
            *files,
            ("relative_paths", (None, "folder/same.npy")),
            ("relative_paths", (None, "folder/same.npy")),
        ],
    )
    assert duplicate.status_code == 400
    assert "Duplicate relative path" in duplicate.json()["detail"]


def test_folder_is_inspected_as_stack_candidate(client):
    files = [
        _npy_upload("a.npy", np.zeros((2, 3), dtype=np.float32)),
        _npy_upload("b.npy", np.ones((2, 3), dtype=np.float32)),
    ]
    response = client.post(
        "/drop/inspect",
        files=[
            *files,
            ("relative_paths", (None, "study/a.npy")),
            ("relative_paths", (None, "study/b.npy")),
        ],
    )
    response.raise_for_status()
    body = response.json()
    assert body["kind"] == "folder"
    assert body["name"] == "study"
    assert body["shape"] == [2, 3, 2]
    client.delete(f"/drop/{body['drop_id']}").raise_for_status()


def test_dicom_folder_is_classified_and_committed(client, tmp_path):
    response = client.post(
        "/drop/inspect",
        files=[("files", ("slice.dcm", _dicom_bytes(tmp_path), "application/dicom"))],
        data={"relative_paths": "series/slice.dcm"},
    )
    response.raise_for_status()
    body = response.json()
    assert body["kind"] == "dicom"
    assert body["shape"] == [3, 2, 1]
    assert body["series"][0]["modality"] == "MR"
    assert set(body["series"][0]) == {
        "selector",
        "series_number",
        "modality",
        "count",
        "shape",
        "dtype",
        "actions",
    }

    committed = client.post(
        "/drop/commit",
        json={"drop_id": body["drop_id"], "action": "open"},
    )
    committed.raise_for_status()
    sid = committed.json()["sid"]
    assert SESSIONS[sid].spatial_meta["dicom_meta"]
    client.post(f"/release/{sid}").raise_for_status()


def test_multi_series_dicom_requires_opaque_selection(client, tmp_path):
    first_uid = generate_uid()
    second_uid = generate_uid()
    files = [
        (
            "files",
            (
                "first.dcm",
                _dicom_bytes(
                    tmp_path, series_uid=first_uid, filename="first.dcm", value=11
                ),
                "application/dicom",
            ),
        ),
        (
            "files",
            (
                "second.dcm",
                _dicom_bytes(
                    tmp_path, series_uid=second_uid, filename="second.dcm", value=22
                ),
                "application/dicom",
            ),
        ),
        ("relative_paths", (None, "series/first.dcm")),
        ("relative_paths", (None, "series/second.dcm")),
    ]
    body = client.post("/drop/inspect", files=files).json()

    assert body["kind"] == "dicom"
    assert body["shape"] is None
    assert body["dtype"] is None
    assert len(body["series"]) == 2
    payload = str(body)
    assert first_uid not in payload
    assert second_uid not in payload

    missing = client.post(
        "/drop/commit", json={"drop_id": body["drop_id"], "action": "open"}
    )
    assert missing.status_code == 409
    selected = client.post(
        "/drop/commit",
        json={
            "drop_id": body["drop_id"],
            "action": "open",
            "series": body["series"][1]["selector"],
        },
    )
    selected.raise_for_status()
    sid = selected.json()["sid"]
    assert int(np.asarray(SESSIONS[sid].data).mean()) in {11, 22}
    client.post(f"/release/{sid}").raise_for_status()


def test_drop_expires_and_cleans_tempdir(client, monkeypatch):
    import arrayview._routes_drop as drop_routes

    monkeypatch.setattr(drop_routes, "DROP_TTL_SECONDS", 0.01)
    body = client.post(
        "/drop/inspect",
        files=[_npy_upload("short-lived.npy", np.zeros((2, 2)))],
    ).json()
    time.sleep(0.08)
    assert client.delete(f"/drop/{body['drop_id']}").status_code == 404


@pytest.mark.browser
def test_file_drop_waits_for_explicit_menu_action(loaded_viewer, sid_2d):
    before = set(SESSIONS)
    buffer = io.BytesIO()
    np.save(buffer, np.zeros(SESSIONS[sid_2d].shape, dtype=np.float32))
    encoded = base64.b64encode(buffer.getvalue()).decode()
    page = loaded_viewer(sid_2d)

    page.evaluate(
        """({encoded}) => {
            const bytes = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
            const transfer = new DataTransfer();
            transfer.items.add(new File([bytes], 'candidate.npy', {type: 'application/octet-stream'}));
            document.body.dispatchEvent(new DragEvent('drop', {
                bubbles: true,
                cancelable: true,
                clientX: 420,
                clientY: 260,
                dataTransfer: transfer,
            }));
        }""",
        {"encoded": encoded},
    )

    menu = page.locator("#drop-action-menu")
    expect(menu).to_have_class("visible")
    expect(menu).to_contain_text("Compare side by side")
    expect(menu).to_contain_text("Open separately")
    expect(menu).to_contain_text("Add as overlay")
    assert set(SESSIONS) == before

    page.keyboard.press("Escape")
    expect(menu).not_to_have_class("visible")


@pytest.mark.browser
def test_compatible_drop_can_enter_compare(loaded_viewer, sid_2d):
    buffer = io.BytesIO()
    np.save(buffer, np.ones(SESSIONS[sid_2d].shape, dtype=np.float32))
    encoded = base64.b64encode(buffer.getvalue()).decode()
    page = loaded_viewer(sid_2d)
    page.evaluate(
        """({encoded}) => {
            const bytes = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
            const transfer = new DataTransfer();
            transfer.items.add(new File([bytes], 'compare.npy', {type: 'application/octet-stream'}));
            document.body.dispatchEvent(new DragEvent('drop', {
                bubbles: true, cancelable: true, clientX: 420, clientY: 260,
                dataTransfer: transfer,
            }));
        }""",
        {"encoded": encoded},
    )

    page.get_by_role("button", name="Compare side by side").click()
    expect(page.locator("#compare-view-wrap")).to_be_visible()
    expect(page.locator("#drop-action-menu")).not_to_have_class("visible")


@pytest.mark.browser
def test_compatible_drop_can_attach_overlay(loaded_viewer, sid_2d):
    buffer = io.BytesIO()
    np.save(buffer, np.ones(SESSIONS[sid_2d].shape, dtype=np.uint8))
    encoded = base64.b64encode(buffer.getvalue()).decode()
    page = loaded_viewer(sid_2d)
    page.evaluate(
        """({encoded}) => {
            const bytes = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
            const transfer = new DataTransfer();
            transfer.items.add(new File([bytes], 'mask.npy', {type: 'application/octet-stream'}));
            document.body.dispatchEvent(new DragEvent('drop', {
                bubbles: true, cancelable: true, clientX: 420, clientY: 260,
                dataTransfer: transfer,
            }));
        }""",
        {"encoded": encoded},
    )

    page.get_by_role("button", name="Add as overlay").click()
    expect(page.locator("#overlay-palette")).to_have_class("visible")
    expect(page.locator("#overlay-palette")).to_contain_text("mask.npy")
