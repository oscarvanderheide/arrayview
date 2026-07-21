import json

import numpy as np
import pytest
from playwright.sync_api import expect
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from arrayview._dicom import discover_dicom_series, load_dicom_series
from arrayview._io import load_data_with_meta


def _write_slice(
    path,
    *,
    series_uid,
    z,
    value,
    instance,
    rows=2,
    cols=3,
    orientation=(1, 0, 0, 0, 1, 0),
    slope=1,
    intercept=0,
    patient_name="PRIVATE^PERSON",
):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = generate_uid()
    ds.Modality = "MR"
    ds.SeriesNumber = 7
    ds.InstanceNumber = instance
    ds.PatientName = patient_name
    ds.Manufacturer = "ArrayView Test"
    ds.ManufacturerModelName = "Fixture"
    ds.MagneticFieldStrength = 3
    ds.RepetitionTime = 2000
    ds.EchoTime = 30
    ds.FlipAngle = 90
    ds.Rows = rows
    ds.Columns = cols
    ds.PixelSpacing = [2, 3]
    ds.SliceThickness = 4
    ds.ImageOrientationPatient = list(orientation)
    ds.ImagePositionPatient = [0, 0, z]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    ds.PixelData = np.full((rows, cols), value, dtype=np.uint16).tobytes()
    ds.save_as(path)


def test_dicom_orders_by_physical_position_not_filename_or_instance(tmp_path):
    uid = generate_uid()
    _write_slice(tmp_path / "c.dcm", series_uid=uid, z=0, value=10, instance=30)
    _write_slice(tmp_path / "a.dcm", series_uid=uid, z=4, value=20, instance=20)
    _write_slice(tmp_path / "b.dcm", series_uid=uid, z=8, value=30, instance=10)

    data, meta = load_data_with_meta(str(tmp_path))

    assert data.shape == (3, 2, 3)
    assert [int(data[..., i].mean()) for i in range(3)] == [10, 20, 30]
    assert meta["voxel_sizes"] == pytest.approx((3, 2, 4))
    assert meta["axis_labels"] == ("R", "A", "S")


def test_dicom_applies_rescale_as_float32(tmp_path):
    uid = generate_uid()
    _write_slice(
        tmp_path / "one.dcm",
        series_uid=uid,
        z=0,
        value=100,
        instance=1,
        slope=2,
        intercept=-1000,
    )

    data, _meta = load_dicom_series(str(tmp_path))

    assert data.dtype == np.float32
    assert np.all(data == -800)


@pytest.mark.parametrize(
    "changed",
    [
        {"rows": 4},
        {"orientation": (0, 1, 0, 1, 0, 0)},
    ],
)
def test_dicom_rejects_inconsistent_geometry(tmp_path, changed):
    uid = generate_uid()
    _write_slice(tmp_path / "one.dcm", series_uid=uid, z=0, value=1, instance=1)
    _write_slice(
        tmp_path / "two.dcm",
        series_uid=uid,
        z=4,
        value=2,
        instance=2,
        **changed,
    )

    with pytest.raises(ValueError, match="inconsistent"):
        load_dicom_series(str(tmp_path))


def test_dicom_metadata_is_structured_json_and_excludes_phi(tmp_path):
    uid = generate_uid()
    _write_slice(tmp_path / "one.dcm", series_uid=uid, z=0, value=1, instance=1)

    _data, meta = load_dicom_series(str(tmp_path))
    payload = json.dumps(meta["dicom_meta"])

    assert [section["label"] for section in meta["dicom_meta"]["sections"]] == [
        "Summary",
        "Acquisition",
        "Geometry",
    ]
    assert "PRIVATE" not in payload
    assert "PatientName" not in payload
    assert uid not in payload
    assert '"provenance": "derived"' in payload


def test_dicom_multi_series_requires_and_honors_selector(tmp_path):
    first_uid = generate_uid()
    second_uid = generate_uid()
    _write_slice(tmp_path / "first.dcm", series_uid=first_uid, z=0, value=11, instance=1)
    _write_slice(tmp_path / "second.dcm", series_uid=second_uid, z=0, value=22, instance=1)
    series = discover_dicom_series(str(tmp_path))
    assert [item["uid"] for item in series] == sorted([first_uid, second_uid])

    with pytest.raises(ValueError, match="Multiple DICOM series"):
        load_dicom_series(str(tmp_path))

    data, _meta = load_dicom_series(str(tmp_path), select=second_uid)
    assert int(data.mean()) == 22


def test_dicom_info_route_exposes_allowlisted_sections(client, tmp_path):
    uid = generate_uid()
    path = tmp_path / "one.dcm"
    _write_slice(path, series_uid=uid, z=0, value=1, instance=1)
    response = client.post("/load", json={"filepath": str(path), "name": "MR series"})
    response.raise_for_status()
    body = response.json()
    assert "sid" in body, body

    info = client.get(f"/info/{body['sid']}").json()
    payload = json.dumps(info["dicom_meta"])
    assert "Acquisition" in payload
    assert "PRIVATE" not in payload
    assert uid not in payload


@pytest.mark.browser
def test_shift_i_has_array_and_dicom_tabs(client, loaded_viewer, tmp_path):
    uid = generate_uid()
    path = tmp_path / "one.dcm"
    _write_slice(path, series_uid=uid, z=0, value=1, instance=1)
    body = client.post("/load", json={"filepath": str(path), "name": "MR series"}).json()
    page = loaded_viewer(body["sid"])

    page.keyboard.press("Shift+I")
    expect(page.locator("#info-overlay")).to_have_class("visible")
    expect(page.locator("#info-tabs")).to_contain_text("Array")
    expect(page.locator("#info-tabs")).to_contain_text("DICOM")
    page.get_by_role("button", name="DICOM").click()
    expect(page.locator("#info-dicom-pane")).to_contain_text("Acquisition")
    expect(page.locator("#info-dicom-pane")).to_contain_text("Geometry")
    expect(page.locator("#info-dicom-pane")).not_to_contain_text("PRIVATE")
