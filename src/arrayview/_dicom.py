"""DICOM series discovery, loading, and privacy-safe acquisition metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


_pydicom_mod = None


def _pydicom():
    global _pydicom_mod
    if _pydicom_mod is None:
        import pydicom

        _pydicom_mod = pydicom
    return _pydicom_mod


@dataclass(frozen=True)
class _SliceHeader:
    path: str
    dataset: object


def _read_header(path: str):
    try:
        return _pydicom().dcmread(path, stop_before_pixels=True)
    except Exception:
        return None


def _candidate_paths(source: str) -> list[str]:
    if os.path.isfile(source):
        folder = os.path.dirname(os.path.abspath(source)) or os.curdir
        return [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, name))
        ]
    paths = []
    for root, dirs, files in os.walk(source):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        paths.extend(os.path.join(root, name) for name in sorted(files))
    return paths


def discover_dicom_series(source: str) -> list[dict]:
    """Return image-bearing DICOM series below *source*, sorted deterministically."""
    groups: dict[str, list[_SliceHeader]] = {}
    source_uid = None
    source_abs = os.path.abspath(source)
    for path in _candidate_paths(source):
        ds = _read_header(path)
        if ds is None or not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        if not uid:
            continue
        groups.setdefault(uid, []).append(_SliceHeader(path, ds))
        if os.path.abspath(path) == source_abs:
            source_uid = uid

    result = []
    for uid, headers in groups.items():
        first = headers[0].dataset
        result.append(
            {
                "uid": uid,
                "series_number": str(getattr(first, "SeriesNumber", "") or ""),
                "modality": str(getattr(first, "Modality", "") or ""),
                "count": len(headers),
                "headers": headers,
                "contains_source": uid == source_uid,
            }
        )

    def _sort_key(item):
        raw = item["series_number"]
        try:
            number = (0, int(raw))
        except ValueError:
            number = (1, raw)
        return number, item["uid"]

    return sorted(result, key=_sort_key)


def is_dicom_source(source: str) -> bool:
    """Return whether a file/directory contains at least one image DICOM object."""
    if os.path.isfile(source) and source.lower().endswith(".dcm"):
        return _read_header(source) is not None
    return bool(discover_dicom_series(source))


def _select_series(series: list[dict], source: str, select=None) -> dict:
    if not series:
        raise ValueError(f"No image DICOM series found in {source!r}.")
    containing = [item for item in series if item["contains_source"]]
    if containing:
        return containing[0]
    if select is not None:
        text = str(select).strip()
        for item in series:
            if text == item["uid"] or text == item["series_number"]:
                return item
        if text.isdigit() and 1 <= int(text) <= len(series):
            return series[int(text) - 1]
        raise ValueError(f"DICOM series selector {select!r} did not match this source.")
    if len(series) == 1:
        return series[0]
    summary = "; ".join(
        f"{index}: series {item['series_number'] or '?'} "
        f"({item['modality'] or 'DICOM'}, {item['count']} images)"
        for index, item in enumerate(series, 1)
    )
    raise ValueError(
        f"Multiple DICOM series found: {summary}. Select one with --series INDEX."
    )


def resolve_dicom_series_path(source: str, select) -> str:
    """Resolve a series selector to one representative slice path."""
    selected = _select_series(discover_dicom_series(source), source, select=select)
    return selected["headers"][0].path


def _vector(ds, name, length):
    value = getattr(ds, name, None)
    if value is None or len(value) != length:
        return None
    try:
        return np.asarray([float(item) for item in value], dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _ordered_headers(headers: list[_SliceHeader]):
    first = headers[0].dataset
    orientation = _vector(first, "ImageOrientationPatient", 6)
    warnings = []
    if orientation is not None:
        row, col = orientation[:3], orientation[3:]
        normal = np.cross(row, col)
        norm = float(np.linalg.norm(normal))
        if norm > 0:
            normal /= norm
            positions = []
            for header in headers:
                current = _vector(header.dataset, "ImageOrientationPatient", 6)
                position = _vector(header.dataset, "ImagePositionPatient", 3)
                if current is not None and not np.allclose(
                    current, orientation, atol=1e-4, rtol=1e-4
                ):
                    raise ValueError("DICOM series contains inconsistent slice orientations.")
                if current is None or position is None:
                    positions = []
                    break
                positions.append(float(np.dot(position, normal)))
            if positions:
                order = np.argsort(positions)
                ordered_positions = np.asarray(positions)[order]
                if len(ordered_positions) > 1 and np.any(
                    np.isclose(np.diff(ordered_positions), 0.0, atol=1e-4)
                ):
                    raise ValueError("DICOM series contains duplicate slice positions.")
                return [headers[int(i)] for i in order], ordered_positions, orientation, warnings

    instances = []
    for header in headers:
        try:
            instances.append(int(header.dataset.InstanceNumber))
        except (AttributeError, TypeError, ValueError):
            raise ValueError(
                "DICOM slices lack usable ImagePositionPatient geometry and unique InstanceNumber values."
            )
    if len(set(instances)) != len(instances):
        raise ValueError("DICOM series contains duplicate InstanceNumber values.")
    warnings.append("Slice order uses InstanceNumber because physical positions are unavailable.")
    order = np.argsort(instances)
    return [headers[int(i)] for i in order], None, orientation, warnings


def _field(label, value, unit=None, *, provenance="header", source=None):
    if value is None or value == "":
        return None
    result = {"label": label, "value": value, "provenance": provenance}
    if unit:
        result["unit"] = unit
    if source:
        result["source"] = source
    return result


def _number(ds, name):
    value = getattr(ds, name, None)
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)):
            value = value[0]
        return float(value)
    except (TypeError, ValueError):
        return None


def _tag_number(ds, tag):
    try:
        value = ds[tag].value
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            value = value[0]
        if isinstance(value, bytes):
            value = value.decode(errors="ignore").split("\\")[0]
        return float(value)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _effective_echo_spacing(ds):
    phase_steps = _number(ds, "NumberOfPhaseEncodingSteps") or _number(
        ds, "AcquisitionMatrixPE"
    )
    if not phase_steps or phase_steps <= 1:
        return None, None, None
    bw_phase = _tag_number(ds, (0x0019, 0x1028)) or _tag_number(
        ds, (0x0021, 0x1153)
    )
    if bw_phase and bw_phase > 0:
        spacing_ms = 1000.0 / (bw_phase * phase_steps)
        return spacing_ms, spacing_ms * (phase_steps - 1), "BWPerPxPhaseEncode + phase steps"
    ge_spacing_us = _tag_number(ds, (0x0043, 0x102C))
    if ge_spacing_us and ge_spacing_us > 0:
        spacing_ms = ge_spacing_us / 1000.0
        return spacing_ms, spacing_ms * (phase_steps - 1), "GE effective echo spacing + phase steps"
    return None, None, None


def _acquisition_matrix(ds):
    value = getattr(ds, "AcquisitionMatrix", None)
    if value is None:
        return None
    try:
        values = [int(item) for item in value]
        if len(values) != 4:
            return None
        frequency = max(values[0], values[1])
        phase = max(values[2], values[3])
        return f"{frequency} x {phase}" if frequency and phase else None
    except (TypeError, ValueError):
        return None


def _diffusion_rows(headers):
    shells: dict[int, dict] = {}
    for header in headers:
        ds = header.dataset
        b_value = _number(ds, "DiffusionBValue")
        if b_value is None:
            b_value = _tag_number(ds, (0x0018, 0x9087))
        if b_value is None:
            continue
        shell = shells.setdefault(
            int(round(b_value)),
            {"nex": _number(ds, "NumberOfAverages"), "directions": set()},
        )
        direction = getattr(ds, "DiffusionGradientOrientation", None)
        if direction is not None:
            try:
                shell["directions"].add(
                    tuple(round(float(component), 3) for component in direction)
                )
            except (TypeError, ValueError):
                pass
    return [
        {
            "b_value": b_value,
            "nex": _clean_number(item["nex"]),
            "directions": len(item["directions"]) or None,
        }
        for b_value, item in sorted(shells.items())
    ]


def _clean_number(value, digits=4):
    if value is None:
        return None
    rounded = round(float(value), digits)
    return int(rounded) if rounded.is_integer() else rounded


def _dicom_metadata(ds, *, headers, shape, spacing, spacing_regular, orientation, warnings):
    rows, cols, slices = int(shape[1]), int(shape[0]), int(shape[2])
    pixel_spacing = _vector(ds, "PixelSpacing", 2)
    fov = None
    if pixel_spacing is not None:
        fov = [
            _clean_number(pixel_spacing[1] * cols),
            _clean_number(pixel_spacing[0] * rows),
        ]
    plane = None
    if orientation is not None:
        normal = np.abs(np.cross(orientation[:3], orientation[3:]))
        plane = ("sagittal", "coronal", "axial")[int(np.argmax(normal))]
    echo_spacing, total_readout, echo_source = _effective_echo_spacing(ds)
    diffusion_rows = _diffusion_rows(headers)

    sop = getattr(ds, "SOPClassUID", None)
    sop_name = getattr(sop, "name", None) if sop is not None else None
    summary = [
        _field("Modality", str(getattr(ds, "Modality", "") or "")),
        _field("Series", str(getattr(ds, "SeriesNumber", "") or "")),
        _field("Images", slices),
        _field("SOP class", sop_name),
        _field("Plane", plane, provenance="derived", source="ImageOrientationPatient"),
        _field("Manufacturer", str(getattr(ds, "Manufacturer", "") or "")),
        _field("Scanner model", str(getattr(ds, "ManufacturerModelName", "") or "")),
        _field("Field strength", _clean_number(_number(ds, "MagneticFieldStrength")), "T"),
    ]
    acquisition = [
        _field("TR", _clean_number(_number(ds, "RepetitionTime")), "ms"),
        _field("TE", _clean_number(_number(ds, "EchoTime")), "ms"),
        _field("TI", _clean_number(_number(ds, "InversionTime")), "ms"),
        _field("Flip angle", _clean_number(_number(ds, "FlipAngle")), "deg"),
        _field("Scanning sequence", str(getattr(ds, "ScanningSequence", "") or "")),
        _field("Sequence variant", str(getattr(ds, "SequenceVariant", "") or "")),
        _field("Echo train length", _clean_number(_number(ds, "EchoTrainLength"))),
        _field("Averages / NEX", _clean_number(_number(ds, "NumberOfAverages"))),
        _field("Pixel bandwidth", _clean_number(_number(ds, "PixelBandwidth")), "Hz/px"),
        _field(
            "Effective echo spacing",
            _clean_number(echo_spacing),
            "ms",
            provenance="inferred",
            source=echo_source,
        ),
        _field(
            "Total readout time",
            _clean_number(total_readout),
            "ms",
            provenance="derived",
            source="effective echo spacing x (phase steps - 1)",
        ),
        _field(
            "In-plane acceleration",
            _clean_number(_number(ds, "ParallelReductionFactorInPlane")),
        ),
        _field(
            "Through-plane acceleration",
            _clean_number(_number(ds, "ParallelReductionFactorOutOfPlane")),
        ),
        _field("Parallel imaging", str(getattr(ds, "ParallelAcquisitionTechnique", "") or "")),
        _field("Partial Fourier", _clean_number(_number(ds, "PartialFourier"))),
        _field("Diffusion b-value", _clean_number(_number(ds, "DiffusionBValue")), "s/mm2"),
        _field("SAR", _clean_number(_number(ds, "SAR")), "W/kg"),
    ]
    geometry = [
        _field("Matrix", f"{cols} x {rows} x {slices}"),
        _field("Acquisition matrix", _acquisition_matrix(ds)),
        _field(
            "Pixel spacing",
            [_clean_number(pixel_spacing[1]), _clean_number(pixel_spacing[0])]
            if pixel_spacing is not None
            else None,
            "mm",
        ),
        _field(
            "Slice spacing",
            _clean_number(spacing),
            "mm",
            provenance="derived",
            source="ImagePositionPatient",
        ),
        _field("Spacing regular", spacing_regular, provenance="derived"),
        _field("Slice thickness", _clean_number(_number(ds, "SliceThickness")), "mm"),
        _field(
            "Reconstructed voxel",
            [
                _clean_number(pixel_spacing[1]),
                _clean_number(pixel_spacing[0]),
                _clean_number(spacing),
            ]
            if pixel_spacing is not None
            else None,
            "mm",
            provenance="derived",
            source="PixelSpacing + physical slice spacing",
        ),
        _field(
            "Field of view",
            fov,
            "mm",
            provenance="derived",
            source="PixelSpacing x matrix",
        ),
        _field("Phase encode direction", str(getattr(ds, "InPlanePhaseEncodingDirection", "") or "")),
        _field("Patient position", str(getattr(ds, "PatientPosition", "") or "")),
    ]
    return {
        "sections": [
            {"id": "summary", "label": "Summary", "fields": [x for x in summary if x]},
            {"id": "acquisition", "label": "Acquisition", "fields": [x for x in acquisition if x]},
            {"id": "geometry", "label": "Geometry", "fields": [x for x in geometry if x]},
        ],
        "diffusion_rows": diffusion_rows,
        "warnings": list(warnings),
    }


def load_dicom_series(source: str, *, select=None):
    """Load one DICOM series and return a canonical RAS array plus metadata."""
    selected = _select_series(discover_dicom_series(source), source, select=select)
    ordered, positions, orientation, warnings = _ordered_headers(selected["headers"])
    first = ordered[0].dataset
    rows, cols = int(first.Rows), int(first.Columns)
    pixel_spacing = _vector(first, "PixelSpacing", 2)
    if pixel_spacing is None or np.any(pixel_spacing <= 0):
        raise ValueError("DICOM series is missing valid PixelSpacing.")

    decoded = []
    rescaled = False
    for header in ordered:
        ds = _pydicom().dcmread(header.path)
        if int(ds.Rows) != rows or int(ds.Columns) != cols:
            raise ValueError("DICOM series contains inconsistent image dimensions.")
        if int(getattr(ds, "SamplesPerPixel", 1) or 1) != 1:
            raise ValueError("Only monochrome DICOM series are supported for now.")
        try:
            arr = np.asarray(ds.pixel_array)
        except Exception as exc:
            syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
            syntax_name = getattr(syntax, "name", None) or str(syntax or "unknown")
            raise ValueError(
                "Cannot decode DICOM pixel data using transfer syntax "
                f"{syntax_name}. Install the matching pydicom codec plugin "
                "(usually pylibjpeg with libjpeg/openjpeg, or pyjpegls)."
            ) from exc
        if arr.ndim != 2:
            raise ValueError("Enhanced or multi-frame DICOM needs explicit per-frame geometry support.")
        slope = _number(ds, "RescaleSlope")
        intercept = _number(ds, "RescaleIntercept")
        slope = 1.0 if slope is None else slope
        intercept = 0.0 if intercept is None else intercept
        if slope != 1.0 or intercept != 0.0:
            arr = arr.astype(np.float32) * np.float32(slope) + np.float32(intercept)
            rescaled = True
        decoded.append(arr)

    volume = np.stack(decoded, axis=-1).transpose(1, 0, 2)
    if rescaled:
        volume = volume.astype(np.float32, copy=False)

    if positions is not None and len(positions) > 1:
        gaps = np.abs(np.diff(positions))
        slice_spacing = float(np.median(gaps))
        spacing_regular = bool(
            np.allclose(gaps, slice_spacing, rtol=0.01, atol=0.1)
        )
        if not spacing_regular:
            warnings.append("Slice spacing is irregular; physical ruler distances may be approximate.")
    else:
        slice_spacing = _number(first, "SpacingBetweenSlices") or _number(first, "SliceThickness") or 1.0
        spacing_regular = None

    if orientation is None:
        orientation = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float64)
    row, col = orientation[:3], orientation[3:]
    normal = np.cross(row, col)
    origin = _vector(first, "ImagePositionPatient", 3)
    if origin is None:
        origin = np.zeros(3, dtype=np.float64)
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = row * pixel_spacing[1]
    affine_lps[:3, 1] = col * pixel_spacing[0]
    affine_lps[:3, 2] = normal * slice_spacing
    affine_lps[:3, 3] = origin
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine = lps_to_ras @ affine_lps

    import nibabel as nib

    source_ornt = nib.orientations.io_orientation(affine)
    canonical_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(source_ornt, canonical_ornt)
    canonical = nib.orientations.apply_orientation(volume, transform)
    affine_canonical = affine @ nib.orientations.inv_ornt_aff(transform, volume.shape)
    rot = affine_canonical[:3, :3]
    voxel_sizes = tuple(float(np.linalg.norm(rot[:, i])) for i in range(3))
    norm_rot = np.column_stack(
        [rot[:, i] / voxel_sizes[i] if voxel_sizes[i] else rot[:, i] for i in range(3)]
    )
    off_diag_max = max(abs(norm_rot[i, j]) for i in range(3) for j in range(3) if i != j)
    spatial_meta = {
        "affine": affine,
        "affine_canonical": affine_canonical,
        "voxel_sizes": voxel_sizes,
        "axis_labels": tuple(str(v) for v in nib.aff2axcodes(affine_canonical)),
        "is_oblique": bool(off_diag_max > 1e-3),
        "canonical_shape": tuple(int(v) for v in canonical.shape),
    }
    spatial_meta["dicom_meta"] = _dicom_metadata(
        first,
        headers=ordered,
        shape=volume.shape,
        spacing=slice_spacing,
        spacing_regular=spacing_regular,
        orientation=orientation,
        warnings=warnings,
    )
    return np.asarray(canonical), spatial_meta
