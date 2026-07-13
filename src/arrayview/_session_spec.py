"""Dependency-free description of data registered with a viewer session."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from typing import Iterable


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class SourceKind(_StrEnum):
    FILE = "file"
    ARRAY = "array"


class SessionCapability(_StrEnum):
    FILESYSTEM = "filesystem"
    IN_MEMORY = "in_memory"
    WATCH = "watch"
    MULTI_SOURCE = "multi_source"
    OVERLAYS = "overlays"
    VECTORFIELD = "vectorfield"


@dataclass(frozen=True)
class SessionSpec:
    """Normalized registration request shared by all invocation paths.

    Array sources are caller-assigned registration identifiers, not array
    objects.  Keeping this contract data-only makes it safe to pass between
    processes without importing NumPy or server modules.
    """

    source: str
    source_kind: SourceKind
    name: str | None = None
    rgb: bool = False
    compare_sources: tuple[str, ...] = ()
    overlays: tuple[str, ...] = ()
    dims: tuple[int, int] | None = None
    key: str | None = None
    watch: bool = False
    vectorfield: str | None = None
    vectorfield_components_dim: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_kind, SourceKind):
            try:
                object.__setattr__(self, "source_kind", SourceKind(self.source_kind))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unknown source kind: {self.source_kind!r}") from exc
        _require_nonempty("source", self.source)
        if self.name is not None:
            _require_nonempty("name", self.name)
        _validate_sources("compare_sources", self.compare_sources)
        _validate_sources("overlays", self.overlays)
        if self.key is not None:
            _require_nonempty("key", self.key)
        if self.dims is not None:
            if len(self.dims) != 2 or any(
                not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
                for dim in self.dims
            ):
                raise ValueError("dims must contain two non-negative integers")
            if self.dims[0] == self.dims[1]:
                raise ValueError("dims must identify two different axes")
        if self.vectorfield is not None:
            _require_nonempty("vectorfield", self.vectorfield)
        if self.vectorfield_components_dim is not None and (
            not isinstance(self.vectorfield_components_dim, int)
            or isinstance(self.vectorfield_components_dim, bool)
            or self.vectorfield_components_dim < 0
        ):
            raise ValueError("vectorfield_components_dim must be non-negative")
        if self.vectorfield_components_dim is not None and self.vectorfield is None:
            raise ValueError("vectorfield_components_dim requires vectorfield")
        if self.watch and self.source_kind is not SourceKind.FILE:
            raise ValueError("watch is only supported for file sources")

    @property
    def required_capabilities(self) -> tuple[SessionCapability, ...]:
        required = [
            SessionCapability.FILESYSTEM
            if self.source_kind is SourceKind.FILE
            else SessionCapability.IN_MEMORY
        ]
        if self.watch:
            required.append(SessionCapability.WATCH)
        if self.compare_sources:
            required.append(SessionCapability.MULTI_SOURCE)
        if self.overlays:
            required.append(SessionCapability.OVERLAYS)
        if self.vectorfield is not None:
            required.append(SessionCapability.VECTORFIELD)
        return tuple(required)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["source_kind"] = self.source_kind.value
        result["required_capabilities"] = [
            capability.value for capability in self.required_capabilities
        ]
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def session_spec_from_cli_file(
    source: str,
    *,
    name: str | None = None,
    rgb: bool = False,
    compare_sources: Iterable[str] = (),
    overlays: Iterable[str] = (),
    dims: str | Iterable[int] | None = None,
    key: str | None = None,
    watch: bool = False,
    vectorfield: str | None = None,
    vectorfield_components_dim: int | None = None,
) -> SessionSpec:
    """Normalize a CLI file registration without loading the file."""
    return SessionSpec(
        source=source,
        source_kind=SourceKind.FILE,
        name=name,
        rgb=rgb,
        compare_sources=tuple(compare_sources),
        overlays=tuple(overlays),
        dims=_normalize_dims(dims),
        key=key,
        watch=watch,
        vectorfield=vectorfield,
        vectorfield_components_dim=vectorfield_components_dim,
    )


def session_spec_from_python_array(
    source_id: str,
    *,
    name: str | None = None,
    rgb: bool = False,
    compare_source_ids: Iterable[str] = (),
    overlay_source_ids: Iterable[str] = (),
    dims: Iterable[int] | None = None,
    vectorfield_source_id: str | None = None,
    vectorfield_components_dim: int | None = None,
) -> SessionSpec:
    """Normalize IDs assigned to arrays by an in-memory registrar."""
    return SessionSpec(
        source=source_id,
        source_kind=SourceKind.ARRAY,
        name=name,
        rgb=rgb,
        compare_sources=tuple(compare_source_ids),
        overlays=tuple(overlay_source_ids),
        dims=_normalize_dims(dims),
        vectorfield=vectorfield_source_id,
        vectorfield_components_dim=vectorfield_components_dim,
    )


def _normalize_dims(dims: str | Iterable[int] | None) -> tuple[int, int] | None:
    if dims is None:
        return None
    if isinstance(dims, str):
        parts = [part.strip() for part in dims.split(",")]
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            return int(parts[0]), int(parts[1])
        markers = {
            part.lower(): index
            for index, part in enumerate(parts)
            if part.lower() in {"x", "y"}
        }
        if set(markers) == {"x", "y"}:
            return markers["x"], markers["y"]
        raise ValueError("dims must be an integer pair or contain one x and one y")
    return tuple(dims)  # type: ignore[return-value]


def _require_nonempty(field: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _validate_sources(field: str, sources: tuple[str, ...]) -> None:
    if not isinstance(sources, tuple):
        raise ValueError(f"{field} must be a tuple")
    for source in sources:
        _require_nonempty(field, source)
