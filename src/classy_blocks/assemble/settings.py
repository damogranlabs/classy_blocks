from dataclasses import dataclass, field
from typing import Optional

from classy_blocks.cbtyping import GeometryType


@dataclass
class Settings:
    prescale: Optional[int] = None
    scale: float = 1
    transform: Optional[str] = None
    merge_type: Optional[str] = None
    check_face_correspondence: Optional[str] = None
    verbose: Optional[str] = None

    # user-provided geometry data
    geometry: GeometryType = field(default_factory=dict)
    # user-provided patch data
    # default patch, single entry only
    default_patch: dict[str, str] = field(default_factory=dict)
    # merged patches, a list of pairs of patch names
    merged_patches: list[tuple[str, str]] = field(default_factory=list)

    patch_settings: dict[str, list[str]] = field(default_factory=dict)

    def add_geometry(self, geometry: GeometryType) -> None:
        self.geometry = {**self.geometry, **geometry}

    def modify_patch(self, name: str, kind: str, settings: Optional[list[str]] = None) -> None:
        if settings is None:
            settings = []

        self.patch_settings[name] = [kind, *settings]

    @property
    def slave_patches(self) -> set[str]:
        # TODO: cache
        return {p[1] for p in self.merged_patches}
