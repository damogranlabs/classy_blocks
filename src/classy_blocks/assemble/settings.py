from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

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
    default_patch: Dict[str, str] = field(default_factory=dict)
    # merged patches, a list of pairs of patch names
    merged_patches: List[Tuple[str, str]] = field(default_factory=list)

    patch_settings: Dict[str, List[str]] = field(default_factory=dict)

    def add_geometry(self, geometry: GeometryType) -> None:
        self.geometry = {**self.geometry, **geometry}

    def modify_patch(self, name: str, kind: str, settings: Optional[List[str]] = None) -> None:
        if settings is None:
            settings = []

        self.patch_settings[name] = [kind, *settings]

    @property
    def slave_patches(self) -> Set[str]:
        # TODO: cache
        return {p[1] for p in self.merged_patches}

    def format(self) -> str:
        """Put self.settings in a proper, blockMesh-readable format"""
        out = ""

        for key, value in asdict(self).items():
            if key in ("patch_settings", "merged_patches", "default_patch", "geometry"):
                # FIXME: this is a temporary fix, Writer will handle that anyway
                continue

            if value is not None:
                out += f"{key} {value};\n"

        out += "\n"

        return out
