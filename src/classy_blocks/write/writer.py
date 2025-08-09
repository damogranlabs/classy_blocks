import datetime
import sys
from dataclasses import asdict
from typing import Callable

from classy_blocks.assemble.dump import AssembledDump
from classy_blocks.assemble.settings import Settings
from classy_blocks.cbtyping import GeometryType
from classy_blocks.util import constants
from classy_blocks.write.formats import format_block, format_edge, format_face, format_patch, format_vertex


def format_geometry(geometry: GeometryType) -> str:
    # nothing to output?
    if len(geometry.items()) == 0:
        return ""

    out = "geometry\n{\n"

    for name, properties in geometry.items():
        out += f"\t{name}\n\t{{\n"

        for prop in properties:
            out += f"\t\t{prop};\n"

        out += "\t}\n"

    out += "};\n\n"

    return out


def format_list(name: str, items: list, formatter: Callable) -> str:
    out = f"{name}\n(\n"

    for item in items:
        out += f"\t{formatter(item)}\n"

    out += ");\n\n"

    return out


def format_settings(settings: Settings):
    out = ""

    if settings.default_patch:
        out += "defaultPatch\n{\n"
        out += f"\tname {settings.default_patch['name']};\n"
        out += f"\ttype {settings.default_patch['kind']};\n"
        out += "}\n\n"

    # a list of merged patches
    out += "mergePatchPairs\n(\n"
    for pair in settings.merged_patches:
        out += f"\t({pair[0]} {pair[1]})\n"
    out += ");\n\n"

    for key, value in asdict(settings).items():
        if key in ("patch_settings", "merged_patches", "default_patch", "geometry"):
            # FIXME: this is a temporary fix, Writer will handle that anyway
            continue

        key_words = key.split("_")
        dict_key = key_words[0] + "".join(word.capitalize() for word in key_words[1:])

        if value is not None:
            out += f"{dict_key} {value};\n"

    out += "\n"

    return out


class MeshWriter:
    def __init__(self, dump: AssembledDump, settings: Settings):
        self.dump = dump
        self.settings = settings

    def write(self, output_path: str):
        with open(output_path, "w", encoding="utf-8") as output:
            output.write(constants.MESH_HEADER.format(script=sys.argv[0], timestamp=str(datetime.datetime.now())))

            output.write(format_geometry(self.settings.geometry))

            output.write(format_list("vertices", self.dump.vertices, format_vertex))
            output.write(format_list("blocks", self.dump.blocks, format_block))
            output.write(format_list("edges", self.dump.edges, format_edge))
            output.write(format_list("boundary", list(self.dump.patch_list.patches.values()), format_patch))
            output.write(format_list("faces", self.dump.face_list.faces, format_face))

            output.write(format_settings(self.settings))

            output.write(constants.MESH_FOOTER)
