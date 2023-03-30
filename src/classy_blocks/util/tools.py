"""Misc utilities"""


def indent(text: str, levels: int) -> str:
    """Indents 'text' by 'levels' tab characters"""
    return "\t" * levels + text + "\n"
