"""Shared Matplotlib configuration for reproducible research figures."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, ft2font

from .legacy_schema import CJK_GLYPH_PROBE


_CHINESE_FONT_CANDIDATES = (
    Path.home() / "Library/Fonts/Kaiti.ttc",
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path.home() / "Library/Fonts/NotoSerifCJKsc-Regular.otf",
)
@dataclass(frozen=True)
class PlotFontConfig:
    chinese_family: str
    chinese_font_path: Path
    english_family: str


def configure_matplotlib(
    *,
    style: str = "seaborn-v0_8-whitegrid",
    chinese_font_path: str | Path | None = None,
    english_family: str = "Times New Roman",
) -> PlotFontConfig:
    """Apply one reproducible plotting style with verified Chinese glyphs.

    The style is applied first because Matplotlib styles may overwrite font
    settings. The selected CJK font is then registered explicitly so notebook
    kernels do not depend on a stale per-user font cache.
    """
    selected_path = _resolve_chinese_font(chinese_font_path)
    font_manager.fontManager.addfont(str(selected_path))
    chinese_family = font_manager.FontProperties(fname=str(selected_path)).get_name()

    plt.style.use(style)
    plt.rcParams.update(
        {
            "font.family": [english_family, chinese_family],
            "font.sans-serif": [chinese_family],
            "axes.unicode_minus": False,
            "axes.titleweight": "semibold",
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )
    return PlotFontConfig(
        chinese_family=chinese_family,
        chinese_font_path=selected_path,
        english_family=english_family,
    )


def _resolve_chinese_font(explicit_path: str | Path | None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())
    elif environment_path := os.environ.get("CB_CJK_FONT_PATH"):
        candidates.append(Path(environment_path).expanduser())
    candidates.extend(_CHINESE_FONT_CANDIDATES)

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.is_file() and _font_supports_text(candidate, CJK_GLYPH_PROBE):
            return candidate.resolve()
    raise FileNotFoundError(
        "No Chinese font with the required glyphs was found. Checked:\n"
        + "\n".join(f"- {path}" for path in checked)
        + "\nSet CB_CJK_FONT_PATH to a usable .ttf/.ttc/.otf file."
    )


def _font_supports_text(font_path: Path, text: str) -> bool:
    character_map = ft2font.FT2Font(str(font_path)).get_charmap()
    return all(ord(character) in character_map for character in text)
