from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPixmap

WEB = Path(__file__).parent / "web"
ASSETS = WEB / "assets"
IMG = ASSETS / "img"
FONTS = [ASSETS / "fonts" / "magdacleanmono-bold.ttf",
         ASSETS / "fonts" / "Lekton-Bold.ttf", ASSETS / "fonts" / "Lekton-Regular.ttf"]

WHITE = QColor(255, 255, 255)
BLACK = QColor(0, 0, 0)
RED = QColor(255, 0, 0, 204)
GREY = QColor(150, 150, 150)
DIM = QColor(255, 255, 255, 110)
YELLOW = QColor(229, 255, 0, 204)

_families: dict[str, str] = {}


def load_fonts():
    for f in FONTS:
        if f.exists():
            fid = QFontDatabase.addApplicationFont(str(f))
            fams = QFontDatabase.applicationFontFamilies(fid)
            if fams:
                _families[f.stem] = fams[0]


def mono(size: float, bold: bool = True) -> QFont:
    fam = _families.get("magdacleanmono-bold") if bold else None
    fam = fam or _families.get("Lekton-Bold" if bold else "Lekton-Regular") or "Consolas"
    f = QFont(fam)
    f.setPointSizeF(size)
    f.setBold(bold and "magda" not in fam.lower())
    f.setStyleHint(QFont.Monospace)
    f.setHintingPreference(QFont.PreferNoHinting)
    return f


@lru_cache(maxsize=64)
def pixmap(name: str) -> QPixmap:
    return QPixmap(str(IMG / f"{name}.png"))
