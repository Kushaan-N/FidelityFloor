#!/usr/bin/env python3
"""Assemble the FidelityFloor slide deck with embedded fonts + figures."""
import base64
import io
from pathlib import Path

DECK = Path(__file__).parent
FIGS = Path("/scratch4/workspace/knaskar_umass_edu-fidelityfloor/outputs/bc822b1d59a6/figures")
SCRATCH = DECK.parent


def b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def jpeg_b64(path: Path, max_w: int, q: int = 87) -> str:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    return base64.standard_b64encode(buf.getvalue()).decode()


ASSETS = {
    "@@F_COND@@": b64(DECK / "cond-sb.woff2"),
    "@@F_MONO@@": b64(DECK / "mono-reg.woff2"),
    "@@F_MONOM@@": b64(DECK / "mono-med.woff2"),
    "@@IMG_FIG1@@": b64(FIGS / "fig1_state_spearman.png"),
    "@@IMG_REGRET@@": b64(FIGS / "fig1_state_normalized_regret.png"),
    "@@IMG_FIG2@@": b64(FIGS / "fig2_matched_error.png"),
    "@@IMG_SHEET@@": jpeg_b64(FIGS / "contact_obs_corruption.png", 1080),
    "@@IMG_SCENE@@": jpeg_b64(SCRATCH / "smoke_frame_v3.png", 640, q=90),
}

html = (DECK / "deck_template.html").read_text()
for k, v in ASSETS.items():
    html = html.replace(k, v)
out = DECK / "fidelityfloor_deck.html"
out.write_text(html)
print(out, f"{out.stat().st_size/1e6:.2f} MB")
