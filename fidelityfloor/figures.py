"""Headline figures (spec §1.4), regenerated from the stats summary JSON in minutes.

fig1 — decision utility vs MEASURED rollout error, one curve per error axis, per
       scoring mode; no-imagination baseline and determinism floor annotated.
fig2 — systematic vs stochastic at matched error magnitude (the paper's core).
fig3 — Tier 1 accuracy vs condition with both anchors.
contact_sheet — G4 visual inspection aid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

AXIS_STYLE = {
    "miscalibration": {"color": "#c0392b", "marker": "o", "label": "miscalibration (systematic)"},
    "dyn_noise": {"color": "#2980b9", "marker": "s", "label": "dyn. noise (stochastic)"},
    "horizon": {"color": "#8e44ad", "marker": "^", "label": "horizon truncation"},
    "obs_corruption": {"color": "#e67e22", "marker": "D", "label": "obs. corruption"},
}


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_utility_vs_error(summary: dict, out_dir: Path, mode: str = "state",
                         metric: str = "spearman", determinism_floor: float | None = None,
                         baseline: float | None = None) -> Path | None:
    plt = _mpl()
    key = f"{mode}_{metric}"
    per = summary["per_condition"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    plotted = False
    for axis, style in AXIS_STYLE.items():
        pts = []
        for cid, entry in per.items():
            if not cid.startswith(axis + "-") or key not in entry:
                continue
            err_key = "err_final" if axis == "obs_corruption" else "err_mean_matched"
            # obs_corruption has no state-space error; skip it on state-scored plots
            if axis == "obs_corruption" and mode == "state":
                continue
            e = entry["induced_error"]["err_mean_matched"]
            m = entry[key]
            pts.append((e, m["mean"], m["lo"], m["hi"]))
        if not pts:
            continue
        pts.sort()
        e, mu, lo, hi = map(np.array, zip(*pts))
        ax.errorbar(e, mu, yerr=[mu - lo, hi - mu], **{k: v for k, v in style.items()
                    if k != "label"}, label=style["label"], capsize=3, lw=1.5)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ident = per.get("identity", {}).get(key)
    if ident:
        ax.axhline(ident["mean"], color="0.4", ls=":", lw=1,
                   label=f"identity ({ident['mean']:.2f})")
    if baseline is not None:
        ax.axhline(baseline, color="k", ls="--", lw=1, label="no-imagination baseline")
    if determinism_floor is not None:
        ax.axvline(determinism_floor, color="0.6", ls="-.", lw=1, label="determinism floor")
    ax.set_xlabel("measured induced rollout error (m, mean matched-step)")
    ax.set_ylabel(f"{metric} vs GT ranking ({mode}-scored)")
    ax.set_title(f"Decision utility vs world-model error — {mode}-scored")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / f"fig1_{mode}_{metric}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_matched_error(summary: dict, out_dir: Path) -> Path | None:
    plt = _mpl()
    me = summary.get("matched_error", {})
    if not me.get("overlap"):
        return None
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    g = np.array(me["error_grid"])
    ax.plot(g, me["utility_miscalibration"], color="#c0392b", lw=2,
            label="systematic (miscalibration)")
    ax.plot(g, me["utility_dyn_noise"], color="#2980b9", lw=2, label="stochastic (dyn noise)")
    ax.fill_between(g, me["utility_miscalibration"], me["utility_dyn_noise"],
                    color="0.85", alpha=0.6)
    pd = me["paired_diff_noise_minus_misc"]
    ax.set_title(
        "Systematic vs stochastic error at matched magnitude\n"
        f"paired Δ(noise−misc) = {pd['mean']:.3f}  95% CI [{pd['lo']:.3f}, {pd['hi']:.3f}]"
    )
    ax.set_xlabel("measured induced rollout error (m)")
    ax.set_ylabel(me["metric"])
    ax.legend()
    fig.tight_layout()
    p = out_dir / "fig2_matched_error.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_tier1(tier1_summary: dict, out_dir: Path) -> Path | None:
    plt = _mpl()
    conds = tier1_summary.get("conditions")
    if not conds:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    names = list(conds.keys())
    means = [conds[n]["accuracy"]["mean"] for n in names]
    los = [conds[n]["accuracy"]["lo"] for n in names]
    his = [conds[n]["accuracy"]["hi"] for n in names]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
           capsize=3, color="#e67e22")
    for anchor, style in (("no_imagination", "k--"), ("perfect_imagination", "g--")):
        if anchor in tier1_summary:
            ax.axhline(tier1_summary[anchor]["mean"], ls="--",
                       color=style[0], lw=1, label=anchor.replace("_", " "))
    ax.set_xticks(x, names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("QA accuracy")
    ax.set_title("Tier 1: imagination-assisted spatial QA vs condition")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "fig3_tier1.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def contact_sheet(image_paths: list, out_path: Path, cols: int = 5,
                  labels: list[str] | None = None) -> Path:
    """G4: tile frames into one inspectable sheet."""
    from PIL import Image, ImageDraw

    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    if not imgs:
        raise ValueError("no images for contact sheet")
    w, h = imgs[0].size
    tw, th = w // 2, h // 2
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), "white")
    for i, im in enumerate(imgs):
        tile = im.resize((tw, th))
        if labels and i < len(labels):
            ImageDraw.Draw(tile).text((4, 4), labels[i], fill="yellow")
        sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
