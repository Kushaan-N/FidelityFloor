# Presentation

Self-contained HTML slide deck (9 slides, plain-language) covering the study
through the F1 state-scored results.

- Published: https://claude.ai/code/artifact/862ca239-aff1-4429-9545-4a572db8e418
- `deck_template.html` — deck source with `@@...@@` placeholders for embedded assets
- `build_deck.py` — inlines fonts (IBM Plex woff2, fetched separately) and the
  figures from `results/figures/` as data URIs, emitting `fidelityfloor_deck.html`

Rebuild: adjust the asset paths at the top of `build_deck.py` (they default to
the scratch outputs; `results/figures/` works too), drop the three woff2 files
next to it, then `python build_deck.py` and open the emitted HTML anywhere —
no network needed.

Navigation: ← / → / space / clickable dots; light + dark theme; scrolls
vertically on narrow screens.
