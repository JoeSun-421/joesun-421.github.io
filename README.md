# Zhaoyuan Sun

This repository contains the source for [Zhaoyuan Sun's personal website](https://joesun-421.github.io/), a static GitHub Pages site.

## Site structure

- `index.html` — homepage
- `blogs/` — technical blog, article Markdown sources, and local figures
- `ideas.html` — research ideas and directions
- `favorite-works.html` — selected papers and projects
- `mmlab-intern-guide.html` — an internship guide
- `daily-report/` — daily reports, the latest edition, and the dated archive

## Verified RSI proposal

The Verified RSI proposal is maintained in a separate project repository. Its source page is `proposal/verified-rsi.html`.

- [Read the proposal](https://joesun-421.github.io/proposal4rsi/proposal/verified-rsi.html)
- [Open the proposal project](https://github.com/JoeSun-421/proposal4rsi)

## Blog publishing

The published site is static HTML and uses `.nojekyll`. The existing homepage, collections, and daily reports remain static files.

- Edit `blogs/posts.json` and each post's `source.md`.
- Install development dependencies with `npm ci --ignore-scripts`.
- Run `npm run build:blogs` to render Markdown and math to HTML.

KaTeX 0.19.1 is pinned; formulas are compiled with strict error checking. Its CSS, fonts and MIT license are hosted under `assets/vendor/`. Article figures are local, with source and license records beside them. The published Markdown replaces project-local PDF links with fixed arXiv versions and omits links to unpublished notes.
