# Project page

Static project page for the BEDA submission. No build step — plain HTML/CSS/JS.

## Local preview

```bash
python3 -m http.server 8000 --directory docs
# http://localhost:8000
```

## Deploy (GitHub Pages)

The repo is public and Pages is enabled on `main` + `/docs`, so this folder *is*
the site: <https://ungungnam.github.io/BEDA/>. A push to `main` rebuilds it;
`gh api repos/ungungnam/BEDA/pages/builds/latest --jq .status` reports the build.

It is linked as "project" from the BEDA card on <https://ungungnam.github.io/>,
which follows the same one-repo-per-project pattern as `/PaPA/`. The repo was
renamed `ABD` → `BEDA` to get this URL; note that Pages paths are not
redirected, so the old `/ABD/` path is gone (github.com repo URLs do redirect).

Everything in a Pages repo is publicly fetchable, this README included.
`index.html` sets `<meta name="robots" content="noindex">`, which discourages
search indexing but does not make the page private.

## Where the content comes from

Text, tables and statistics come from the current draft
(`../../docs/_ICRA_2027__BEDA__Bidirectinal_Evaluation_of_Data_Affordance (7).pdf`,
Oct 3): title, abstract, Table I, Table II, the Fig. 6b ablation numbers, the
multi-robot scaling results, and the cost/error analysis. Author names and BibTeX
are deliberately absent while the paper is under review.

The `.tex` files in `../../docs` are a Sep 5 snapshot and no longer match the
draft — do not take numbers from them.

## Assets

Everything under `static/` is generated from `../../docs` (the LaTeX/media working
folder), which is not part of this repo:

| Published file | Source | Command |
| --- | --- | --- |
| `static/images/concept.png` | `concept_v9.pdf` (= Fig. 1) | `sips -s format png <src> --out <dst> -Z 2000` |
| `static/images/overview.png` | `overview_v2.pdf` (= Fig. 2) | `sips ... -Z 2400` |
| `static/images/vlm_table.png` | `vlm_table.png` (= Table II) | copied |
| `static/images/throughput_time.png` | draft p5, Fig. 3(a) | PyMuPDF clip render, see below |
| `static/images/affordance_time.png` | draft p5, Fig. 3(b) | PyMuPDF clip render |
| `static/images/ablation_throughput.png` | draft p7, Fig. 6(a) | PyMuPDF clip render |
| `static/images/scaling_throughput.png` | draft p6, Fig. 4(a) | embedded image, extracted natively |
| `static/images/scaling_waiting.png` | draft p6, Fig. 4(b) | embedded image, extracted natively |
| `static/images/fn_streaks.png` | draft p6, Fig. 5 | embedded image, extracted natively |
| `static/images/hardware.jpg` | `hardware.pdf` | `sips -s format jpeg -s formatOptions 75 ... -Z 1600` |
| `static/videos/teaser.mp4` | `abd_demo_v39_compressed.mp4` | `ffmpeg -ss 3 -i <src> -t 173 -an -vf scale=1280:-2 -c:v libx264 -crf 30 -preset slow -movflags +faststart <dst>` |
| `static/videos/{pp,cup,dr}_{fwd,rev}.mp4` | `demo_videos/*.mp4` | same, `scale=640:-2` |
| `static/videos/cause_{degrade,infeasible}.mp4` | `cause_*_run0513.mp4` | same, `scale=854:-2` |
| `static/images/teaser_poster.jpg` | `static/videos/teaser.mp4` | `ffmpeg -ss 1 -i <src> -frames:v 1 -q:v 4 <dst>` |

Fig. 1 and Fig. 2 were confirmed current by matching their captions against the
draft. The `-ss 3 -t 173` on the teaser is deliberate: the source video's opening
title card and closing "thank you" card both print the submission number, so both
are cut. Keep them cut while the paper is under review, and re-check the first and
last frames after any re-export.

Keep assets web-sized; the whole folder is ~13 MB today.

## Pulling figures out of the draft

The plots have no standalone export in `../../docs`, so they come straight out of
the draft PDF. There is no `pdftoppm`/`pdftotext` on this machine and `sips` only
converts page 1, so use PyMuPDF in a throwaway venv:

```bash
python3 -m venv /tmp/pdfvenv && /tmp/pdfvenv/bin/pip install pymupdf
```

Vector figures are rendered from a clip rectangle at zoom 6 (~432 dpi); the two
Fig. 4 panels and Fig. 5 are embedded rasters and come out at native resolution
(~6000 px wide, downscale with `sips -Z 1800`):

```python
import pymupdf
d = pymupdf.open('paper.pdf')
# page index, clip rect in PDF points — page.get_text('blocks') gives the
# caption/axis-label boxes these were read off, so re-derive them if the
# layout moves
vec = {'throughput_time':     (4, (82, 52, 240, 181)),   # Fig. 3(a)
       'affordance_time':     (4, (248, 52, 532, 181)),  # Fig. 3(b)
       'ablation_throughput': (6, (86, 54, 266, 202))}   # Fig. 6(a)
for name, (pno, clip) in vec.items():
    d[pno].get_pixmap(matrix=pymupdf.Matrix(6, 6),
                      clip=pymupdf.Rect(*clip), alpha=False).save(f'{name}.png')

for info in d[5].get_image_info(xrefs=True):   # Fig. 4(a), 4(b), Fig. 5
    img = d.extract_image(info['xref'])
    print(info['bbox'], img['width'], img['height'])
```

Re-run this after any draft revision; the clip rects are tied to the current
layout, so check each output before publishing.

## TODO before/after review

- [ ] Fig. 3(a) ends at ~0.52 for Unstructured VLM and ~0.11 for RADAR, while
      Table I reports 0.619 and 0.149 for the same two baselines. Both are on the
      page as the draft has them — worth checking which aggregation is intended.

- [ ] Decide on the baseline-comparison videos (`beda/oracle/periodic/continuous.mp4`,
      ~70 MB each in the source folder — compress before adding).
- [ ] Extend the task-suite grid beyond the three pools now shown.
- [ ] After acceptance: add authors, affiliations, paper/arXiv link, code link, and
      BibTeX; drop the `noindex` meta and the "under review" badge.
