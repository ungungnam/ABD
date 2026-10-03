# Project page

Static project page for the BEDA submission. No build step — plain HTML/CSS/JS.

## Local preview

```bash
python3 -m http.server 8000 --directory docs
# http://localhost:8000
```

## Deploy (GitHub Pages)

Repo **Settings → Pages → Build and deployment**: source `Deploy from a branch`,
branch `main`, folder `/docs`. The page is then served at
`https://ungungnam.github.io/ABD/`.

A Pages site is public even when the repo is private. `index.html` sets
`<meta name="robots" content="noindex">`, but that only discourages indexing — it
does not keep the page private.

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

## Missing: the plots

The draft's line plots have no current export in `../../docs` (the plot PDFs there
are the Sep 5 run, whose baseline numbers no longer match Table I), so the page
currently carries no curves. Export these from the draft and they can be wired in:

- [ ] **Fig. 3** (a) normalized throughput over elapsed time, (b) data affordance over
      elapsed time → goes in `#results`, replacing nothing (it is additive).
- [ ] **Fig. 4** (a) normalized throughput over deployed robot number, (b) robot waiting
      time over deployed robot number → goes in `#scaling`, which is currently text + stats.
- [ ] **Fig. 6a** data throughput over elapsed time for the ablation variants → goes in
      `#ablation`, next to the table.

Drop each as a single-page PDF in `../../docs`, then convert with
`sips -s format png <src> --out static/images/<name>.png -Z 1600` and add a
`<figure class="plot-wrap"><span class="plot"><img ...></span><figcaption>…</figcaption></figure>`
block — same pattern as the Table II figure in `#evaluator`.

## TODO before/after review

- [ ] Add the three figure groups above.
- [ ] Decide on the baseline-comparison videos (`beda/oracle/periodic/continuous.mp4`,
      ~70 MB each in the source folder — compress before adding).
- [ ] Extend the task-suite grid beyond the three pools now shown.
- [ ] After acceptance: add authors, affiliations, paper/arXiv link, code link, and
      BibTeX; drop the `noindex` meta and the "under review" badge.
