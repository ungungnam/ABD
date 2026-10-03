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

Note: GitHub Pages is public even on a private repo's paid plans settings — publishing
during the review period makes the submission's media public. `index.html` sets
`<meta name="robots" content="noindex">` but that only discourages indexing.

## Assets

Everything under `static/` is generated from `../../docs` (the LaTeX/media working folder),
which is not part of this repo:

| Published file | Source | Command |
| --- | --- | --- |
| `static/images/concept.png` | `concept_v9.pdf` | `sips -s format png <src> --out <dst> -Z 2000` |
| `static/images/overview.png` | `overview_v2.pdf` | `sips ... -Z 2400` |
| `static/images/throughput.png` | `data_throughput_methods_taskavg.pdf` | `sips ... -Z 1400` |
| `static/images/affordance.png` | `productivity_newdef.pdf` | `sips ... -Z 1600` |
| `static/images/ablation.png` | `data_ablation_throughput_taskavg.pdf` | `sips ... -Z 1400` |
| `static/images/vlm_table.png` | `vlm_table.png` | copied |
| `static/images/hardware.jpg` | `hardware.pdf` | `sips -s format jpeg -s formatOptions 75 ... -Z 1600` |
| `static/videos/teaser.mp4` | `abd_demo_v39_compressed.mp4` | `ffmpeg -i <src> -an -vf scale=1280:-2 -c:v libx264 -crf 30 -preset slow -movflags +faststart <dst>` |
| `static/videos/{pp,cup,dr}_{fwd,rev}.mp4` | `demo_videos/*.mp4` | same, `scale=640:-2` |
| `static/videos/cause_{degrade,infeasible}.mp4` | `cause_*_run0513.mp4` | same, `scale=854:-2` |
| `static/images/teaser_poster.jpg` | `static/videos/teaser.mp4` | `ffmpeg -ss 2 -i <src> -frames:v 1 -q:v 4 <dst>` |

Keep assets web-sized; the whole folder is ~14 MB today.

## TODO before this goes public

- [ ] Re-export the result plots and the comparison table from the **current** draft.
      The present numbers/plots come from the Sep 5 LaTeX snapshot (6 task suites), while the
      current abstract reports 15 real-world tasks and 84.3% of the throughput upper bound.
- [ ] Add the fleet-scaling figure (`#scaling` section is a placeholder).
- [ ] Extend the task grid to the full 15-task suite.
- [ ] Decide on the baseline-comparison videos (`beda/oracle/periodic/continuous.mp4`,
      ~70 MB each in the source folder — compress before adding).
- [ ] After acceptance: add authors, affiliations, paper/arXiv link, code link, and BibTeX;
      drop the `noindex` meta and the "under review" badge.
