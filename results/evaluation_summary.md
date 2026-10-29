# Member B consolidated evaluation results

## Table 1: Sampling strategy comparison and 2×2 solver/schedule ablation

| Sampler | Schedule | Steps | Latency (s/image) | FID ↓ | CLIP ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| DDIM | Native | 30 | 5.434 | 151.328 | 31.140 |
| Euler | No Karras | 30 | 5.424 | 152.114 | 31.220 |
| Euler | Karras | 30 | 5.425 | 151.837 | 31.257 |
| DPM++ 2M | No Karras | 30 | 5.348 | 150.440 | 31.178 |
| DPM++ 2M | Karras | 30 | 5.435 | 150.110 | 31.331 |

The four Euler/DPM++ rows form the 2×2 factorial comparison; DDIM is an additional baseline.

## Table 2: DPM++ 2M Karras sampling-step ablation

| Steps | Latency (s/image) | FID ↓ | CLIP ↑ |
| ---: | ---: | ---: | ---: |
| 10 | 1.728 | 151.080 | 31.029 |
| 20 | 3.280 | 151.437 | 31.310 |
| 30 | 5.435 | 150.110 | 31.331 |
| 40 | 6.345 | 151.592 | 31.256 |
| 50 | 7.878 | 150.216 | 31.265 |

## Definitions and interpretation

- All nine configurations use the same 200 COCO prompt–image pairs, seeds 1000 + prompt_id, SD v1.5, CFG 7.5, 512×512 and fp16. Sampler identifies the solver; schedule identifies Karras use.
- `evaluation_raw_results.csv` has 1,800 rows: one row per prompt/configuration. `inference_time_sec` is that prompt's recorded original-generation pipeline call on NVIDIA T4; it includes text encoding, denoising and VAE decoding, after warm-up, and excludes loading/saving. No per-prompt latency is inferred from batch averages.
- Raw `clip_score` is the saved unclamped `clip_raw_100cos` (100× cosine similarity of normalised CLIP ViT-B/32 embeddings). Summary/table CLIP is max(mean(raw CLIP), 0), matching the evaluation protocol.
- `fid_score` is clean-fid/clean/Inception-v3/2048 for the entire 200-image batch against the same 200 reference images. It is repeated on raw rows only to support joins; it is not a per-image FID. Use the nine-row summary for FID comparisons; do not average repeated FID across configurations.
- Summary latency is the mean of the 200 matching original timings. The separately saved 20-prompt same-session timing recheck is supplementary evidence and is not pooled into these tables.
- With 200 images and one seed per prompt, FID is a biased exploratory estimate. Small differences do not establish general superiority. The original SD checkpoint revision was not recorded; the retained metric provenance states this reproducibility limitation.
- Source CSVs, metric provenance and run configurations are preserved. `evaluation_manifest.json` records input/output checksums and is written last after successful validation.
