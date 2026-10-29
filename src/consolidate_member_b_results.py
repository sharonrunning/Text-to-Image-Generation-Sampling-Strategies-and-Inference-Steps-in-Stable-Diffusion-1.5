#!/usr/bin/env python3
"""Consolidate completed CA6127 Member B evaluation; Python 3.10+, stdlib only.

Run from the project root: python src/consolidate_member_b_results.py
Or: python /path/to/consolidate_member_b_results.py --project-root /path/to/project
Use --check-only to validate every input without creating or replacing outputs.

FID is a distribution-level metric, NOT an individual-image score. Its batch
value is repeated in the requested seven-column master CSV for a convenient
join. Never average these repeated FID values across different configurations.
Per-prompt CLIP is the recorded, unclamped 100*cos; the batch CLIP score is
max(mean(per-prompt CLIP), 0). Latencies come from the original 200-image
generation log, not the separate 20-prompt same-session timing recheck.

This script does not generate images, calculate new metrics, or delete inputs.
All nine batches must be complete. Validation happens before any output changes;
each output is replaced atomically and a checksum manifest is committed last.

The three handoff files are written to results/. Inputs are found anywhere under
results/ (normally results/evaluation_details/), and the checksum manifest is
written to results/evaluation_details/. An output that is already current is
left untouched.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile


EXPECTED_COUNT = 200
BASELINE_VERSIONS = {
    "torch": "2.11.0+cu130", "diffusers": "0.40.0", "transformers": "5.18.0"
}
# batch -> source sampler, exported solver, noise schedule, denoising steps
BATCHES = {
    "ddim_s30": ("ddim", "ddim", "native", 30),
    "euler_s30": ("euler", "euler", "non_karras", 30),
    "dpmpp_s30": ("dpmpp", "dpmpp", "karras", 30),
    "euler_karras_s30": ("euler_karras", "euler", "karras", 30),
    "dpmpp_nokarras_s30": ("dpmpp_nokarras", "dpmpp", "non_karras", 30),
    **{f"dpmpp_s{s}": ("dpmpp", "dpmpp", "karras", s) for s in (10, 20, 40, 50)},
}
RAW_COLUMNS = [
    "prompt_id", "sampler", "steps", "schedule", "inference_time_sec",
    "clip_score", "fid_score",
]
SUMMARY_COLUMNS = [
    "batch", "sampler", "steps", "schedule", "n_images", "inference_time_sec",
    "inference_time_std_sec", "inference_time_median_sec", "fid_score",
    "clip_score", "timing_gpu", "latency_status",
]
SKIP_DIRS = {"cache", ".cache", "__pycache__", ".ipynb_checkpoints"}
# Breakdown files and the checksum manifest live here; handoff files stay in results/.
DETAILS_DIR = "evaluation_details"


class ConsolidationError(ValueError):
    """A missing, incomplete or conflicting input; outputs remain untouched."""


def fail(message):
    raise ConsolidationError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path, required):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(required).issubset(fields):
            fail(f"{path}: invalid CSV header; required columns: {', '.join(required)}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        fail(f"{path}: malformed or truncated CSV row")
    if not rows:
        fail(f"{path}: empty CSV")
    return rows


def integer(value, context):
    try:
        if isinstance(value, bool) or str(value).strip() != str(int(value)):
            fail(f"{context}: expected an integer, got {value!r}")
        return int(value)
    except (TypeError, ValueError) as error:
        raise ConsolidationError(f"{context}: expected an integer, got {value!r}") from error


def number(value, context, *, minimum=None, maximum=None):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConsolidationError(f"{context}: missing/invalid number {value!r}") from error
    if (not math.isfinite(result) or minimum is not None and result < minimum
            or maximum is not None and result > maximum):
        fail(f"{context}: nonfinite or out-of-range number {value!r}")
    return result


def same_number(actual, expected, context):
    value = number(actual, context)
    if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-8):
        fail(f"{context}: {value!r} disagrees with joined raw values ({expected!r})")


def add_unique(mapping, key, row, context):
    if key in mapping and mapping[key] != row:
        fail(f"{context}: conflicting duplicate for {key!r}; inspect sources instead of choosing one")
    mapping[key] = row


def check_ids(rows, ids, context):
    actual = set(rows)
    if actual != ids:
        fail(f"{context}: expected {EXPECTED_COUNT} matching prompt IDs; "
             f"missing={sorted(ids - actual)[:10]}, extra={sorted(actual - ids)[:10]}")


def is_link(path):
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        return True
    # Python 3.10/3.11 lack Path.is_junction(); Windows readlink also recognises
    # directory junctions. Ordinary files and cloud placeholders raise OSError.
    if os.name == "nt":
        try:
            os.readlink(path)
            return True
        except OSError:
            pass
    return False


def checked_path(root, path):
    """Reject linked child paths as well as targets escaping the selected root."""
    try:
        relative = path.relative_to(root)
        path.resolve(strict=True).relative_to(root)
    except (ValueError, OSError) as error:
        raise ConsolidationError(f"Input/output must remain inside the project: {path}") from error
    current = root
    for part in relative.parts:
        current /= part
        if is_link(current):
            fail(f"Refusing linked child path (symlink/junction): {current}")
    return path


def check_results_path(root):
    results = root / "results"
    if is_link(results):
        fail(f"Refusing to publish into a symlink/junction results directory: {results}")
    try:
        results.resolve(strict=True).relative_to(root)
    except (ValueError, OSError) as error:
        raise ConsolidationError("results/ must resolve inside the selected project root") from error


def check_active_locks(root):
    # The generation helper's lock is a DIRECTORY, not just a regular file.
    locks = sorted(path for path in root.rglob("*.lock")
                   if any(token in path.name.lower() for token in ("generation", "evaluation", "cleanup")))
    if locks:
        fail("A generation/evaluation/cleanup lock exists; wait for the job to finish. Do not delete the lock: "
             + ", ".join(str(path.relative_to(root)) for path in locks))


def discover(root):
    """Find completed per-image CSVs and canonical summaries, never cache chunks."""
    clip_paths, summary_paths = [], []
    for directory, subdirs, filenames in os.walk(root / "results", followlinks=False):
        subdirs[:] = sorted(name for name in subdirs
                           if name not in SKIP_DIRS and not name.startswith("preflight")
                           and "backup" not in name and "staging" not in name
                           and not is_link(Path(directory) / name))
        for name in sorted(filenames):
            path = Path(directory) / name
            if is_link(path):
                continue
            if name.endswith("_clip_per_image.csv"):
                clip_paths.append(path)
            elif name in {"summary.csv", "metrics.csv"}:
                summary_paths.append(path)
    if not clip_paths:
        fail("No *_clip_per_image.csv found in results/. Run the evaluation notebook first.")
    if not summary_paths:
        fail("No completed summary.csv or metrics.csv found in results/.")
    return clip_paths, summary_paths


def read_manifest(root):
    path = root / "data" / "captions.json"
    with path.open(encoding="utf-8-sig") as stream:
        records = json.load(stream)
    if not isinstance(records, list) or len(records) != EXPECTED_COUNT:
        fail(f"{path}: expected exactly {EXPECTED_COUNT} caption records")
    by_id = {}
    for record in records:
        if not isinstance(record, dict):
            fail(f"{path}: caption records must be objects")
        pid = integer(record.get("prompt_id"), "caption prompt_id")
        iid = integer(record.get("image_id"), f"caption {pid} image_id")
        seed = integer(record.get("seed"), f"caption {pid} seed")
        caption = record.get("caption")
        if not isinstance(caption, str) or not caption or seed != 1000 + pid:
            fail(f"caption {pid}: missing caption or seed != 1000 + prompt_id")
        if pid in by_id:
            fail(f"{path}: duplicate prompt_id {pid}")
        by_id[pid] = dict(prompt_id=pid, image_id=iid, seed=seed, caption=caption)
    return by_id, path


def read_summaries(paths, caption_hash):
    by_batch = {}
    for path in paths:
        rows = load_csv(path, ["batch", "sampler", "schedule", "steps", "status",
                               "n_real", "n_generated", "fid_clean", "clip_score_100"])
        for row in rows:
            batch = row["batch"]
            if batch not in BATCHES:
                continue
            source_sampler, _, schedule, steps = BATCHES[batch]
            if (row["sampler"] != source_sampler or row["schedule"] != schedule
                    or integer(row["steps"], f"{batch} steps") != steps):
                fail(f"{path}: incorrect solver/schedule/steps metadata for {batch}")
            if row["status"] != "evaluated":
                fail(f"{path}: {batch} is {row['status']!r}; FID and CLIP must both be completed")
            if any(integer(row[field], f"{batch} {field}") != EXPECTED_COUNT
                   for field in ("n_real", "n_generated")):
                fail(f"{batch}: FID was not calculated on exactly 200 real and 200 generated images")
            if row.get("manifest_sha256") != caption_hash:
                fail(f"{batch}: summary caption manifest hash does not match data/captions.json")
            number(row["fid_clean"], f"{batch} FID", minimum=0)
            number(row["clip_score_100"], f"{batch} CLIP", minimum=0, maximum=100.001)
            add_unique(by_batch, batch, row, str(path))
    if set(by_batch) != set(BATCHES):
        fail(f"Missing completed batch summaries: {', '.join(sorted(set(BATCHES) - set(by_batch)))}")
    return by_batch


def read_clip(paths, records):
    by_batch = {batch: {} for batch in BATCHES}
    for path in paths:
        rows = load_csv(path, ["batch", "sampler", "steps", "prompt_id", "image_id",
                               "seed", "image_file", "caption", "clip_raw_100cos"])
        for row in rows:
            batch = row["batch"]
            if batch not in BATCHES:
                fail(f"{path}: unexpected batch {batch!r} in per-image CSV")
            source_sampler, _, _, steps = BATCHES[batch]
            pid = integer(row["prompt_id"], f"{path} prompt_id")
            if pid not in records:
                fail(f"{path}: unknown prompt_id {pid}")
            record = records[pid]
            if (row["sampler"] != source_sampler or integer(row["steps"], "CLIP steps") != steps
                    or integer(row["image_id"], "CLIP image_id") != record["image_id"]
                    or integer(row["seed"], "CLIP seed") != record["seed"]
                    or row["caption"] != record["caption"]
                    or row["image_file"] != f"{pid:04d}.png"):
                fail(f"{path}: caption/image/seed/batch mismatch for {batch}/{pid}")
            number(row["clip_raw_100cos"], f"{batch}/{pid} CLIP", minimum=-100.001, maximum=100.001)
            add_unique(by_batch[batch], pid, row, str(path))
    for batch, rows in by_batch.items():
        check_ids(rows, set(records), f"{batch} CLIP")
    return by_batch


def read_timings(root, records, summaries, caption_hash):
    path = root / "results" / "generation_log.csv"
    rows = load_csv(path, ["prompt_id", "image_id", "prompt", "seed", "sampler", "steps",
                           "cfg", "height", "width", "gen_time_s", "image_path"])
    batch_for_run = {(value[0], value[3]): batch for batch, value in BATCHES.items()}
    by_batch = {batch: {} for batch in BATCHES}
    config_paths = []
    for row in rows:
        key = (row["sampler"], integer(row["steps"], "generation steps"))
        batch = batch_for_run.get(key)
        if batch is None:
            continue
        pid = integer(row["prompt_id"], f"{batch} timing prompt_id")
        if pid not in records:
            fail(f"generation log: unknown prompt_id {pid} in {batch}")
        record = records[pid]
        image_path = row["image_path"].replace("\\", "/")
        if (integer(row["seed"], "timing seed") != record["seed"]
                or integer(row["image_id"], "timing image_id") != record["image_id"]
                or row["prompt"] != record["caption"]
                or not image_path.endswith(f"results/images/{batch}/{pid:04d}.png")):
            fail(f"generation log: caption/image/seed mismatch for {batch}/{pid}")
        for field, expected in (("cfg", 7.5), ("height", 512), ("width", 512)):
            same_number(row[field], expected, f"{batch}/{pid} timing {field}")
        duration = number(row["gen_time_s"], f"{batch}/{pid} inference time", minimum=0)
        if duration <= 0:
            fail(f"{batch}/{pid}: inference time must be positive; no mean-time imputation allowed")
        add_unique(by_batch[batch], pid, row, str(path))
    for batch, indexed in by_batch.items():
        check_ids(indexed, set(records), f"{batch} inference times")
        source_sampler, _, schedule, steps = BATCHES[batch]
        config_path = root / "results" / "images" / batch / "run_config.json"
        with config_path.open(encoding="utf-8-sig") as stream:
            config = json.load(stream)
        gpu = str(config.get("gpu", config.get("gpu_name", "")))
        if (config.get("sampler") != source_sampler or config.get("steps") != steps
                or config.get("cfg") != 7.5 or config.get("height") != 512
                or config.get("width") != 512 or config.get("dtype") != "float16"
                or config.get("model_id") != "stable-diffusion-v1-5/stable-diffusion-v1-5"):
            fail(f"{config_path}: generation configuration does not match the coursework setup")
        if "T4" not in gpu.upper():
            fail(f"{batch}: recorded GPU is {gpu!r}; coursework latency requires NVIDIA T4")
        if any(config.get("versions", {}).get(key) != value for key, value in BASELINE_VERSIONS.items()):
            fail(f"{batch}: generation software versions differ from Member A's timing environment")
        scope = str(config.get("timing_scope", ""))
        if "pipeline call only" not in scope or "warm-up" not in scope:
            fail(f"{batch}: pipeline timing scope/warm-up is not recorded")
        if config.get("caption_sha256", caption_hash) != caption_hash:
            fail(f"{batch}: run configuration caption hash mismatch")
        karras = config.get("scheduler_config", {}).get("use_karras_sigmas", False)
        if batch != "ddim_s30" and bool(karras) != (schedule == "karras"):
            fail(f"{batch}: saved scheduler use_karras_sigmas contradicts batch metadata")
        summary = summaries[batch]
        if summary.get("generation_config_sha256") != sha256_file(config_path):
            fail(f"{batch}: run_config.json changed after evaluation; re-evaluate provenance")
        if summary.get("latency_status") != "eligible_T4" or "T4" not in summary.get("timing_gpu", "").upper():
            fail(f"{batch}: completed summary does not mark the latency as eligible_T4")
        times = [float(row["gen_time_s"]) for row in indexed.values()]
        same_number(summary.get("coursework_latency_mean_s"), statistics.mean(times), f"{batch} latency mean")
        same_number(summary.get("reported_latency_mean_s"), statistics.mean(times), f"{batch} reported mean")
        config_paths.append(config_path)
    return by_batch, [path, *config_paths]


def csv_text(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def markdown_table(rows, *, steps_only=False):
    labels = {"ddim": "DDIM", "euler": "Euler", "dpmpp": "DPM++ 2M"}
    schedules = {"native": "Native", "non_karras": "No Karras", "karras": "Karras"}
    header = ("| Steps | Latency (s/image) | FID ↓ | CLIP ↑ |\n"
              "| ---: | ---: | ---: | ---: |\n" if steps_only else
              "| Sampler | Schedule | Steps | Latency (s/image) | FID ↓ | CLIP ↑ |\n"
              "| --- | --- | ---: | ---: | ---: | ---: |\n")
    lines = [header.rstrip()]
    for row in rows:
        prefix = f"{row['steps']}" if steps_only else (
            f"{labels[row['sampler']]} | {schedules[row['schedule']]} | {row['steps']}")
        lines.append(f"| {prefix} | {row['inference_time_sec']:.3f} | "
                     f"{row['fid_score']:.3f} | {row['clip_score']:.3f} |")
    return "\n".join(lines)


def tables_text(summary):
    by_batch = {row["batch"]: row for row in summary}
    sampler_order = ["ddim_s30", "euler_s30", "euler_karras_s30", "dpmpp_nokarras_s30", "dpmpp_s30"]
    step_order = [f"dpmpp_s{s}" for s in (10, 20, 30, 40, 50)]
    return (
        "# Member B consolidated evaluation results\n\n"
        "## Table 1: Sampling strategy comparison and 2×2 solver/schedule ablation\n\n"
        + markdown_table([by_batch[batch] for batch in sampler_order])
        + "\n\nThe four Euler/DPM++ rows form the 2×2 factorial comparison; DDIM is an additional baseline.\n\n"
        "## Table 2: DPM++ 2M Karras sampling-step ablation\n\n"
        + markdown_table([by_batch[batch] for batch in step_order], steps_only=True)
        + "\n\n## Definitions and interpretation\n\n"
        "- All nine configurations use the same 200 COCO prompt–image pairs, seeds 1000 + prompt_id, "
        "SD v1.5, CFG 7.5, 512×512 and fp16. Sampler identifies the solver; schedule identifies Karras use.\n"
        "- `evaluation_raw_results.csv` has 1,800 rows: one row per prompt/configuration. "
        "`inference_time_sec` is that prompt's recorded original-generation pipeline call on NVIDIA T4; "
        "it includes text encoding, denoising and VAE decoding, after warm-up, and excludes loading/saving. "
        "No per-prompt latency is inferred from batch averages.\n"
        "- Raw `clip_score` is the saved unclamped `clip_raw_100cos` (100× cosine similarity of normalised "
        "CLIP ViT-B/32 embeddings). Summary/table CLIP is max(mean(raw CLIP), 0), matching the evaluation protocol.\n"
        "- `fid_score` is clean-fid/clean/Inception-v3/2048 for the entire 200-image batch against the "
        "same 200 reference images. It is repeated on raw rows only to support joins; it is not a per-image FID. "
        "Use the nine-row summary for FID comparisons; do not average repeated FID across configurations.\n"
        "- Summary latency is the mean of the 200 matching original timings. The separately saved "
        "20-prompt same-session timing recheck is supplementary evidence and is not pooled into these tables.\n"
        "- With 200 images and one seed per prompt, FID is a biased exploratory estimate. Small differences "
        "do not establish general superiority. The original SD checkpoint revision was not recorded; "
        "the retained metric provenance states this reproducibility limitation.\n"
        "- Source CSVs, metric provenance and run configurations are preserved. "
        "`evaluation_manifest.json` records input/output checksums and is written last after successful validation.\n"
    )


def atomic_outputs(results, contents, manifest):
    """Stage everything first; publish independent atomic files, then the manifest."""
    temporary = []
    try:
        for name, content in contents.items():
            destination = results / name
            if is_link(destination):
                fail(f"Refusing to replace symlink output: {destination}")
            if destination.is_file() and destination.read_bytes() == content.encode("utf-8"):
                continue  # Already current; replacing it would only churn the shared folder.
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=results,
                                             prefix=f".{name}.", suffix=".tmp", delete=False) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                temporary.append((Path(stream.name), destination))
        details = results / DETAILS_DIR
        manifest_path = details / "evaluation_manifest.json"
        if is_link(details) or is_link(manifest_path):
            fail(f"Refusing to replace symlink output: {manifest_path}")
        details.mkdir(exist_ok=True)
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=details,
                                         prefix=".evaluation_manifest.", suffix=".tmp", delete=False) as stream:
            stream.write(manifest_text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary.append((Path(stream.name), manifest_path))
        for source, destination in temporary:
            os.replace(source, destination)
    finally:
        for source, _ in temporary:
            source.unlink(missing_ok=True)


def consolidate(project_root, *, check_only=False):
    root = Path(project_root).expanduser().resolve(strict=True)
    results = root / "results"
    if not root.is_dir() or not (root / "data" / "captions.json").is_file() or not results.is_dir():
        fail(f"{root}: project root must contain data/captions.json and results/")
    check_results_path(root)
    check_active_locks(root)
    checked_path(root, root / "data" / "captions.json")
    records, captions_path = read_manifest(root)
    caption_hash = sha256_file(captions_path)
    clip_paths, summary_paths = discover(root)
    config_paths = [results / "images" / batch / "run_config.json" for batch in BATCHES]
    provenance_paths = sorted(results.rglob("metric_provenance.json"))
    source_paths = {captions_path, *clip_paths, *summary_paths, *config_paths,
                    results / "generation_log.csv"}
    source_paths.update(path for path in provenance_paths
                        if not any(part in SKIP_DIRS or part.startswith("preflight") for part in path.parts))
    for path in source_paths:
        checked_path(root, path)
    before_hashes = {path: sha256_file(path) for path in source_paths}
    summaries = read_summaries(summary_paths, caption_hash)
    clips = read_clip(clip_paths, records)
    timings, _ = read_timings(root, records, summaries, caption_hash)
    raw, summary = [], []
    for batch, (_, sampler, schedule, steps) in BATCHES.items():
        source = summaries[batch]
        fid = float(source["fid_clean"])
        values = [float(clips[batch][pid]["clip_raw_100cos"]) for pid in sorted(records)]
        clip_mean = statistics.mean(values)
        same_number(source["clip_score_100"], max(clip_mean, 0), f"{batch} aggregate CLIP")
        same_number(source.get("clip_raw_mean_100"), clip_mean, f"{batch} raw mean CLIP")
        times = [float(timings[batch][pid]["gen_time_s"]) for pid in sorted(records)]
        row = dict(batch=batch, sampler=sampler, steps=steps, schedule=schedule,
                   n_images=EXPECTED_COUNT, inference_time_sec=statistics.mean(times),
                   inference_time_std_sec=statistics.stdev(times),
                   inference_time_median_sec=statistics.median(times), fid_score=fid,
                   clip_score=max(clip_mean, 0), timing_gpu=source["timing_gpu"], latency_status="eligible_T4")
        summary.append(row)
        for pid, value, duration in zip(sorted(records), values, times):
            raw.append(dict(prompt_id=pid, sampler=sampler, steps=steps, schedule=schedule,
                            inference_time_sec=duration, clip_score=value, fid_score=fid))
    contents = {
        "evaluation_raw_results.csv": csv_text(raw, RAW_COLUMNS),
        "evaluation_summary.csv": csv_text(summary, SUMMARY_COLUMNS),
        "evaluation_summary.md": tables_text(summary),
    }
    after_hashes = {path: sha256_file(path) for path in source_paths}
    if before_hashes != after_hashes:
        fail("An input changed during consolidation; wait for the evaluation job to finish and rerun")
    manifest = dict(
        schema_version=1, status="complete", generated_utc=datetime.now(timezone.utc).isoformat(),
        expected_batch_count=len(BATCHES), expected_prompt_count=EXPECTED_COUNT,
        raw_rows=len(raw), summary_rows=len(summary),
        raw_key=["prompt_id", "sampler", "steps", "schedule"],
        raw_columns=RAW_COLUMNS, summary_columns=SUMMARY_COLUMNS,
        batch_definitions={batch: dict(source_sampler=value[0], sampler=value[1], schedule=value[2], steps=value[3])
                           for batch, value in BATCHES.items()},
        definitions=dict(fid="batch-level clean-fid score, repeated on raw rows; not a per-image FID",
                         clip_raw="unclamped 100*cos normalised embeddings",
                         clip_summary="max(mean(raw clip_score),0)",
                         latency="original per-prompt 200-image T4 generation log; not the 20-prompt recheck"),
        sources=[dict(path=path.relative_to(root).as_posix(), sha256=after_hashes[path])
                 for path in sorted(source_paths)],
        outputs={name: dict(sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                            **({"rows": len(raw)} if name.endswith("raw_results.csv") else
                               {"rows": len(summary)} if name.endswith("summary.csv") else {}))
                 for name, content in contents.items()},
    )
    if not check_only:
        check_results_path(root)
        check_active_locks(root)
        atomic_outputs(results, contents, manifest)
    return manifest


def default_root():
    cwd = Path.cwd()
    if (cwd / "data" / "captions.json").is_file():
        return cwd
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "src" else script_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=default_root())
    parser.add_argument("--check-only", action="store_true", help="Validate inputs; do not write or replace anything")
    args = parser.parse_args(argv)
    try:
        manifest = consolidate(args.project_root, check_only=args.check_only)
    except (ConsolidationError, OSError, json.JSONDecodeError) as error:
        print(f"Consolidation failed: {error}", file=sys.stderr)
        return 1
    verb = "Validated" if args.check_only else "Wrote"
    print(f"{verb}: {manifest['raw_rows']} per-prompt rows across {manifest['summary_rows']} batches.")
    if not args.check_only:
        results = Path(args.project_root).resolve() / "results"
        for name in manifest["outputs"]:
            print(results / name)
        print(results / DETAILS_DIR / "evaluation_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
