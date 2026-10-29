"""CA6127 Member B evaluation, usable from a local notebook or the CLI.

Protocol: clean-fid Inception-v3/2048, clean resizing, sample covariance;
CLIP ViT-B/32, float32 normalized embeddings, 100*cos per matched pair and
max(mean, 0) for the folder score. This matches the TorchMetrics aggregation.
Official references:
https://github.com/GaParmar/clean-fid
https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/multimodal/clip_score.py
https://huggingface.co/openai/clip-vit-base-patch32

No generation is performed here. Non-T4 generation timings are retained but
excluded from the coursework latency column. With only 200 images, FID is an
exploratory, biased small-sample estimate; avoid claims about tiny differences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from datetime import datetime, timezone


EXPECTED_COUNT = 200
CLIP_MODEL = "openai/clip-vit-base-patch32"
# Immutable official repository commit, verified against the Hugging Face
# commit page: https://huggingface.co/openai/clip-vit-base-patch32/commit/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASELINE_VERSIONS = {"torch": "2.11.0+cu130", "diffusers": "0.40.0", "transformers": "5.18.0"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
BATCHES = {
    "ddim_s30": ("ddim", "native", 30),
    "euler_s30": ("euler", "non_karras", 30),
    "dpmpp_s30": ("dpmpp", "karras", 30),
    "euler_karras_s30": ("euler_karras", "karras", 30),
    "dpmpp_nokarras_s30": ("dpmpp_nokarras", "non_karras", 30),
    **{f"dpmpp_s{s}": ("dpmpp", "karras", s) for s in (10, 20, 40, 50)},
}
SUMMARY_COLUMNS = [
    "batch", "sampler", "schedule", "steps", "status", "n_real", "n_generated",
    "fid_clean", "clip_score_100", "clip_raw_mean_100", "clip_std_100",
    "captions_truncated", "reported_latency_mean_s", "reported_latency_std_s",
    "reported_latency_median_s", "coursework_latency_mean_s", "timing_gpu",
    "latency_status", "manifest_sha256", "image_content_sha256", "metric_key",
    "generation_gpu", "generation_config_sha256", "generation_environment_matches_baseline",
    "generation_model_revision", "generation_model_revision_unknown",
    "generation_model_revision_status", "generation_revision_adoption_sha256",
    "generation_weights_match_baseline", "baseline_model_revision",
]


def exact_revision(config):
    """Only an explicitly recorded immutable commit can establish revision identity."""
    value = (config or {}).get("model_revision")
    if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value):
        return value.lower()
    return None


def fid_checkpoint_key(protocol, reference_hash, image_hash):
    return hash_object(dict(kind="fid_numeric_v1", protocol=protocol,
                            reference=reference_hash, images=image_hash))


def clip_checkpoint_key(protocol, manifest_hash, image_hash, records):
    return hash_object(dict(kind="clip_chunks_v1", protocol=protocol,
                            manifest=manifest_hash, images=image_hash,
                            prompt_ids=[record["prompt_id"] for record in records]))


def validate_fid_cache(values, key):
    if (values.get("schema_version") != 1 or values.get("fid_key") != key
            or not isinstance(values.get("fid_clean"), (int, float))
            or isinstance(values.get("fid_clean"), bool)
            or not math.isfinite(values["fid_clean"])):
        raise ValueError("Invalid independent FID checkpoint; inspect or remove the named cache")
    return float(values["fid_clean"])


def clip_values(scores):
    import numpy as np
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.shape != (EXPECTED_COUNT,) or not np.isfinite(scores).all()
            or np.any(np.abs(scores) > 100.001)):
        raise ValueError("Incomplete, out-of-range or nonfinite CLIP scores")
    return dict(clip_score_100=max(float(scores.mean()), 0.0),
                clip_raw_mean_100=float(scores.mean()),
                clip_std_100=float(scores.std(ddof=1)))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = stream.name
    os.replace(temporary, path)


def atomic_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="",
                                     dir=path.parent, delete=False, suffix=".tmp") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = stream.name
    os.replace(temporary, path)


def atomic_arrays(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False,
                                     suffix=".tmp") as stream:
        np.savez_compressed(stream, **arrays)
        temporary = stream.name
    os.replace(temporary, path)


def package_versions():
    versions = {}
    for name in ("clean-fid", "torch", "torchvision", "transformers", "numpy",
                 "scipy", "Pillow", "tokenizers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def load_manifest(project_root):
    """Validate the complete paired study. Returns records in prompt-ID order."""
    root = Path(project_root).resolve()
    path = root / "data" / "captions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else data.get("records", data.get("captions"))
    if not isinstance(records, list) or len(records) != EXPECTED_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_COUNT} caption records in {path}")
    for field in ("prompt_id", "image_id", "seed"):
        values = [record.get(field) for record in records]
        if any(type(value) is not int for value in values) or len(set(values)) != len(values):
            raise ValueError(f"Manifest {field} must contain unique integer values")
    if set(record["prompt_id"] for record in records) != set(range(EXPECTED_COUNT)):
        raise ValueError("Expected prompt IDs 0 through 199")
    for record in records:
        if not isinstance(record.get("caption"), str) or not record["caption"].strip():
            raise ValueError(f"Empty caption for prompt {record['prompt_id']}")
        if record["seed"] != 1000 + record["prompt_id"]:
            raise ValueError(f"Unexpected seed for prompt {record['prompt_id']}")
    return sorted(records, key=lambda record: record["prompt_id"]), sha256_file(path)


def reference_paths(project_root, records):
    root = Path(project_root).resolve()
    paths = []
    for record in records:
        value = record.get("file_name") or record.get("filename")
        if value:
            path = root / "data" / "images" / Path(str(value)).name
        elif record.get("image_path"):
            path = root / record["image_path"]
        else:
            path = root / "data" / "images" / f"{record['image_id']:012d}.jpg"
        path = path.resolve()
        if not path.is_relative_to((root / "data" / "images").resolve()):
            raise ValueError(f"Reference path escapes data/images: {path}")
        paths.append(path)
    if len(set(paths)) != EXPECTED_COUNT:
        raise ValueError("Reference files are not unique")
    actual = {path.resolve() for path in (root / "data" / "images").rglob("*")
              if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    if actual != set(paths):
        raise ValueError(f"Reference file set mismatch: {len(set(paths) - actual)} missing, "
                         f"{len(actual - set(paths))} unexpected")
    return paths


def validate_images(paths, generated=False):
    """Decode every image fully; CLIP and clean-fid both use explicit RGB conversion."""
    from PIL import Image
    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
                if generated and (image.format != "PNG" or image.size != (512, 512)):
                    raise ValueError("Generated images must be 512x512 PNG files")
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("Image has invalid dimensions")
                image.convert("RGB").load()
        except Exception as error:
            raise ValueError(f"Cannot validate {path}: {error}") from error


def content_hash(paths):
    return hash_object([(path.name, sha256_file(path)) for path in paths])


def validate_batch(folder, records):
    folder = Path(folder)
    if not folder.exists():
        return None
    if not folder.is_dir():
        raise ValueError(f"Batch is not a directory: {folder}")
    expected = [folder / f"{record['prompt_id']:04d}.png" for record in records]
    actual = {path.resolve() for path in folder.rglob("*")
              if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    if actual != {path.resolve() for path in expected}:
        missing = [path.name for path in expected if not path.exists()]
        raise ValueError(f"Invalid batch {folder.name}: expected exactly 200 named PNGs; "
                         f"missing={missing[:8]}, found={len(actual)}")
    validate_images(expected, generated=True)
    return expected


def read_run_config(folder):
    path = Path(folder) / "run_config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def validate_config(config, sampler, steps):
    """Check recorded generation controls whenever metadata is available."""
    if config is None:
        return
    checks = {"model_id": SD_MODEL, "sampler": sampler, "steps": steps, "cfg": 7.5,
              "height": 512, "width": 512, "dtype": "float16"}
    for key, expected in checks.items():
        if config.get(key) != expected:
            raise ValueError(f"run_config {key}={config.get(key)!r}; expected {expected!r}")
    scheduler = config.get("scheduler_config", {})
    expected_class = ("DPMSolverMultistepScheduler" if sampler.startswith("dpmpp") else
                      "EulerDiscreteScheduler" if sampler.startswith("euler") else "DDIMScheduler")
    if config.get("scheduler_class") != expected_class:
        raise ValueError(f"Unexpected scheduler class for {sampler}")
    if sampler.startswith("dpmpp"):
        if scheduler.get("algorithm_type") != "dpmsolver++" or scheduler.get("solver_order") != 2:
            raise ValueError("Expected second-order DPM++ solver")
    if sampler != "ddim":
        expected_karras = sampler in {"dpmpp", "euler_karras"}
        if bool(scheduler.get("use_karras_sigmas", False)) != expected_karras:
            raise ValueError(f"Unexpected Karras setting for {sampler}")


def generation_latency(project_root, folder, records, sampler, steps):
    """Accept the T4 coursework timing only after checking manifest/config/log joins."""
    result = {"latency_status": "no_generation_log"}
    log = Path(project_root) / "results" / "generation_log.csv"
    if not log.exists():
        return result
    with log.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row.get("sampler") == sampler and int(row.get("steps", -1)) == steps]
    if not rows:
        return result
    if len(rows) != EXPECTED_COUNT:
        raise ValueError(f"Expected exactly 200 timing rows for {sampler}/{steps}; got {len(rows)}")
    by_prompt = {int(row["prompt_id"]): row for row in rows}
    if len(by_prompt) != EXPECTED_COUNT or set(by_prompt) != {r["prompt_id"] for r in records}:
        raise ValueError(f"Duplicate/mismatched timing prompt IDs for {sampler}/{steps}")
    for record in records:
        row = by_prompt[record["prompt_id"]]
        if int(row["seed"]) != record["seed"]:
            raise ValueError(f"Timing seed mismatch for {sampler}/{steps}/{record['prompt_id']}")
        if row.get("image_id") and int(row["image_id"]) != record["image_id"]:
            raise ValueError("Timing image_id mismatch")
        for field, expected in (("cfg", 7.5), ("height", 512), ("width", 512)):
            if float(row.get(field, "nan")) != expected:
                raise ValueError(f"Timing {field} mismatch for {sampler}/{steps}")
        if row.get("prompt") != record["caption"]:
            raise ValueError("Timing prompt does not match the manifest caption")
        image_path = str(row.get("image_path", "")).replace("\\", "/")
        if Path(image_path).name != f"{record['prompt_id']:04d}.png":
            raise ValueError("Timing image filename mismatch")
    config = read_run_config(folder)
    validate_config(config, sampler, steps)
    gpu = str((config or {}).get("gpu", (config or {}).get("gpu_name", "unknown")))
    result["timing_gpu"] = gpu
    durations = [row.get("gen_time_s", "").strip() for row in rows]
    if not any(durations):
        result["latency_status"] = "not_benchmarked_quality_only"
        return result
    if not all(durations):
        raise ValueError(f"Mixed timed/untimed rows for {sampler}/{steps}")
    times = [float(value) for value in durations]
    if any(not math.isfinite(value) or value <= 0 for value in times):
        raise ValueError(f"Invalid timing measurements for {sampler}/{steps}")
    result.update(reported_latency_mean_s=statistics.mean(times),
                  reported_latency_std_s=statistics.stdev(times),
                  reported_latency_median_s=statistics.median(times), timing_gpu=gpu)
    required = {"sampler", "steps", "model_id", "cfg", "height", "width", "timing_scope", "versions"}
    if config is None or not required.issubset(config):
        result["latency_status"] = "excluded_incomplete_run_config"
    elif "T4" not in gpu.upper():
        result["latency_status"] = "excluded_non_T4_or_unknown_gpu"
    elif any(config.get("versions", {}).get(key) != value for key, value in BASELINE_VERSIONS.items()):
        result["latency_status"] = "excluded_generation_environment_differs_from_baseline"
    elif "pipeline call only" not in config["timing_scope"] or "warm-up" not in config["timing_scope"]:
        result["latency_status"] = "excluded_inconsistent_timing_scope"
    else:
        result["latency_status"] = "eligible_T4"
        result["coursework_latency_mean_s"] = result["reported_latency_mean_s"]
    return result


class MetricEngine:
    """Lazy model initialization; one Inception and one CLIP model per evaluation."""

    def __init__(self, cache_dir, device, batch_size, protocol):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.protocol = protocol
        self.inception = self.clip = self.processor = None
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def inception_features(self, paths, image_hash):
        import numpy as np
        from cleanfid import fid
        from cleanfid.features import build_feature_extractor
        key = hash_object({"protocol": self.protocol, "images": image_hash, "kind": "inception"})
        path = self.cache_dir / f"inception_{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                features = archive["features"]
        else:
            if self.inception is None:
                # clean-fid stores its TorchScript weight file in cwd on Windows.
                original = Path.cwd()
                weights = self.cache_dir / "inception_weights"
                weights.mkdir(parents=True, exist_ok=True)
                try:
                    os.chdir(weights)
                    self.inception = build_feature_extractor("clean", self.device,
                                                              use_dataparallel=False)
                finally:
                    os.chdir(original)
            with self.torch.inference_mode():
                features = fid.get_files_features(
                    [str(path) for path in paths], self.inception,
                    num_workers=0, batch_size=self.batch_size, device=self.device,
                    mode="clean", description="Inception features")
            atomic_arrays(path, features=features)
        if features.shape != (EXPECTED_COUNT, 2048) or not np.isfinite(features).all():
            raise ValueError(f"Invalid Inception feature cache: {path}")
        return features

    def load_clip(self):
        if self.clip is None:
            from transformers import CLIPModel, CLIPProcessor
            if not re.fullmatch(r"[0-9a-f]{40}", CLIP_REVISION):
                raise ValueError("CLIP_REVISION must be the verified immutable repository commit")
            self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)
            self.clip = CLIPModel.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION).float().to(self.device).eval()

    def text_features(self, records, manifest_hash):
        import numpy as np
        key = hash_object({"protocol": self.protocol, "manifest": manifest_hash, "kind": "clip_text"})
        path = self.cache_dir / f"clip_text_{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                values, truncated, counts = archive["features"], archive["truncated"], archive["token_counts"]
            if values.shape != (EXPECTED_COUNT, 512) or not np.isfinite(values).all():
                raise ValueError(f"Invalid CLIP text feature cache: {path}")
            if truncated.shape != (EXPECTED_COUNT,) or counts.shape != (EXPECTED_COUNT,):
                raise ValueError(f"Invalid CLIP token metadata cache: {path}")
            return values, truncated, counts
        self.load_clip()
        tokenizer = self.processor.tokenizer
        max_length = int(self.clip.config.text_config.max_position_embeddings)
        if max_length != 77:
            raise ValueError(f"Unexpected CLIP text context length: {max_length}")
        counts = np.array([len(tokenizer(record["caption"], truncation=False)["input_ids"])
                           for record in records], dtype=np.int32)
        features = []
        for offset in range(0, len(records), self.batch_size):
            captions = [record["caption"] for record in records[offset:offset + self.batch_size]]
            # Tokenizer truncation retains special tokens, including the final EOS.
            tokens = tokenizer(captions, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt")
            ids, mask = tokens["input_ids"], tokens["attention_mask"]
            for index in range(len(captions)):
                if ids[index, int(mask[index].sum()) - 1].item() != tokenizer.eos_token_id:
                    raise ValueError("CLIP truncation did not preserve final EOS")
            with self.torch.inference_mode():
                embeddings = self.clip.get_text_features(**{k: v.to(self.device) for k, v in tokens.items()})
                # Transformers 5 returns projected pooler_output; older releases return a tensor.
                embeddings = getattr(embeddings, "pooler_output", embeddings)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            features.append(embeddings.cpu().numpy())
        values = np.concatenate(features)
        atomic_arrays(path, features=values, truncated=counts > max_length, token_counts=counts)
        return values, counts > max_length, counts

    def _clip_score_chunk(self, paths, text_features):
        from PIL import Image
        self.load_clip()
        images = []
        try:
            for path in paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            processed = self.processor(images=images, return_tensors="pt")
            with self.torch.inference_mode():
                embeddings = self.clip.get_image_features(pixel_values=processed["pixel_values"].to(self.device))
                embeddings = getattr(embeddings, "pooler_output", embeddings)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                target = self.torch.from_numpy(text_features).to(self.device)
                return (100 * (embeddings * target).sum(dim=-1)).cpu().tolist()
        finally:
            for image in images:
                image.close()

    def clip_scores(self, paths, text_features, records, image_hash, manifest_hash):
        """Commit each CLIP image chunk; reuse only verified matching ordered pairs."""
        import numpy as np
        if len(paths) != EXPECTED_COUNT or len(records) != EXPECTED_COUNT:
            raise ValueError("CLIP checkpoint requires the complete ordered study")
        key = clip_checkpoint_key(self.protocol, manifest_hash, image_hash, records)
        folder = self.cache_dir / f"clip_chunks_{key}"
        folder.mkdir(parents=True, exist_ok=True)
        values = np.full(EXPECTED_COUNT, np.nan, dtype=np.float64)
        prompt_ids = [record["prompt_id"] for record in records]
        # Only atomically committed .json files are candidates. A killed writer's
        # .tmp file is ignored; no unfinished chunk is treated as a result.
        for path in sorted(folder.glob("chunk_*.json")):
            chunk = json.loads(path.read_text(encoding="utf-8"))
            digest = chunk.get("chunk_sha256")
            payload = {k: v for k, v in chunk.items() if k != "chunk_sha256"}
            offset, stop = chunk.get("offset"), chunk.get("stop")
            if (type(offset) is not int or type(stop) is not int
                    or not 0 <= offset < stop <= EXPECTED_COUNT):
                raise ValueError(f"Invalid CLIP checkpoint range: {path}")
            scores = np.asarray(chunk.get("scores", []), dtype=np.float64)
            if (digest != hash_object(payload) or chunk.get("schema_version") != 1
                    or chunk.get("clip_key") != key
                    or chunk.get("prompt_ids") != prompt_ids[offset:stop]
                    or scores.shape != (stop - offset,) or not np.isfinite(scores).all()
                    or np.any(np.abs(scores) > 100.001)
                    or path.name != f"chunk_{offset:04d}_{stop:04d}.json"
                    or np.isfinite(values[offset:stop]).any()):
                raise ValueError(f"Invalid or overlapping CLIP checkpoint: {path}")
            values[offset:stop] = scores
        offset = 0
        while offset < EXPECTED_COUNT:
            if math.isfinite(values[offset]):
                offset += 1
                continue
            stop = offset
            while stop < min(offset + self.batch_size, EXPECTED_COUNT) and not math.isfinite(values[stop]):
                stop += 1
            scores = np.asarray(self._clip_score_chunk(paths[offset:stop], text_features[offset:stop]),
                                dtype=np.float64)
            if (scores.shape != (stop - offset,) or not np.isfinite(scores).all()
                    or np.any(np.abs(scores) > 100.001)):
                raise ValueError("CLIP returned incomplete, out-of-range or nonfinite chunk scores")
            chunk = dict(schema_version=1, clip_key=key, offset=offset, stop=stop,
                         prompt_ids=prompt_ids[offset:stop], scores=scores.tolist())
            chunk["chunk_sha256"] = hash_object(chunk)
            atomic_json(folder / f"chunk_{offset:04d}_{stop:04d}.json", chunk)
            values[offset:stop] = scores
            offset = stop
        if values.shape != (EXPECTED_COUNT,) or not np.isfinite(values).all():
            raise ValueError("CLIP returned incomplete or nonfinite scores")
        return values


def evaluate_project(project_root, image_root=None, output_dir=None, batches=None,
                     batch_size=16, device="auto", validate_only=False):
    """Return summary rows; safely resume cached metrics for unchanged inputs."""
    root = Path(project_root).resolve()
    image_root = Path(image_root).resolve() if image_root else root / "results" / "images"
    output_dir = Path(output_dir).resolve() if output_dir else root / "results" / "evaluation_details"
    selected = list(BATCHES) if batches is None else list(batches)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(BATCHES):
        raise ValueError("Choose unique batch names from BATCHES")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    records, manifest_hash = load_manifest(root)
    real_paths = reference_paths(root, records)
    validate_images(real_paths)
    real_hash = content_hash(real_paths)
    baseline_revision = exact_revision(read_run_config(image_root / "dpmpp_s30"))
    revision_adoptions = {}
    validated = []
    for name in selected:
        sampler, schedule, steps = BATCHES[name]
        folder = image_root / name
        paths = validate_batch(folder, records)
        if paths is not None:
            config = read_run_config(folder)
            validate_config(config, sampler, steps)
        row = dict(batch=name, sampler=sampler, schedule=schedule, steps=steps,
                   status="validated" if paths else "missing", n_real=len(real_paths),
                   n_generated=len(paths) if paths else 0, manifest_sha256=manifest_hash,
                   image_content_sha256=content_hash(paths) if paths else "",
                   baseline_model_revision=baseline_revision or "",
                   generation_model_revision="", generation_model_revision_unknown=True,
                   generation_model_revision_status="unknown_historical_revision",
                   generation_revision_adoption_sha256="",
                   generation_weights_match_baseline="unverified_revision_unknown")
        if paths:
            row.update(generation_latency(root, folder, records, sampler, steps))
            row.update(generation_gpu=(config or {}).get("gpu", "unknown"),
                       generation_config_sha256=sha256_file(folder / "run_config.json") if config else "",
                       generation_environment_matches_baseline=bool(config) and
                       all(config.get("versions", {}).get(k) == v for k, v in BASELINE_VERSIONS.items()))
            revision = exact_revision(config)
            row.update(generation_model_revision=revision or "",
                       generation_model_revision_unknown=revision is None,
                       generation_model_revision_status="pinned_commit_recorded" if revision else
                       "unknown_historical_revision",
                       generation_weights_match_baseline=(
                           "verified_revision_equal" if revision == baseline_revision else
                           "verified_revision_differs") if revision and baseline_revision else
                       "unverified_revision_unknown")
            adoption_path = folder / "revision_adoption.json"
            if adoption_path.exists():
                adoption = json.loads(adoption_path.read_text(encoding="utf-8"))
                if not isinstance(adoption, dict):
                    raise ValueError(f"Revision adoption metadata must be an object: {adoption_path}")
                revision_adoptions[name] = adoption
                # Historical images/configs are not retroactively declared pinned.
                # The sidecar documents that a resumed batch crossed this boundary.
                row.update(generation_model_revision_status="mixed_historical_unknown_and_pinned",
                           generation_model_revision_unknown=True,
                           generation_weights_match_baseline="unverified_mixed_historical_and_pinned",
                           generation_revision_adoption_sha256=sha256_file(adoption_path))
        validated.append((row, paths))
    # All existing batches are validated before any model download or computation.
    versions = package_versions()
    protocol = dict(fid="clean-fid/clean/inception_v3/2048", clip_model=CLIP_MODEL,
                    clip_model_revision=CLIP_REVISION, checkpoint_schema=2,
                    clip_aggregation="max(mean(100*cos(normalized embeddings)),0)",
                    clip_precision="float32", text_context=77, truncation="tokenizer/EOS preserved",
                    versions=versions, implementation_sha256=sha256_file(__file__))
    if not validate_only and any(paths for _, paths in validated):
        import numpy as np
        import torch
        from cleanfid import fid
        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; select cpu or install a CUDA-enabled PyTorch build")
        protocol.update(device=resolved_device,
                        evaluation_gpu=torch.cuda.get_device_name(0) if resolved_device == "cuda" else "CPU")
        engine = MetricEngine(output_dir / "cache", resolved_device, batch_size, protocol)
        real_statistics = None
        text_metadata = None
        for row, paths in validated:
            if not paths:
                continue
            key = hash_object(dict(protocol=protocol, reference=real_hash, manifest=manifest_hash,
                                   images=row["image_content_sha256"],
                                   revision_adoption=row["generation_revision_adoption_sha256"]))
            cached = output_dir / "cache" / f"metrics_{key}.json"
            fid_key = fid_checkpoint_key(protocol, real_hash, row["image_content_sha256"])
            fid_cached = output_dir / "cache" / f"fid_{fid_key}.json"
            clip_key = clip_checkpoint_key(protocol, manifest_hash, row["image_content_sha256"], records)
            if cached.exists():
                values = json.loads(cached.read_text(encoding="utf-8"))
                if (values.get("schema_version") != 2 or values.get("metric_key") != key
                        or values.get("fid_key") != fid_key or values.get("clip_key") != clip_key
                        or values.get("prompt_ids") != [record["prompt_id"] for record in records]):
                    raise ValueError(f"Metric checkpoint identity mismatch: {cached}")
                fid_value = validate_fid_cache(dict(schema_version=1, fid_key=fid_key,
                                                    fid_clean=values.get("fid_clean")), fid_key)
                scores = np.asarray(values.get("scores", []), dtype=np.float64)
                clip_summary = clip_values(scores)
                if any(values.get(field) != value for field, value in clip_summary.items()):
                    raise ValueError(f"Metric checkpoint aggregate mismatch: {cached}")
                truncated, counts = np.asarray(values.get("truncated", [])), np.asarray(values.get("token_counts", []))
                if (truncated.shape != (EXPECTED_COUNT,) or counts.shape != (EXPECTED_COUNT,)
                        or not np.isin(truncated, [True, False]).all()
                        or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 2)):
                    raise ValueError(f"Invalid metric checkpoint token metadata: {cached}")
            else:
                if fid_cached.exists():
                    fid_value = validate_fid_cache(json.loads(fid_cached.read_text(encoding="utf-8")), fid_key)
                else:
                    if real_statistics is None:
                        real_features = engine.inception_features(real_paths, real_hash)
                        real_statistics = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
                    features = engine.inception_features(paths, row["image_content_sha256"])
                    fid_value = float(fid.frechet_distance(features.mean(axis=0),
                                      np.cov(features, rowvar=False), *real_statistics))
                    if not math.isfinite(fid_value):
                        raise ValueError("FID returned a nonfinite result")
                    # Commit FID before tokenizer/model initialization or any CLIP
                    # inference. A later failure cannot discard the numeric FID.
                    atomic_json(fid_cached, dict(schema_version=1, fid_key=fid_key, fid_clean=fid_value,
                                                protocol=protocol, reference_content_sha256=real_hash,
                                                image_content_sha256=row["image_content_sha256"]))
                row.update(fid_clean=fid_value, status="fid_evaluated", metric_key=key)
                atomic_csv(output_dir / "summary.csv", [row for row, _ in validated], SUMMARY_COLUMNS)
                if text_metadata is None:
                    text_metadata = engine.text_features(records, manifest_hash)
                text, truncated, counts = text_metadata
                scores = engine.clip_scores(paths, text, records,
                                            row["image_content_sha256"], manifest_hash)
                clip_summary = clip_values(scores)
                atomic_json(cached, dict(schema_version=2, metric_key=key, fid_key=fid_key,
                                         clip_key=clip_key, prompt_ids=[r["prompt_id"] for r in records],
                                         fid_clean=fid_value, **clip_summary, scores=scores.tolist(),
                                         truncated=np.asarray(truncated, dtype=bool).tolist(),
                                         token_counts=np.asarray(counts, dtype=np.int32).tolist()))
            row.update(fid_clean=fid_value, **clip_summary, status="evaluated", metric_key=key,
                       captions_truncated=int(truncated.sum()))
            per_image = [dict(batch=row["batch"], sampler=row["sampler"], steps=row["steps"],
                              prompt_id=r["prompt_id"], image_id=r["image_id"], seed=r["seed"],
                              image_file=p.name, caption=r["caption"], clip_raw_100cos=score,
                              caption_tokens=int(count), truncated=bool(trunc))
                         for r, p, score, count, trunc in zip(records, paths, scores, counts, truncated)]
            atomic_csv(output_dir / f"{row['batch']}_clip_per_image.csv", per_image,
                       ["batch", "sampler", "steps", "prompt_id", "image_id", "seed", "image_file", "caption", "clip_raw_100cos",
                        "caption_tokens", "truncated"])
            atomic_csv(output_dir / "summary.csv", [row for row, _ in validated], SUMMARY_COLUMNS)
            print(f"{row['batch']}: FID={row['fid_clean']:.4f}, CLIP={row['clip_score_100']:.4f}")
    rows = [row for row, _ in validated]
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), project_root=str(root),
                      image_root=str(image_root), validation_only=validate_only,
                      manifest_sha256=manifest_hash, reference_content_sha256=real_hash,
                      baseline_model_revision=baseline_revision,
                      generation_revision_adoptions=revision_adoptions,
                      generation_revision_identity_verified=bool(baseline_revision) and any(
                          paths is not None for _, paths in validated) and all(
                          row["generation_weights_match_baseline"] == "verified_revision_equal"
                          for row, paths in validated if paths),
                      protocol=protocol, batches=rows,
                      limitations="200-image FID is an exploratory biased estimate; one seed per prompt.")
    atomic_csv(output_dir / "summary.csv", rows, SUMMARY_COLUMNS)
    atomic_json(output_dir / "metric_provenance.json", provenance)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batches", nargs="+", choices=list(BATCHES), default=list(BATCHES))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    rows = evaluate_project(**vars(args))
    for row in rows:
        print(f"{row['batch']}: {row['status']} ({row['n_generated']} generated images)")


if __name__ == "__main__":
    main()
