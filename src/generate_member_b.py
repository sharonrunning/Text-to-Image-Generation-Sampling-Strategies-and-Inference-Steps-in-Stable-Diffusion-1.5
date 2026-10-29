"""Member B's SD 1.5 step generation with guarded NVIDIA T4 benchmarking.

Default quality_only mode does not start a timer and writes blank gen_time_s.
The optional t4_benchmark mode checks both nvidia-smi and PyTorch for a T4.
Heavy imports occur inside run_generation, so validation can run without CUDA.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
DEFAULT_STEPS = (10, 20, 40, 50)
FIELDS = ["prompt_id", "image_id", "prompt", "seed", "sampler", "steps", "cfg",
          "height", "width", "gen_time_s", "image_path"]
HELPER_VERSION = 1
TIMING_SCOPE = ("pipeline call only (text encoding + denoising + VAE decode); "
                "excludes model loading and image saving; per-sampler warm-up")


def _canonical(value: Any) -> str:
    # The supplied Diffusers config contains lambda_min_clipped = -Infinity.
    # Python JSON supports that value, as does Member A's saved configuration.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)


def _load_inputs(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    caption_path = project_root / "data" / "captions.json"
    caption_bytes = caption_path.read_bytes()
    records = json.loads(caption_bytes)
    if not isinstance(records, list) or len(records) != 200:
        raise ValueError("data/captions.json must contain exactly the 200 handover prompts.")
    for record in records:
        for name in ("prompt_id", "image_id", "caption", "seed", "file_name"):
            if name not in record:
                raise ValueError(f"A caption record is missing {name}.")
        if not isinstance(record["prompt_id"], int) or not isinstance(record["image_id"], int):
            raise ValueError("Prompt IDs and image IDs must be integers.")
        if not isinstance(record["caption"], str) or not record["caption"].strip():
            raise ValueError("Every prompt must contain a nonempty caption.")
        if record["seed"] != 1000 + record["prompt_id"]:
            raise ValueError(f"Unexpected seed for prompt {record['prompt_id']}.")
    ids = [record["prompt_id"] for record in records]
    if sorted(ids) != list(range(200)):
        raise ValueError("Expected 200 unique prompt IDs, numbered 0 through 199.")
    if len({record["image_id"] for record in records}) != 200:
        raise ValueError("Expected 200 unique COCO image IDs.")
    records.sort(key=lambda record: record["prompt_id"])
    baseline = json.loads((project_root / "results" / "images" / "dpmpp_s30" /
                           "run_config.json").read_text(encoding="utf-8"))
    expected = {"model_id": MODEL_ID, "sampler": "dpmpp", "steps": 30,
                "cfg": 7.5, "height": 512, "width": 512, "dtype": "float16",
                "scheduler_class": "DPMSolverMultistepScheduler", "safety_checker": None}
    for field, value in expected.items():
        if baseline.get(field) != value:
            raise ValueError(f"Unexpected 30-step baseline setting: {field}.")
    _validate_scheduler(baseline.get("scheduler_config", {}))
    return records, baseline, hashlib.sha256(caption_bytes).hexdigest()


def _validate_scheduler(config: dict[str, Any]) -> None:
    expected = {"algorithm_type": "dpmsolver++", "solver_order": 2,
                "use_karras_sigmas": True}
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"The step study requires DPM++ 2M Karras: invalid {field}.")


def _check_versions(actual: dict[str, str], baseline: dict[str, Any],
                    strict_torch: bool = False) -> None:
    expected = baseline.get("versions", {})
    problems = []
    for package in ("torch", "diffusers", "transformers"):
        if package not in expected:
            raise ValueError(f"The reference configuration does not record {package}.")
        # For quality-only generation the CUDA suffix may differ. Benchmarking
        # also matches the CUDA build so the timing evaluator can accept it.
        compare_core = package == "torch" and not strict_torch
        got = actual[package].split("+", 1)[0] if compare_core else actual[package]
        want = str(expected[package]).split("+", 1)[0] if compare_core else str(expected[package])
        if got != want:
            problems.append(f"{package}: installed {actual[package]}, reference {expected[package]}")
    if problems:
        raise RuntimeError("Generation stopped to avoid changing the reference software. "
                           "Install the notebook's matching environment and restart the kernel. "
                           + "; ".join(problems))


def _nvidia_gpu_names() -> list[str]:
    executable = shutil.which("nvidia-smi")
    windows_fallback = Path("C:/Windows/System32/nvidia-smi.exe")
    if executable is None and windows_fallback.is_file():
        executable = str(windows_fallback)
    if executable is None:
        raise RuntimeError("nvidia-smi was not found. Install/check the NVIDIA driver before T4 benchmarking.")
    completed = subprocess.run(
        [executable, "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    names = [name.strip() for name in completed.stdout.splitlines() if name.strip()]
    if not names:
        raise RuntimeError("nvidia-smi did not report a GPU.")
    return names


def _require_t4(torch: Any, device_index: int) -> list[str]:
    torch_name = torch.cuda.get_device_name(device_index)
    names = _nvidia_gpu_names()
    # Requiring all visible physical GPUs to be T4 avoids a CUDA_VISIBLE_DEVICES
    # remapping accidentally validating a different physical card.
    if "T4" not in torch_name or any("T4" not in name for name in names):
        raise RuntimeError(
            f"T4 benchmarking is forbidden on this host: PyTorch GPU={torch_name!r}, "
            f"nvidia-smi GPUs={names!r}. Select quality_only on an RTX 5070 Ti."
        )
    return names


def _read_log(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return FIELDS.copy(), []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if any(field not in columns for field in FIELDS):
            raise ValueError("generation_log.csv does not have the handover's expected columns.")
        rows = list(reader)
    return columns, rows


def _row_key(row: dict[str, str]) -> tuple[str, int, int]:
    try:
        return row["sampler"], int(row["steps"]), int(row["prompt_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("generation_log.csv contains an invalid sampler/steps/prompt ID.") from error


def _atomic_json(path: Path, value: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".member_b_{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_log(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".member_b_", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resume_identity(config: dict[str, Any]) -> dict[str, Any]:
    keys = ("member_b_generation", "model_id", "sampler", "steps", "cfg", "height", "width",
            "dtype", "safety_checker", "data", "scheduler_class", "scheduler_config", "versions",
            "gpu", "device_index", "mode", "timing_scope", "caption_sha256", "model_revision",
            "generator_device", "warmup_steps", "baseline_scheduler_sha256")
    identity = {key: config.get(key) for key in keys}
    scheduler = identity.get("scheduler_config")
    if isinstance(scheduler, dict):
        # Diffusers creates this metadata list from a set, whose ordering can
        # change across processes. Retain every actual scheduler parameter.
        identity["scheduler_config"] = {key: value for key, value in scheduler.items()
                                        if key != "_use_default_values"}
    return identity


def _identity_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(_resume_identity(config)).encode("utf-8")).hexdigest()


def _normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {field: row[field] for field in FIELDS}
    for field in ("prompt_id", "image_id", "seed", "steps", "height", "width"):
        normalized[field] = int(normalized[field])
    normalized["cfg"] = float(normalized["cfg"])
    timing = normalized["gen_time_s"]
    normalized["gen_time_s"] = "" if str(timing).strip() == "" else float(timing)
    normalized["image_path"] = str(normalized["image_path"]).replace("\\", "/")
    return normalized


def _row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(_normalized_row(row)).encode("utf-8")).hexdigest()


def _validate_row(row: dict[str, Any], folder: Path, config: dict[str, Any],
                  record: dict[str, Any]) -> None:
    try:
        normalized = _normalized_row(row)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid generation record for {folder.name}.") from error
    prompt_id = record["prompt_id"]
    expected = {"prompt_id": prompt_id, "image_id": record["image_id"],
                "prompt": record["caption"], "seed": record["seed"], "sampler": "dpmpp",
                "steps": config["steps"], "cfg": 7.5, "height": 512, "width": 512,
                "image_path": f"results/images/{folder.name}/{prompt_id:04d}.png"}
    if any(normalized.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Generation settings differ for {folder.name}, prompt {prompt_id}.")
    timing = normalized["gen_time_s"]
    if config["mode"] == "quality_only" and timing != "":
        raise ValueError(f"A quality_only row contains runtime data: {folder.name}, prompt {prompt_id}.")
    if config["mode"] == "t4_benchmark" and (timing == "" or not math.isfinite(timing) or timing <= 0):
        raise ValueError(f"Invalid T4 timing for {folder.name}, prompt {prompt_id}.")


def _verify_image(path: Path) -> None:
    from PIL import Image
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (512, 512) or image.mode != "RGB":
            raise ValueError(f"Expected a 512x512 RGB PNG: {path}")
        image.verify()
    with Image.open(path) as image:
        image.load()


def _pending_path(folder: Path, prompt_id: int) -> Path:
    return folder / f".member_b_pending_{prompt_id:04d}.json"



def _png_temp_path(folder: Path, prompt_id: int) -> Path:
    return folder / f".member_b_image_{prompt_id:04d}.png.tmp"


def _initialization_path(folder: Path) -> Path:
    return folder.parent / f".member_b_initialize_{folder.name}.json"


def _owned_json_temp(path: Path) -> bool:
    # These names were emitted by this helper's atomic JSON writer, including
    # version 1's random eight-character names. Pending journals are excluded.
    return bool(re.fullmatch(
        r"\.member_b_[a-zA-Z0-9_]{8}\.json|"
        r"\.member_b_(?:run_config\.json|owner\.json|revision_adoption\.json|"
        r"\.member_b_pending_[0-9]{4}\.json|\.member_b_initialize_dpmpp_s[0-9]+\.json)"
        r"\.[a-zA-Z0-9_]{8}\.tmp", path.name))


def _cleanup_json_transaction_temps(target: Path) -> None:
    pattern = re.compile(r"^\.member_b_" + re.escape(target.name) + r"\.[a-zA-Z0-9_]{8}\.tmp$")
    for path in target.parent.iterdir():
        if path.is_file() and pattern.fullmatch(path.name):
            path.unlink()


def _read_owned_configuration(folder: Path, recover_empty: bool = False) -> dict[str, Any] | None:
    config_path = folder / "run_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("member_b_generation") != HELPER_VERSION:
            raise RuntimeError(f"Refusing a batch not owned by this helper: {folder}")
        return config
    initialization = _initialization_path(folder)
    if initialization.exists():
        config = json.loads(initialization.read_text(encoding="utf-8"))
        if config.get("member_b_generation") != HELPER_VERSION:
            raise RuntimeError(f"Invalid batch initialization journal: {initialization}")
        if config.get("steps") != int(folder.name.rsplit("s", 1)[1]):
            raise RuntimeError(f"Wrong batch initialization journal: {initialization}")
        return config
    if not folder.exists():
        return None
    entries = list(folder.iterdir())
    candidates = []
    for path in entries:
        if not path.is_file() or not _owned_json_temp(path):
            continue
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            continue
        if (isinstance(candidate, dict) and candidate.get("member_b_generation") == HELPER_VERSION
                and candidate.get("steps") == int(folder.name.rsplit("s", 1)[1])):
            candidates.append(candidate)
    if candidates and all(_identity_digest(candidate) == _identity_digest(candidates[0])
                          for candidate in candidates):
        return candidates[0]
    if recover_empty and all(path.is_file() and _owned_json_temp(path) for path in entries):
        # Empty legacy folders do not establish ownership by themselves. The
        # explicit flag is the operator's confirmation of an interrupted setup.
        return None
    raise RuntimeError(f"Refusing existing folder without proven provenance: {folder}. "
                       "An empty interrupted helper folder needs --recover-empty-batches.")


def _atomic_image(image: Any, path: Path, pending_path: Path) -> None:
    # The journal is committed first. A hard termination leaves only a temp PNG
    # or a complete final PNG; the final path is never partially written.
    if not pending_path.is_file() or path.exists():
        raise RuntimeError(f"Image transaction is not safe to commit: {path}")
    temporary = _png_temp_path(path.parent, int(path.stem))
    with temporary.open("xb") as handle:
        image.save(handle, format="PNG")
        handle.flush()
        os.fsync(handle.fileno())
    _verify_image(temporary)
    if path.exists():
        raise RuntimeError(f"An image appeared during the transaction: {path}")
    os.replace(temporary, path)


def _prepare_batch(folder: Path, config: dict[str, Any], records: list[dict[str, Any]],
                   rows: list[dict[str, Any]], log_path: Path, columns: list[str],
                   recover_empty: bool = False) -> set[int]:
    config_path = folder / "run_config.json"
    old = _read_owned_configuration(folder, recover_empty)
    if old is not None and _identity_digest(old) != _identity_digest(config):
        raise RuntimeError(f"Resume configuration differs for {folder}; preserve and inspect its run_config.json.")
    if not config_path.is_file():
        # Persist ownership BEFORE creating the directory. This makes every
        # interrupted setup stage recoverable without guessing at provenance.
        folder.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(_initialization_path(folder), config)
        folder.mkdir(exist_ok=True)
        _atomic_json(config_path, config)
    initialization = _initialization_path(folder)
    if initialization.exists():
        init_config = json.loads(initialization.read_text(encoding="utf-8"))
        if _identity_digest(init_config) != _identity_digest(config):
            raise RuntimeError(f"Conflicting initialization journal: {initialization}")
        initialization.unlink()
    _cleanup_json_transaction_temps(initialization)
    for path in folder.iterdir():
        if path.is_file() and _owned_json_temp(path):
            path.unlink()  # Proven helper-owned directory and exact temp pattern.
    expected_names = {f"{record['prompt_id']:04d}.png" for record in records}
    expected_names.update(_pending_path(folder, record["prompt_id"]).name for record in records)
    expected_names.update(_png_temp_path(folder, record["prompt_id"]).name for record in records)
    expected_names.add("revision_adoption.json")
    unexpected = sorted(path.name for path in folder.iterdir()
                        if path.is_file() and path.name != "run_config.json" and path.name not in expected_names)
    if unexpected:
        raise RuntimeError(f"Unexpected files in {folder}: {unexpected[:5]}")
    wanted = {record["prompt_id"]: record for record in records}
    counts: dict[int, int] = {}
    for row in rows:
        sampler, steps, prompt_id = _row_key(row)
        if sampler != "dpmpp" or steps != config["steps"]:
            continue
        if prompt_id not in wanted:
            raise ValueError(f"An unexpected prompt ID {prompt_id} is logged for {folder.name}.")
        _validate_row(row, folder, config, wanted[prompt_id])
        counts[prompt_id] = counts.get(prompt_id, 0) + 1
    if any(count != 1 for count in counts.values()):
        raise ValueError(f"Duplicate generation-log rows for {folder.name}; repair them before resuming.")
    for prompt_id, record in wanted.items():
        pending_path = _pending_path(folder, prompt_id)
        temporary = _png_temp_path(folder, prompt_id)
        if not pending_path.exists():
            if temporary.exists():
                raise RuntimeError(f"Temp PNG has no ownership journal; refusing removal: {temporary}")
            continue
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        if (pending.get("member_b_generation") != HELPER_VERSION or
                pending.get("config_identity_sha256") != _identity_digest(config)):
            raise RuntimeError(f"Pending journal has different provenance: {pending_path}")
        row = pending.get("row")
        if not isinstance(row, dict) or set(row) != set(FIELDS):
            raise ValueError(f"Invalid pending generation record: {pending_path}")
        _validate_row(row, folder, config, record)
        key = ("dpmpp", config["steps"], prompt_id)
        indices = [index for index, existing in enumerate(rows) if _row_key(existing) == key]
        if indices:
            old_row = rows[indices[0]]
            if (_row_digest(old_row) != _row_digest(row) and
                    pending.get("previous_row_sha256") != _row_digest(old_row)):
                raise RuntimeError(f"Pending journal conflicts with an existing log row: {pending_path}")
        elif pending.get("previous_row_sha256") is not None:
            raise RuntimeError(f"Pending journal's previous log row is missing: {pending_path}")
        image_path = folder / f"{prompt_id:04d}.png"
        if image_path.exists():
            try:
                _verify_image(image_path)
            except (OSError, ValueError, SyntaxError):
                # The verified matching journal proves this is an unfinished
                # helper transaction, including version 1's non-atomic save.
                image_path.unlink()
        if not image_path.exists() and temporary.exists():
            try:
                _verify_image(temporary)
            except (OSError, ValueError, SyntaxError):
                temporary.unlink()
            else:
                os.replace(temporary, image_path)
        elif temporary.exists():
            temporary.unlink()  # Complete final image takes precedence.
        if not image_path.exists():
            pending_path.unlink()  # Retry only this unfinished image.
            continue
        if indices:
            if _row_digest(rows[indices[0]]) != _row_digest(row):
                rows[indices[0]] = row
                _atomic_log(log_path, columns, rows)
        else:
            rows.append(row)
            _atomic_log(log_path, columns, rows)
        counts[prompt_id] = 1
        pending_path.unlink()
        print(f"Recovered image/log transaction: {folder.name}/{prompt_id:04d}.png")
    done = set()
    for prompt_id in wanted:
        path = folder / f"{prompt_id:04d}.png"
        if not path.exists():
            continue
        if prompt_id not in counts:
            raise RuntimeError(f"Existing image has no matching log row; refusing to overwrite: {path}")
        _verify_image(path)  # A corrupt completed image without a journal is not silently deleted.
        done.add(prompt_id)
    return done


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        process = kernel.OpenProcess(0x1000, False, pid)  # query only
        if not process:
            return kernel.GetLastError() == 5  # Access denied: conservatively alive.
        code = wintypes.DWORD()
        try:
            return bool(kernel.GetExitCodeProcess(process, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(process)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_lock(path: Path, recover_stale: bool = False) -> str:
    if path.exists():
        if not recover_stale:
            raise RuntimeError(f"Generation lock exists at {path}. Do not interrupt an active run. "
                               "Only after confirming its runtime has stopped, use --recover-stale-lock.")
        if not path.is_dir():
            raise RuntimeError(f"Unexpected lock path: {path}")
        entries = list(path.iterdir())
        owner_path = path / "owner.json"
        if owner_path.is_file():
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            if owner.get("hostname") == socket.gethostname() and _pid_alive(int(owner.get("pid", -1))):
                raise RuntimeError(f"The lock owner is still alive; refusing recovery: {path}")
        if any(entry.name != "owner.json" and not _owned_json_temp(entry) for entry in entries):
            raise RuntimeError(f"Lock contains unknown files; refusing recovery: {path}")
        for entry in entries:
            if not entry.is_file():
                raise RuntimeError(f"Unexpected lock contents: {entry}")
            entry.unlink()
        path.rmdir()  # Explicit confirmed-stale recovery; never automatic.
    path.mkdir()
    token = uuid.uuid4().hex
    try:
        _atomic_json(path / "owner.json", {
            "member_b_generation": HELPER_VERSION, "token": token,
            "hostname": socket.gethostname(), "pid": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        })
    except BaseException:
        for entry in path.iterdir():
            if entry.is_file() and _owned_json_temp(entry):
                entry.unlink()
        path.rmdir()
        raise
    return token


def _release_lock(path: Path, token: str) -> None:
    owner_path = path / "owner.json"
    if not owner_path.is_file():
        raise RuntimeError(f"Own lock metadata disappeared; refusing lock removal: {path}")
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    if owner.get("token") != token:
        raise RuntimeError(f"Lock ownership changed; refusing lock removal: {path}")
    owner_path.unlink()
    path.rmdir()


def _validate_revision(revision: str) -> str:
    if not isinstance(revision, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("model_revision must be a full 40-character model repository commit SHA.")
    return revision.lower()


def _record_revision_adoption(folder: Path, config: dict[str, Any], revision: str,
                              completed_ids: set[int], accepted: bool) -> None:
    path = folder / "revision_adoption.json"
    identity = _identity_digest(config)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("config_identity_sha256") != identity or old.get("pinned_future_revision") != revision:
            raise RuntimeError(f"Different revision adoption already exists: {path}")
        return
    if not accepted:
        raise RuntimeError(f"Incomplete batch {folder.name} has an unknown historical model revision. "
                           "Recover cached revision proof, or explicitly use --adopt-unknown-revision "
                           "to acknowledge that future images use the pinned commit while old weights remain unverified.")
    _atomic_json(path, {
        "member_b_generation": HELPER_VERSION, "config_identity_sha256": identity,
        "historical_model_revision": config.get("model_revision"),
        "historical_revision_verified": False, "pinned_future_revision": revision,
        "historical_prompt_ids": sorted(completed_ids),
        "future_prompt_ids": sorted(set(range(200)) - completed_ids),
        "operator_confirmed_adoption": True,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "limitation": "Historical weights were not identified; within-batch weight equivalence is unverified. "
                      "Original run_config.json is preserved, not rewritten as pinned provenance.",
    })


def _model_commit(pipe: Any) -> str | None:
    # Older runs did not save the model revision, so exact checkpoint identity
    # across A/B cannot be established. Save whatever this loader exposes.
    config = getattr(pipe, "config", {})
    commit = config.get("_commit_hash") if hasattr(config, "get") else None
    if commit:
        return str(commit)
    components = (getattr(pipe, "unet", None), getattr(pipe, "text_encoder", None))
    for component in components:
        config = getattr(component, "config", None)
        commit = getattr(config, "_commit_hash", None)
        if commit:
            return str(commit)
    return None


def run_generation(project_root: Path | str, steps: list[int] | tuple[int, ...] = DEFAULT_STEPS,
                   mode: str = "quality_only", device_index: int = 0,
                   model_revision: str = MODEL_REVISION, adopt_unknown_revision: bool = False,
                   recover_stale_lock: bool = False, recover_empty_batches: bool = False) -> dict[str, Any]:
    """Generate the four missing DPM++ batches; safely resume matching runs.

    Returns counts and provenance. Does not install packages or evaluate metrics.
    Existing baseline images/configurations are never overwritten.
    """
    project_root = Path(project_root).expanduser().resolve()
    steps = list(steps)
    if not steps or len(set(steps)) != len(steps) or any(step not in DEFAULT_STEPS for step in steps):
        raise ValueError("Choose unique missing step counts from 10, 20, 40, 50; reuse the existing 30-step batch.")
    if mode not in ("quality_only", "t4_benchmark"):
        raise ValueError("mode must be quality_only or t4_benchmark.")
    records, baseline, caption_hash = _load_inputs(project_root)

    results = project_root / "results"
    results.mkdir(exist_ok=True)
    lock_path = results / ".member_b_generation.lock"
    token = _acquire_lock(lock_path, recover_stale_lock)
    pipe = None
    torch = None
    try:
        log_path = results / "generation_log.csv"
        columns, rows = _read_log(log_path)
        completed = {}
        existing_configs = {}
        for step in steps:
            folder = results / "images" / f"dpmpp_s{step}"
            old = _read_owned_configuration(folder, recover_empty_batches)
            if old is not None:
                expected = {"model_id": MODEL_ID, "sampler": "dpmpp", "steps": step,
                            "cfg": 7.5, "height": 512, "width": 512, "dtype": "float16",
                            "safety_checker": None, "scheduler_class": "DPMSolverMultistepScheduler",
                            "caption_sha256": caption_hash, "mode": mode}
                if any(old.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(f"Existing batch captions/mode differ: {folder}")
                _validate_scheduler(old.get("scheduler_config", {}))
                existing_configs[step] = old
                completed[step] = _prepare_batch(folder, old, records, rows, log_path, columns,
                                                  recover_empty_batches)
            else:
                completed[step] = set()
        if all(len(completed[step]) == 200 for step in steps):
            # Historical completed batches are valid inputs even if no commit
            # was saved. Do not load a model or rewrite their configuration.
            print("All requested batches are complete; no GPU/model load or provenance rewrite is needed.")
            return {"generated": 0, "resumed": 200 * len(steps), "steps": steps,
                    "n_prompts": 200, "mode": mode, "historical_configs_preserved": True}
        model_revision = _validate_revision(model_revision)
        for step, old in existing_configs.items():
            if len(completed[step]) == 200:
                continue
            recorded_revision = old.get("model_revision")
            if recorded_revision is not None and str(recorded_revision).lower() != model_revision:
                raise RuntimeError(f"Pinned revision differs from incomplete batch dpmpp_s{step}.")
            if recorded_revision is None:
                _record_revision_adoption(results / "images" / f"dpmpp_s{step}", old,
                                          model_revision, completed[step], adopt_unknown_revision)

        import torch
        import diffusers
        import transformers
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
        versions = {"torch": str(torch.__version__), "diffusers": str(diffusers.__version__),
                    "transformers": str(transformers.__version__)}
        _check_versions(versions, baseline, strict_torch=(mode == "t4_benchmark"))
        if not torch.cuda.is_available() or not 0 <= device_index < torch.cuda.device_count():
            raise RuntimeError("Select an available NVIDIA CUDA GPU. CPU generation is disabled for this study.")
        device = f"cuda:{device_index}"
        gpu = torch.cuda.get_device_name(device_index)
        torch.cuda.set_device(device_index)
        nvidia_names = _require_t4(torch, device_index) if mode == "t4_benchmark" else None
        print(f"Generation mode: {mode}; GPU: {gpu}; software: {versions}; pinned revision: {model_revision}")
        print("Do not run another generation process against this project while this cell runs.")
        pipe = StableDiffusionPipeline.from_pretrained(
            baseline["model_id"], revision=model_revision, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(baseline["scheduler_config"])
        scheduler_config = dict(pipe.scheduler.config)
        _validate_scheduler(scheduler_config)
        exposed_revision = _model_commit(pipe)
        if exposed_revision and str(exposed_revision).lower() != model_revision:
            raise RuntimeError("The loader exposed a different checkpoint commit than the pinned revision.")
        shared_config = {
            "member_b_generation": HELPER_VERSION, "model_id": baseline["model_id"],
            "model_revision": model_revision, "reference_model_revision": baseline.get("model_revision"),
            "sampler": "dpmpp", "cfg": 7.5, "height": 512, "width": 512,
            "dtype": "float16", "safety_checker": None, "data": "data/captions.json",
            "scheduler_class": "DPMSolverMultistepScheduler", "scheduler_config": scheduler_config,
            "versions": versions, "gpu": gpu, "device_index": device_index, "mode": mode,
            "generator_device": device, "caption_sha256": caption_hash,
            "warmup_steps": 5 if mode == "t4_benchmark" else 0,
            "timing_scope": TIMING_SCOPE if mode == "t4_benchmark" else "not measured (quality_only)",
            "nvidia_smi_gpus": nvidia_names,
            "baseline_scheduler_sha256": hashlib.sha256(_canonical(baseline["scheduler_config"]).encode()).hexdigest(),
            "comparison_note": "Future generation is pinned to a repository commit. Member A's historical checkpoint remains unverified unless independently recovered.",
        }
        configurations = {}
        for step in steps:
            proposed = dict(shared_config, steps=step)
            folder = results / "images" / f"dpmpp_s{step}"
            if step in existing_configs:
                old = existing_configs[step]
                if len(completed[step]) == 200:
                    configurations[step] = old
                    continue
                comparison = dict(proposed, model_revision=old.get("model_revision"))
                if _identity_digest(comparison) != _identity_digest(old):
                    raise RuntimeError(f"Resume settings differ from preserved configuration: {folder}")
                configurations[step] = old
            else:
                configurations[step] = proposed
                completed[step] = _prepare_batch(folder, proposed, records, rows, log_path, columns,
                                                  recover_empty_batches)
        if mode == "t4_benchmark" and any(len(completed[step]) < 200 for step in steps):
            _require_t4(torch, device_index)  # Must succeed before importing/starting timers.
            import time
            with torch.inference_mode():
                pipe("warm up", num_inference_steps=5, height=512, width=512)
            torch.cuda.synchronize(device_index)
        generated = 0
        resumed = 0
        for step in steps:
            folder = results / "images" / f"dpmpp_s{step}"
            done = completed[step]
            resumed += len(done)
            print(f"DPM++ 2M Karras, {step} steps: {len(done)}/200 already complete.")
            for record in records:
                prompt_id = record["prompt_id"]
                if prompt_id in done:
                    continue
                image_path = folder / f"{prompt_id:04d}.png"
                generator = torch.Generator(device).manual_seed(record["seed"])
                with torch.inference_mode():
                    if mode == "t4_benchmark":
                        torch.cuda.synchronize(device_index)
                        started = time.perf_counter()
                    image = pipe(record["caption"], num_inference_steps=step, guidance_scale=7.5,
                                 height=512, width=512, generator=generator).images[0]
                    elapsed = ""
                    if mode == "t4_benchmark":
                        torch.cuda.synchronize(device_index)
                        elapsed = round(time.perf_counter() - started, 4)
                key = ("dpmpp", step, prompt_id)
                row = {"prompt_id": prompt_id, "image_id": record["image_id"],
                       "prompt": record["caption"], "seed": record["seed"], "sampler": "dpmpp",
                       "steps": step, "cfg": 7.5, "height": 512, "width": 512,
                       "gen_time_s": elapsed,
                       "image_path": image_path.relative_to(project_root).as_posix()}
                _validate_row(row, folder, configurations[step], record)
                previous = [old for old in rows if _row_key(old) == key]
                pending_path = _pending_path(folder, prompt_id)
                if pending_path.exists():
                    raise RuntimeError(f"An unexpected pending journal appeared during generation: {pending_path}")
                _atomic_json(pending_path, {
                    "member_b_generation": HELPER_VERSION,
                    "config_identity_sha256": _identity_digest(configurations[step]),
                    "previous_row_sha256": _row_digest(previous[0]) if previous else None,
                    "row": row,
                })
                _atomic_image(image, image_path, pending_path)
                rows = [row for row in rows if _row_key(row) != key]
                rows.append(row)
                _atomic_log(log_path, columns, rows)
                pending_path.unlink()
                done.add(prompt_id)
                generated += 1
                if len(done) % 20 == 0 or len(done) == 200:
                    print(f"  {step} steps: {len(done)}/200 complete")
        summary = {"generated": generated, "resumed": resumed, "mode": mode, "gpu": gpu,
                   "steps": steps, "n_prompts": len(records), "versions": versions,
                   "log_path": str(log_path), "model_revision": model_revision}
        print(f"Finished: {generated} new images, {resumed} reused local images.")
        return summary
    finally:
        primary_error = sys.exc_info()[0] is not None
        try:
            if pipe is not None:
                del pipe
            if torch is not None:
                try:
                    torch.cuda.empty_cache()
                except Exception as error:
                    # CUDA cleanup must not hide a generation error or leave
                    # a stale lock after an otherwise ordinary process exit.
                    print(f"Warning: CUDA cleanup failed: {error}", file=sys.stderr)
        finally:
            try:
                _release_lock(lock_path, token)
            except Exception as error:
                if primary_error:
                    print(f"Warning: lock cleanup also failed; preserve the original error: {error}",
                          file=sys.stderr)
                else:
                    raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--mode", choices=("quality_only", "t4_benchmark"), default="quality_only")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--model-revision", default=MODEL_REVISION,
                        help="Full 40-character repository commit; future loads always pass revision=SHA.")
    parser.add_argument("--adopt-unknown-revision", action="store_true",
                        help="Explicitly acknowledge unresolved historical weights in an incomplete owned batch.")
    parser.add_argument("--recover-stale-lock", action="store_true",
                        help="Operator confirms the previous runtime has stopped; live local owners are refused.")
    parser.add_argument("--recover-empty-batches", action="store_true",
                        help="Operator confirms an empty legacy batch directory is interrupted helper setup.")
    args = parser.parse_args()
    run_generation(args.project_root, args.steps, args.mode, args.device_index,
                   args.model_revision, args.adopt_unknown_revision,
                   args.recover_stale_lock, args.recover_empty_batches)


if __name__ == "__main__":
    main()
