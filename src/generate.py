"""
Step 2 (Member A): generate images with Stable Diffusion v1.5 under different samplers.
Everything is fixed (model / prompt / seed / 512x512 / CFG / steps) except the scheduler.

Output:
  results/images/<sampler>_s<steps>/<prompt_id>.png
  results/generation_log.csv   (one row per image, appended; safe to resume)

Usage (Colab, GPU runtime):
  # Exp 1: three samplers, 30 steps
  python src/generate.py --samplers ddim euler dpmpp --steps 30
  # quick test
  python src/generate.py --samplers ddim euler dpmpp --steps 30 --limit 5
  # Member B can reuse it for the steps experiment
  python src/generate.py --samplers dpmpp --steps 10 20 40 50
"""
import argparse, csv, json, os, time

import torch
import diffusers, transformers
from diffusers import (StableDiffusionPipeline, DDIMScheduler,
                       EulerDiscreteScheduler, DPMSolverMultistepScheduler)

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def make_scheduler(name, base_config):
    if name == "ddim":
        return DDIMScheduler.from_config(base_config)
    if name == "euler":
        return EulerDiscreteScheduler.from_config(base_config)
    if name == "dpmpp":  # DPM++ 2M Karras
        return DPMSolverMultistepScheduler.from_config(
            base_config, algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True)
    # --- ablation: separate the solver from the Karras noise schedule ---
    if name == "dpmpp_nokarras":  # DPM++ 2M with the model's default schedule
        return DPMSolverMultistepScheduler.from_config(
            base_config, algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=False)
    if name == "euler_karras":  # Euler with the Karras schedule
        return EulerDiscreteScheduler.from_config(base_config, use_karras_sigmas=True)
    raise ValueError(name)


FIELDS = ["prompt_id", "image_id", "prompt", "seed", "sampler", "steps", "cfg",
          "height", "width", "gen_time_s", "image_path"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/captions.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--samplers", nargs="+", default=["ddim", "euler", "dpmpp"])
    ap.add_argument("--steps", nargs="+", type=int, default=[30])
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="only first N prompts (testing)")
    args = ap.parse_args()

    records = json.load(open(args.data))[: args.limit]

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    base_config = pipe.scheduler.config

    log_path = f"{args.out}/generation_log.csv"
    os.makedirs(args.out, exist_ok=True)
    new_file = not os.path.exists(log_path)
    # rows already logged -> only these count as "done" when resuming (no duplicate rows,
    # and an image saved without a log row gets regenerated)
    done = set()
    if not new_file:
        for row in csv.DictReader(open(log_path)):
            done.add((row["sampler"], int(row["steps"]), int(row["prompt_id"])))
    logf = open(log_path, "a", newline="")
    writer = csv.DictWriter(logf, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    for sampler in args.samplers:
        pipe.scheduler = make_scheduler(sampler, base_config)
        # warm-up so the first timed image isn't slowed by CUDA init
        pipe("warm up", num_inference_steps=5, height=args.size, width=args.size)

        for steps in args.steps:
            folder = f"{args.out}/images/{sampler}_s{steps}"
            os.makedirs(folder, exist_ok=True)
            # record the full run config; refuse to resume into a folder made with different settings
            run_cfg = {
                "model_id": MODEL_ID, "sampler": sampler, "steps": steps, "cfg": args.cfg,
                "height": args.size, "width": args.size, "dtype": "float16", "safety_checker": None,
                "data": args.data,
                "scheduler_class": type(pipe.scheduler).__name__,
                "scheduler_config": dict(pipe.scheduler.config),
                "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                             "transformers": transformers.__version__},
                "gpu": torch.cuda.get_device_name(0),
                "timing_scope": "pipeline call only (text encoding + denoising + VAE decode); "
                                "excludes model loading and image saving; per-sampler warm-up",
            }
            cfg_path = f"{folder}/run_config.json"
            if os.path.exists(cfg_path):
                old = json.load(open(cfg_path))
                for k in ["model_id", "sampler", "steps", "cfg", "height", "width", "scheduler_class", "data"]:
                    if old.get(k) != run_cfg[k]:
                        raise SystemExit(f"Config mismatch in {folder}: {k} {old.get(k)} != {run_cfg[k]}")
            else:
                json.dump(run_cfg, open(cfg_path, "w"), indent=2, default=str)
            print(f"== {sampler} | {steps} steps | {len(records)} prompts")

            for r in records:
                img_path = f"{folder}/{r['prompt_id']:04d}.png"
                if (sampler, steps, r["prompt_id"]) in done and os.path.exists(img_path):
                    continue  # resume support
                g = torch.Generator("cuda").manual_seed(r["seed"])
                torch.cuda.synchronize(); t0 = time.perf_counter()
                img = pipe(r["caption"], num_inference_steps=steps, guidance_scale=args.cfg,
                           height=args.size, width=args.size, generator=g).images[0]
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                img.save(img_path)
                writer.writerow({
                    "prompt_id": r["prompt_id"], "image_id": r["image_id"], "prompt": r["caption"],
                    "seed": r["seed"], "sampler": sampler, "steps": steps, "cfg": args.cfg,
                    "height": args.size, "width": args.size, "gen_time_s": round(dt, 4),
                    "image_path": img_path,
                })
                logf.flush()

    logf.close()
    print("Done. Log:", log_path)


if __name__ == "__main__":
    main()
