#!/usr/bin/env python3
"""Sequential pair runner for Google Cloud free-tier T4 GPU.

Creates a patched copy of train_gpt.py that works on T4 (compute capability 7.5),
then runs baseline and candidate sequentially and compares val_bpb.

T4 compatibility fixes applied automatically:
  1. bfloat16 → float16  (T4 has no native bf16)
  2. flash_sdp → mem_efficient_sdp  (T4 flash SDP doesn't support GQA / head_dim 64 well)
  3. fused=True → fused=False on Adam  (fused Adam requires newer GPUs)
  4. torch.compile fullgraph → disabled  (T4 compile support is limited)
  5. DDP / torchrun → single-GPU  (free tier = 1 GPU)
  6. Reduced batch sizes to fit 16 GB VRAM

Usage:
    # Default baseline vs baseline (sanity check):
    python3 run_t4_step1000_pair.py

    # Baseline vs leaky_relu2 candidate:
    python3 run_t4_step1000_pair.py --candidate-env ACTIVATION_TYPE=leaky_relu2 LEAKY_RELU_SLOPE=0.5

    # Compare existing logs without retraining:
    python3 run_t4_step1000_pair.py --compare-only logs_t4/baseline.txt logs_t4/candidate.txt
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

STEP_RE = re.compile(r"^step:(\d+)/(\d+) val_loss:([0-9.]+) val_bpb:([0-9.]+)")
TARGET_STEPS = (1000,)


def parse_env_pairs(pairs: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got: {pair}")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Expected non-empty env key in: {pair}")
        env[key] = value
    return env


def patch_for_t4(source: str) -> str:
    """Apply source-level patches to make train_gpt.py run on T4."""
    patched = source

    # 1. bfloat16 → float16 for autocast and model dtype
    #    BUT keep Muon's Newton-Schulz in fp32 (fp16 overflows there)
    patched = patched.replace('dtype=torch.bfloat16, enabled=True', 'dtype=torch.float16, enabled=True')
    # Keep model in fp32; autocast handles fp16 compute. GradScaler needs fp32 gradients.
    patched = patched.replace(").to(device).bfloat16()", ").to(device)")
    # Muon updates_flat buffer — keep fp32 to avoid overflow
    patched = patched.replace(
        "updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)",
        "updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.float32)",
    )
    # Newton-Schulz: keep in fp32 instead of converting to fp16
    patched = patched.replace("X = G.bfloat16()", "X = G.float()")

    # 2. SDP backends: flash requires SM80+ so disable it; enable cudnn + mem_efficient + math
    patched = patched.replace("enable_cudnn_sdp(False)", "enable_cudnn_sdp(True)")
    patched = patched.replace("enable_flash_sdp(True)", "enable_flash_sdp(False)")
    patched = patched.replace("enable_mem_efficient_sdp(False)", "enable_mem_efficient_sdp(True)")
    patched = patched.replace("enable_math_sdp(False)", "enable_math_sdp(True)")

    # 3. GQA: T4 backends don't support enable_gqa, so manually repeat KV heads
    patched = patched.replace(
        '            y = F.scaled_dot_product_attention(\n'
        '                q,\n'
        '                k,\n'
        '                v,\n'
        '                attn_mask=None,\n'
        '                is_causal=True,\n'
        '                enable_gqa=(self.num_kv_heads != self.num_heads),\n'
        '            ).transpose(1, 2)',
        '            if self.num_kv_heads != self.num_heads:\n'
        '                reps = self.num_heads // self.num_kv_heads\n'
        '                k = k.repeat_interleave(reps, dim=1)\n'
        '                v = v.repeat_interleave(reps, dim=1)\n'
        '            y = F.scaled_dot_product_attention(\n'
        '                q, k, v, attn_mask=None, is_causal=True,\n'
        '            ).transpose(1, 2)',
    )

    # 4. fused=True → fused=False on Adam (fused not supported on T4)
    patched = patched.replace("fused=True", "fused=False")

    # 5. torch.compile — keep enabled, Colab PyTorch 2.x + CUDA 12 supports T4
    #    (was previously disabled; re-enabled for performance)

    # 6. Force single-GPU (no torchrun needed)
    #    The script already handles non-distributed mode, so we just need to make sure
    #    WORLD_SIZE validation doesn't block us. Patch the divisor check.
    patched = patched.replace(
        'if 8 % world_size != 0:\n        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")',
        'if 8 % world_size != 0:\n        world_size = 1  # T4: force single GPU',
    )

    # 7. Checkpoint system for resuming interrupted training
    #    Add config vars to Hyperparameters
    patched = patched.replace(
        '    seed = int(os.environ.get("SEED", 1337))',
        '    seed = int(os.environ.get("SEED", 1337))\n'
        '    checkpoint_every = int(os.environ.get("CHECKPOINT_EVERY", 200))\n'
        '    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "")',
    )
    #    Inject checkpoint load before the training loop
    patched = patched.replace(
        '    training_time_ms = 0.0\n'
        '    stop_after_step: int | None = None\n'
        '    torch.cuda.synchronize()\n'
        '    t0 = time.perf_counter()\n'
        '\n'
        '    step = 0',
        '    training_time_ms = 0.0\n'
        '    stop_after_step: int | None = None\n'
        '    step = 0\n'
        '    # --- Checkpoint: attempt resume ---\n'
        '    _ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(os.environ.get("OUT_DIR", "logs")) / "checkpoints"\n'
        '    _ckpt_path = _ckpt_dir / f"{args.run_id}_ckpt.pt"\n'
        '    if _ckpt_path.exists():\n'
        '        _ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)\n'
        '        base_model.load_state_dict(_ckpt["model"])\n'
        '        for _i, _opt in enumerate(optimizers):\n'
        '            _opt.load_state_dict(_ckpt["optimizers"][_i])\n'
        '        step = _ckpt["step"]\n'
        '        training_time_ms = _ckpt["training_time_ms"]\n'
        '        if "grad_scaler" in _ckpt:\n'
        '            grad_scaler.load_state_dict(_ckpt["grad_scaler"])\n'
        '        # Fast-forward the data loader\n'
        '        for _ in range(step * grad_accum_steps):\n'
        '            train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)\n'
        '        log0(f"checkpoint:resumed from step {step} (training_time:{training_time_ms:.0f}ms)")\n'
        '        del _ckpt\n'
        '    torch.cuda.synchronize()\n'
        '    t0 = time.perf_counter()',
    )
    #    Inject checkpoint save after step logging
    patched = patched.replace(
        '        # Needed to sync whether we\'ve reached the wallclock cap.',
        '        # --- Checkpoint: periodic save ---\n'
        '        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:\n'
        '            _ckpt_dir.mkdir(parents=True, exist_ok=True)\n'
        '            _ckpt_save = {\n'
        '                "step": step,\n'
        '                "model": base_model.state_dict(),\n'
        '                "optimizers": [_opt.state_dict() for _opt in optimizers],\n'
        '                "training_time_ms": training_time_ms + 1000.0 * (time.perf_counter() - t0),\n'
        '                "grad_scaler": grad_scaler.state_dict() if "grad_scaler" in dir() else {},\n'
        '            }\n'
        '            torch.save(_ckpt_save, _ckpt_path)\n'
        '            del _ckpt_save\n'
        '            log0(f"checkpoint:saved at step {step} to {_ckpt_path}")\n'
        '\n'
        '        # Needed to sync whether we\'ve reached the wallclock cap.',
    )

    # 8. Add GradScaler for fp16 (required to avoid underflow/overflow)
    #    Create scaler after device setup
    patched = patched.replace(
        "torch.backends.cuda.matmul.allow_tf32 = True",
        "grad_scaler = torch.amp.GradScaler('cuda')\n    torch.backends.cuda.matmul.allow_tf32 = True",
    )
    #    Warmup loop: scale loss and unscale before optimizer step
    patched = patched.replace(
        "(warmup_loss * grad_scale).backward()\n"
        "            for opt in optimizers:\n"
        "                opt.step()",
        "grad_scaler.scale(warmup_loss * grad_scale).backward()\n"
        "            for opt in optimizers:\n"
        "                grad_scaler.unscale_(opt)\n"
        "            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)\n"
        "            for opt in optimizers:\n"
        "                grad_scaler.step(opt)\n"
        "            grad_scaler.update()",
    )
    #    Main loop: scale loss
    patched = patched.replace(
        "(loss * grad_scale).backward()\n"
        "        train_loss /= grad_accum_steps",
        "grad_scaler.scale(loss * grad_scale).backward()\n"
        "        train_loss /= grad_accum_steps",
    )
    #    Main loop: unscale + clip + scaler step
    patched = patched.replace(
        "        if args.grad_clip_norm > 0:\n"
        "            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)\n"
        "        for opt in optimizers:\n"
        "            opt.step()\n"
        "        zero_grad_all()",
        "        for opt in optimizers:\n"
        "            grad_scaler.unscale_(opt)\n"
        "        torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)\n"
        "        for opt in optimizers:\n"
        "            grad_scaler.step(opt)\n"
        "        grad_scaler.update()\n"
        "        zero_grad_all()",
    )

    return patched


def create_patched_script(source_path: Path, out_dir: Path) -> Path:
    """Read the original train_gpt.py, apply T4 patches, write to a temp file."""
    source = source_path.read_text(encoding="utf-8")
    patched = patch_for_t4(source)
    patched_path = out_dir / "_train_gpt_t4_patched.py"
    patched_path.write_text(patched, encoding="utf-8")
    return patched_path


def run_one(
    label: str,
    script_path: Path,
    repo_root: Path,
    out_dir: Path,
    env_overrides: dict[str, str],
) -> Path:
    """Spawn a single training run as a subprocess and wait for it to finish."""
    env = os.environ.copy()
    env.update(env_overrides)
    run_id = env_overrides["RUN_ID"]
    log_path = out_dir / f"{run_id}.txt"

    # Run with plain python (no torchrun) for single T4
    cmd = [sys.executable, str(script_path)]

    print(f"\n{'='*60}")
    print(f"[{label}] Starting — RUN_ID={run_id}")
    print(f"[{label}] command: {' '.join(cmd)}")
    print(f"[{label}] env overrides: { {k: v for k, v in env_overrides.items() if k != 'PYTHONUNBUFFERED'} }")
    print(f"{'='*60}\n")

    subprocess.run(cmd, cwd=repo_root, env=env, check=True)

    gc.collect()

    if not log_path.exists():
        # Check if log ended up in the default logs/ dir instead
        alt = repo_root / "logs" / f"{run_id}.txt"
        if alt.exists():
            log_path = alt
        else:
            raise FileNotFoundError(
                f"Expected log file was not created: {log_path}\n"
                f"Also checked: {alt}"
            )
    print(f"\n[{label}] Done — log saved to {log_path}")
    return log_path


def parse_val_bpb_by_step(log_path: Path) -> dict[int, float]:
    by_step: dict[int, float] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = STEP_RE.match(line)
        if match:
            by_step[int(match.group(1))] = float(match.group(4))
    return by_step


def classify(delta_1000: float) -> str:
    if delta_1000 <= -0.003:
        return "✅ PROMOTE"
    if delta_1000 >= 0.001:
        return "❌ REJECT"
    return "⚠️  BORDERLINE"


def compare(baseline_log: Path, candidate_log: Path) -> None:
    baseline_vals = parse_val_bpb_by_step(baseline_log)
    candidate_vals = parse_val_bpb_by_step(candidate_log)

    for label, vals in [("baseline", baseline_vals), ("candidate", candidate_vals)]:
        missing = [s for s in TARGET_STEPS if s not in vals]
        if missing:
            available = sorted(vals.keys())
            raise ValueError(
                f"{label} log missing validation at steps {missing}. "
                f"Available steps: {available}. "
                "Ensure VAL_LOSS_EVERY=200."
            )

    delta_1000 = candidate_vals[1000] - baseline_vals[1000]
    verdict = classify(delta_1000)

    print(f"\n{'='*60}")
    print("  Step-1000 T4 Pair Summary")
    print(f"{'='*60}")
    print(f"  baseline_log:  {baseline_log}")
    print(f"  candidate_log: {candidate_log}")
    print(f"{'─'*60}")
    print(f"  baseline  val_bpb: {baseline_vals[1000]:.6f}")
    print(f"  candidate val_bpb: {candidate_vals[1000]:.6f}")
    print(f"  delta:             {delta_1000:+.6f}")
    print(f"{'─'*60}")
    print(f"  verdict: {verdict}")
    print(f"  Rule: promote if delta ≤ -0.003, reject if delta ≥ 0.001")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline then candidate on a T4 GPU (sequential, T4-patched) and compare val_bpb."
    )
    parser.add_argument(
        "--env", action="extend", nargs="+", default=[], metavar="KEY=VALUE",
        help="Environment override applied to both runs.",
    )
    parser.add_argument(
        "--candidate-env", action="extend", nargs="+", default=[], metavar="KEY=VALUE",
        help="Environment override applied only to the candidate run.",
    )
    parser.add_argument(
        "--baseline-env", action="extend", nargs="+", default=[], metavar="KEY=VALUE",
        help="Environment override applied only to the baseline run.",
    )
    parser.add_argument("--seed", default="1337", help="Shared seed (default: 1337).")
    parser.add_argument("--label", default="t4_pair", help="Prefix for RUN_ID values.")
    parser.add_argument(
        "--iterations", default="1000", help="Training iterations (default: 1000).",
    )
    parser.add_argument(
        "--compare-only", nargs=2, metavar=("BASELINE_LOG", "CANDIDATE_LOG"),
        help="Skip training — just compare two existing log files.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    source_script = here / "train_gpt.py"
    if not source_script.exists():
        source_script = repo_root / "train_gpt.py"
    if not source_script.exists():
        raise FileNotFoundError("Cannot find train_gpt.py in records folder or repo root")
    out_dir = here / "logs_t4"

    if args.compare_only:
        compare(Path(args.compare_only[0]), Path(args.compare_only[1]))
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Create patched script for T4
    patched_script = create_patched_script(source_script, here)
    print(f"Created T4-patched script: {patched_script}")

    # T4-friendly defaults: smaller batches for 16 GB VRAM
    common_env = {
        "SEED": args.seed,
        "OUT_DIR": str(out_dir),
        "PYTHONUNBUFFERED": "1",
        "ITERATIONS": args.iterations,
        "WARMDOWN_ITERS": args.iterations,
        "VAL_LOSS_EVERY": "0",  # only validate on last step
        "TRAIN_LOG_EVERY": "100",
        "TRAIN_BATCH_TOKENS": "65536",  # 64K tokens per step (vs 512K default)
        "VAL_BATCH_SIZE": "65536",
        "WARMUP_STEPS": "5",
        "MAX_WALLCLOCK_SECONDS": "0",  # no wallclock cap, use iterations
        "ATTENTION_BACKEND": "sdpa",   # force SDPA, no flash_attn_3
    }
    common_env.update(parse_env_pairs(args.env))

    tag = uuid.uuid4().hex[:8]

    # --- Run 1: Baseline ---
    baseline_env = dict(common_env)
    baseline_env.update(parse_env_pairs(args.baseline_env))
    baseline_env["RUN_ID"] = f"{args.label}_baseline_{tag}"
    baseline_log = run_one("BASELINE", patched_script, repo_root, out_dir, baseline_env)

    # --- Run 2: Candidate ---
    candidate_env = dict(common_env)
    candidate_env.update(parse_env_pairs(args.candidate_env))
    candidate_env["RUN_ID"] = f"{args.label}_candidate_{tag}"
    candidate_log = run_one("CANDIDATE", patched_script, repo_root, out_dir, candidate_env)

    # --- Compare ---
    compare(baseline_log, candidate_log)

    # Clean up patched script
    patched_script.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
