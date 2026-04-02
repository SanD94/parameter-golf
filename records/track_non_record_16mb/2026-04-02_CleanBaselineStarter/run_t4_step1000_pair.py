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
TARGET_STEPS = (600, 800, 1000)


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

    # 1. bfloat16 → float16 everywhere
    patched = patched.replace("torch.bfloat16", "torch.float16")
    patched = patched.replace(".bfloat16()", ".half()")
    patched = patched.replace("G.bfloat16()", "G.half()")
    patched = patched.replace('dtype=torch.bfloat16, enabled=True', 'dtype=torch.float16, enabled=True')

    # 2. Flash SDP → mem_efficient SDP (T4 flash kernel has issues with GQA)
    patched = patched.replace("enable_flash_sdp(True)", "enable_flash_sdp(False)")
    patched = patched.replace("enable_mem_efficient_sdp(False)", "enable_mem_efficient_sdp(True)")

    # 3. fused=True → fused=False on Adam (fused not supported on T4)
    patched = patched.replace("fused=True", "fused=False")

    # 4. Disable torch.compile (unreliable on T4 / older CUDA)
    patched = patched.replace(
        "zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)",
        "# zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)  # disabled for T4",
    )
    patched = patched.replace(
        "compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)",
        "compiled_model = base_model  # torch.compile disabled for T4",
    )

    # 5. Force single-GPU (no torchrun needed)
    #    The script already handles non-distributed mode, so we just need to make sure
    #    WORLD_SIZE validation doesn't block us. Patch the divisor check.
    patched = patched.replace(
        'if 8 % world_size != 0:\n        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")',
        'if 8 % world_size != 0:\n        world_size = 1  # T4: force single GPU',
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


def classify(delta_1000: float, mean_early_delta: float) -> str:
    if delta_1000 <= -0.003 and mean_early_delta <= -0.002:
        return "✅ PROMOTE"
    if delta_1000 >= 0.001 or mean_early_delta >= 0.001:
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

    deltas = {step: candidate_vals[step] - baseline_vals[step] for step in TARGET_STEPS}
    mean_early_delta = sum(deltas.values()) / len(TARGET_STEPS)
    verdict = classify(deltas[1000], mean_early_delta)

    print(f"\n{'='*60}")
    print("  Step-1000 T4 Pair Summary")
    print(f"{'='*60}")
    print(f"  baseline_log:  {baseline_log}")
    print(f"  candidate_log: {candidate_log}")
    print(f"{'─'*60}")
    print(f"  {'Step':<8} {'Baseline':>12} {'Candidate':>12} {'Delta':>12}")
    print(f"  {'─'*44}")
    for step in TARGET_STEPS:
        print(
            f"  {step:<8} {baseline_vals[step]:>12.6f} {candidate_vals[step]:>12.6f} "
            f"{deltas[step]:>+12.6f}"
        )
    print(f"{'─'*60}")
    print(f"  mean_early_delta(600,800,1000): {mean_early_delta:+.6f}")
    print(f"  verdict: {verdict}")
    print(f"{'─'*60}")
    print("  Rule: promote if delta_1000 ≤ -0.003 AND mean_early ≤ -0.002")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline then candidate on a T4 GPU (sequential, T4-patched) and compare val_bpb."
    )
    parser.add_argument(
        "--env", action="append", default=[], metavar="KEY=VALUE",
        help="Environment override applied to both runs.",
    )
    parser.add_argument(
        "--candidate-env", action="append", default=[], metavar="KEY=VALUE",
        help="Environment override applied only to the candidate run.",
    )
    parser.add_argument(
        "--baseline-env", action="append", default=[], metavar="KEY=VALUE",
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
        "VAL_LOSS_EVERY": "200",
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
