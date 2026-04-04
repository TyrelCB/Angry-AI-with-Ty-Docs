#!/usr/bin/env python3
"""
spark-bench.py — DGX Spark performance benchmark utility
Usage:
  python spark-bench.py baseline                     # capture & save baseline
  python spark-bench.py measure                      # measure and compare to baseline
  python spark-bench.py measure --warn 10            # degrade warning threshold (%, default 10)
  python spark-bench.py show                         # print saved baseline
  python spark-bench.py llm --model <path-or-id>     # LLM token generation benchmark
  python spark-bench.py llm --model <path> --prompt "..." --max-tokens 256
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

BASELINE_FILE = Path(__file__).parent / "spark-bench-baseline.json"
DEGRADATION_WARN_PCT = 10   # default warning threshold
DEGRADATION_FAIL_PCT = 25   # hard-fail threshold

DEFAULT_PROMPT = (
    "Explain the architecture of the NVIDIA Grace Blackwell Superchip, "
    "including its CPU-GPU interconnect, memory subsystem, and key advantages "
    "for AI inference workloads."
)
DEFAULT_MODEL_OLLAMA = "qwen3.5:0.8b"

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(s):    return f"{GREEN}{s}{RESET}"
def warn(s):  return f"{YELLOW}{s}{RESET}"
def fail(s):  return f"{RED}{s}{RESET}"
def head(s):  return f"{BOLD}{CYAN}{s}{RESET}"


# ── timing helper ─────────────────────────────────────────────────────────────
def cuda_time(fn, warmup=3, runs=10):
    """Return median wall-clock seconds for a CUDA kernel function."""
    device = torch.device("cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end) / 1000)   # → seconds
    return float(np.median(times))


def cpu_time(fn, warmup=2, runs=5):
    """Return median wall-clock seconds for a CPU function."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ── GPU monitor ───────────────────────────────────────────────────────────────

class GpuMonitor:
    """Polls nvidia-smi in a background thread during LLM inference runs.

    Captures per-sample: utilization %, SM clock MHz, power W, memory BW %.
    Call start() before a run, stop() after — returns a stats dict with avg/peak.
    """
    _QUERY = "utilization.gpu,clocks.sm,power.draw,utilization.memory"
    _FMT   = "csv,noheader,nounits"

    def __init__(self, interval_s=0.25):
        self.interval_s = interval_s
        self._samples: list[tuple] = []
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._samples.clear()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict | None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=3)
        return self.stats()

    @staticmethod
    def _parse(val: str) -> float | None:
        v = val.strip()
        if v.startswith("[") or v == "N/A":
            return None
        try:
            return float(v)
        except ValueError:
            return None

    @staticmethod
    def _unified_mem_pct() -> float | None:
        """Read unified memory usage % from /proc/meminfo (matches nvtop on GB10)."""
        try:
            with open("/proc/meminfo") as f:
                info = {k.strip(): int(v.split()[0])
                        for k, v in (line.split(":", 1) for line in f if ":" in line)}
            total = info.get("MemTotal", 0)
            avail = info.get("MemAvailable", 0)
            if total:
                return (total - avail) / total * 100.0
        except Exception:
            pass
        return None

    def _poll(self):
        while not self._stop_evt.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self._QUERY}",
                     f"--format={self._FMT}"],
                    capture_output=True, text=True, timeout=2,
                )
                mem_pct = self._unified_mem_pct()
                for line in r.stdout.strip().splitlines():
                    parts = [self._parse(p) for p in line.split(",")]
                    if len(parts) == 4 and all(p is not None for p in parts):
                        self._samples.append(tuple(parts) + (mem_pct,))
            except Exception:
                pass
            self._stop_evt.wait(self.interval_s)

    def stats(self) -> dict | None:
        if not self._samples:
            return None
        util, clk, pwr, membw, mem_pct = zip(*self._samples)
        s = {
            "util_pct":  {"avg": float(np.mean(util)),  "peak": float(np.max(util))},
            "clk_mhz":   {"avg": float(np.mean(clk)),   "peak": float(np.max(clk))},
            "power_w":   {"avg": float(np.mean(pwr)),   "peak": float(np.max(pwr))},
            "membw_pct": {"avg": float(np.mean(membw)), "peak": float(np.max(membw))},
        }
        valid_mem = [m for m in mem_pct if m is not None]
        if valid_mem:
            s["unified_mem_pct"] = {"avg": float(np.mean(valid_mem)),
                                    "peak": float(np.max(valid_mem))}
        return s

    @staticmethod
    def fmt_inline(s: dict) -> str:
        """One-line summary for appending to a run row."""
        mem_str = ""
        if "unified_mem_pct" in s:
            mem_str = f"  mem {s['unified_mem_pct']['peak']:.0f}%"
        return (f"GPU {s['util_pct']['avg']:.0f}%  "
                f"{s['clk_mhz']['avg']:.0f} MHz  "
                f"{s['power_w']['avg']:.0f} W"
                f"{mem_str}")

    @staticmethod
    def fmt_summary(all_stats: list) -> list[str]:
        """Multi-line summary across all runs for printing at the end."""
        if not all_stats:
            return []
        util  = [s["util_pct"]["avg"]  for s in all_stats]
        clk   = [s["clk_mhz"]["avg"]   for s in all_stats]
        pwr   = [s["power_w"]["avg"]   for s in all_stats]
        ppeak = [s["power_w"]["peak"]  for s in all_stats]
        lines = [
            f"  {'GPU utilization (avg)':<28} {np.mean(util):.0f} %",
            f"  {'GPU SM clock (avg)':<28} {np.mean(clk):.0f} MHz",
            f"  {'GPU power (avg / peak)':<28} {np.mean(pwr):.0f} W  /  {max(ppeak):.0f} W",
        ]
        mem_vals = [s["unified_mem_pct"]["peak"] for s in all_stats if "unified_mem_pct" in s]
        if mem_vals:
            lines.append(
                f"  {'Unified mem used (peak)':<28} {max(mem_vals):.0f} %"
                f"  ({max(mem_vals)/100 * 128:.0f} / 128 GB)"
            )
        return lines


# ── individual benchmarks ─────────────────────────────────────────────────────

def bench_gpu_matmul(dtype, size=8192):
    """GPU matrix multiply (GEMM) – primary Tensor Core stress test."""
    a = torch.randn(size, size, device="cuda", dtype=dtype)
    b = torch.randn(size, size, device="cuda", dtype=dtype)
    def fn():
        torch.matmul(a, b)
    elapsed = cuda_time(fn)
    flops = 2 * size**3
    tflops = flops / elapsed / 1e12
    return {"seconds": elapsed, "tflops": tflops, "size": size, "dtype": str(dtype)}


def bench_gpu_memory_bandwidth(size_mb=2048):
    """GPU memory copy bandwidth (device-to-device)."""
    n = (size_mb * 1024 * 1024) // 4          # float32 elements
    src = torch.ones(n, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    def fn():
        dst.copy_(src)
    elapsed = cuda_time(fn)
    bytes_transferred = src.nbytes * 2         # read + write
    gb_s = bytes_transferred / elapsed / 1e9
    return {"seconds": elapsed, "gb_s": gb_s, "size_mb": size_mb}


def bench_gpu_vector_ops(size=256 * 1024 * 1024):
    """GPU element-wise ops (memory-bound kernel stress)."""
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)
    def fn():
        torch.add(a, b, out=a)
    elapsed = cuda_time(fn)
    gb_s = (a.nbytes * 3) / elapsed / 1e9      # 2 reads + 1 write
    return {"seconds": elapsed, "gb_s": gb_s, "size_elements": size}


def bench_gpu_conv2d():
    """GPU Conv2D (ResNet-50-like workload)."""
    x = torch.randn(64, 64, 56, 56, device="cuda", dtype=torch.float16)
    conv = torch.nn.Conv2d(64, 256, kernel_size=1, bias=False).half().cuda()
    def fn():
        conv(x)
    elapsed = cuda_time(fn)
    return {"seconds": elapsed}


def bench_gpu_attention(seq=2048, heads=32, dim=128):
    """Scaled dot-product attention (transformer workload)."""
    q = torch.randn(8, heads, seq, dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    def fn():
        torch.nn.functional.scaled_dot_product_attention(q, k, v)
    elapsed = cuda_time(fn)
    return {"seconds": elapsed, "seq_len": seq, "heads": heads, "head_dim": dim}


def bench_cpu_matmul(size=4096):
    """CPU matrix multiply (measures IPC + cache efficiency)."""
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    def fn():
        np.matmul(a, b)
    elapsed = cpu_time(fn)
    flops = 2 * size**3
    gflops = flops / elapsed / 1e9
    return {"seconds": elapsed, "gflops": gflops, "size": size}


def bench_cpu_memory_bandwidth(size_mb=1024):
    """CPU memory bandwidth (sequential read/write via numpy)."""
    n = (size_mb * 1024 * 1024) // 4
    a = np.ones(n, dtype=np.float32)
    b = np.empty_like(a)
    def fn():
        np.copyto(b, a)
    elapsed = cpu_time(fn)
    gb_s = (a.nbytes * 2) / elapsed / 1e9
    return {"seconds": elapsed, "gb_s": gb_s, "size_mb": size_mb}


def bench_gpu_transfer(size_mb=512):
    """Host↔Device transfer bandwidth."""
    n = (size_mb * 1024 * 1024) // 4
    host   = torch.ones(n, dtype=torch.float32, pin_memory=True)
    device = torch.empty(n, device="cuda", dtype=torch.float32)
    def h2d():
        device.copy_(host, non_blocking=False)
    def d2h():
        host.copy_(device, non_blocking=False)
    t_h2d = cuda_time(h2d)
    t_d2h = cuda_time(d2h)
    gb_s_h2d = host.nbytes / t_h2d / 1e9
    gb_s_d2h = host.nbytes / t_d2h / 1e9
    return {
        "h2d_seconds": t_h2d, "h2d_gb_s": gb_s_h2d,
        "d2h_seconds": t_d2h, "d2h_gb_s": gb_s_d2h,
        "size_mb": size_mb,
    }


def bench_gpu_matmul_fp8(size=8192):
    """GPU FP8 GEMM (E4M3) via torch._scaled_mm — Blackwell Tensor Cores."""
    a = torch.randn(size, size, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    scale = torch.tensor(1.0, device="cuda")
    def fn():
        torch._scaled_mm(a, b.T, scale_a=scale, scale_b=scale, out_dtype=torch.float16)
    elapsed = cuda_time(fn)
    flops  = 2 * size**3
    tflops = flops / elapsed / 1e12
    return {"seconds": elapsed, "tflops": tflops, "size": size, "dtype": "float8_e4m3fn"}


def bench_gpu_matmul_nvfp4(size=4096):
    """GPU NVFP4 GEMM via torch._scaled_mm with blockwise 1×16 scaling.

    Uses float4_e2m1fn_x2 (packed) with float8_e4m3fn block scales —
    the native NVFP4 Tensor Core path on Blackwell (GB10, CC 12.x).
    """
    M = N = K = size
    a = torch.randint(0, 255, (M, K // 2), device="cuda",
                      dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    b = torch.randint(0, 255, (N, K // 2), device="cuda",
                      dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    # blockwise 1×16: one scale per row-block of 16 columns
    sa = torch.ones(M * (K // 16), device="cuda", dtype=torch.float8_e4m3fn)
    sb = torch.ones(N * (K // 16), device="cuda", dtype=torch.float8_e4m3fn)
    def fn():
        torch._scaled_mm(a, b.T, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    elapsed = cuda_time(fn)
    flops  = 2 * M * N * K
    tflops = flops / elapsed / 1e12
    return {"seconds": elapsed, "tflops": tflops, "size": size, "dtype": "nvfp4 (float4_e2m1fn_x2)"}


# ── benchmark suite ───────────────────────────────────────────────────────────

SUITE = [
    ("gpu_matmul_fp16",  lambda: bench_gpu_matmul(torch.float16),  "GPU GEMM FP16 (Tensor Cores)"),
    ("gpu_matmul_bf16",  lambda: bench_gpu_matmul(torch.bfloat16), "GPU GEMM BF16 (Tensor Cores)"),
    ("gpu_matmul_fp32",  lambda: bench_gpu_matmul(torch.float32),  "GPU GEMM FP32"),
    ("gpu_matmul_fp8",   bench_gpu_matmul_fp8,                     "GPU GEMM FP8 E4M3 (Tensor Cores)"),
    ("gpu_matmul_nvfp4", bench_gpu_matmul_nvfp4,                   "GPU GEMM NVFP4 (Tensor Cores)"),
    ("gpu_mem_bw",       bench_gpu_memory_bandwidth,                "GPU Memory Bandwidth"),
    ("gpu_vector_ops",   bench_gpu_vector_ops,                      "GPU Vector Ops (memory-bound)"),
    ("gpu_conv2d",       bench_gpu_conv2d,                          "GPU Conv2D FP16"),
    ("gpu_attention",    bench_gpu_attention,                        "GPU Attention FP16"),
    ("gpu_h2d_transfer", bench_gpu_transfer,                        "Host↔Device Transfer"),
    ("cpu_matmul",       bench_cpu_matmul,                          "CPU GEMM FP32"),
    ("cpu_mem_bw",       bench_cpu_memory_bandwidth,                "CPU Memory Bandwidth"),
]


def run_suite():
    results = {}
    print()
    for key, fn, label in SUITE:
        print(f"  {label:<42}", end="", flush=True)
        try:
            r = fn()
            results[key] = {"status": "ok", **r}
            metric = _primary_metric(key, r)
            print(ok(f"  {metric}"))
        except Exception as exc:
            results[key] = {"status": "error", "error": str(exc)}
            print(fail(f"  ERROR: {exc}"))
    return results


def run_llm_bench(model=DEFAULT_MODEL_OLLAMA, max_tokens=200, runs=3):
    """Run LLM benchmark and return {prefill_tps, decode_tps} for baseline storage."""
    import urllib.request, urllib.error

    api     = "http://localhost:11434/api/generate"
    payload = json.dumps({
        "model": model, "prompt": DEFAULT_PROMPT,
        "stream": False, "options": {"num_predict": max_tokens},
    }).encode()

    label = f"LLM decode  ({model})"
    print(f"  {label:<42}", end="", flush=True)

    try:
        # warm-up
        req = urllib.request.Request(api, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            _ = json.loads(resp.read())

        prefill_list, decode_list = [], []
        for _ in range(runs):
            req = urllib.request.Request(api, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                r = json.loads(resp.read())
            n_prompt = r.get("prompt_eval_count", 0)
            n_gen    = r.get("eval_count", 0)
            pre_ns   = r.get("prompt_eval_duration", 0)
            dec_ns   = r.get("eval_duration", 0)
            if pre_ns: prefill_list.append(n_prompt / (pre_ns / 1e9))
            if dec_ns: decode_list.append(n_gen    / (dec_ns / 1e9))

        med_pre = float(np.median(prefill_list)) if prefill_list else 0
        med_dec = float(np.median(decode_list))  if decode_list  else 0
        result  = {"status": "ok", "model": model,
                   "prefill_tps": med_pre, "decode_tps": med_dec}
        print(ok(f"  prefill {med_pre:.0f} tok/s  decode {med_dec:.1f} tok/s"))
        return result

    except Exception as exc:
        print(warn(f"  SKIP ({exc})"))
        return {"status": "error", "error": str(exc), "model": model}


def _primary_metric(key, r):
    if "tflops" in r:  return f"{r['tflops']:.2f} TFLOPS"
    if "gb_s"   in r:  return f"{r['gb_s']:.1f} GB/s"
    if "gflops" in r:  return f"{r['gflops']:.1f} GFLOPS"
    if "h2d_gb_s" in r:
        return f"H2D {r['h2d_gb_s']:.1f} GB/s  D2H {r['d2h_gb_s']:.1f} GB/s"
    return f"{r.get('seconds', 0)*1000:.1f} ms"


def _scalar_metric(key, r):
    """Return (metric_name, value) used for baseline comparison."""
    if "tflops"   in r: return "tflops",   r["tflops"]
    if "gb_s"     in r: return "gb_s",     r["gb_s"]
    if "gflops"   in r: return "gflops",   r["gflops"]
    if "h2d_gb_s" in r: return "h2d_gb_s", r["h2d_gb_s"]
    return "seconds", r.get("seconds", 0)


# ── baseline I/O ──────────────────────────────────────────────────────────────

def save_baseline(results, llm_result, versions):
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "versions":    versions,
        "results":     results,
        "llm":         llm_result,
    }
    BASELINE_FILE.write_text(json.dumps(payload, indent=2))
    print(f"\n  Baseline saved → {BASELINE_FILE}")


def load_baseline():
    if not BASELINE_FILE.exists():
        sys.exit(fail(f"No baseline found at {BASELINE_FILE}. Run: python spark-bench.py baseline"))
    return json.loads(BASELINE_FILE.read_text())


# ── comparison / degradation ──────────────────────────────────────────────────

def compare(current, cur_llm, baseline_data, warn_pct, fail_pct):
    baseline = baseline_data["results"]
    bas_llm  = baseline_data.get("llm", {})
    print()
    print(f"  {'Benchmark':<42} {'Baseline':>12} {'Current':>12} {'Delta':>10}  Status")
    print("  " + "─" * 86)

    any_warn = any_fail = False

    def _row(label, bas_val, cur_val, unit, higher_is_better=True):
        nonlocal any_warn, any_fail
        if bas_val is None or cur_val is None:
            print(f"  {label:<42}  {'N/A':>12}  {'N/A':>12}  {'N/A':>10}  {warn('SKIP')}")
            return
        delta_pct = ((cur_val - bas_val) / bas_val * 100) if higher_is_better \
                    else ((bas_val - cur_val) / bas_val * 100)
        bas_str = f"{bas_val:.2f} {unit}"
        cur_str = f"{cur_val:.2f} {unit}"
        dlt_str = f"{delta_pct:+.1f}%"
        if delta_pct <= -fail_pct:
            status = fail("FAIL ✗"); any_fail = True
        elif delta_pct <= -warn_pct:
            status = warn("WARN ⚠"); any_warn = True
        else:
            status = ok("OK  ✓")
        print(f"  {label:<42}  {bas_str:>12}  {cur_str:>12}  {dlt_str:>10}  {status}")

    for key, fn, label in SUITE:
        cur = current.get(key, {})
        bas = baseline.get(key, {})
        if cur.get("status") == "error" or bas.get("status") == "error":
            print(f"  {label:<42}  {'N/A':>12}  {'N/A':>12}  {'N/A':>10}  {warn('SKIP')}")
            continue
        mname, cur_val = _scalar_metric(key, cur)
        _,     bas_val = _scalar_metric(key, bas)
        unit = {"tflops": "TFLOPS", "gb_s": "GB/s", "gflops": "GFLOPS",
                "h2d_gb_s": "GB/s", "seconds": "s"}.get(mname, "")
        _row(label, bas_val, cur_val, unit, higher_is_better=(mname != "seconds"))

    # LLM rows
    print("  " + "─" * 86)
    llm_model = bas_llm.get("model", DEFAULT_MODEL_OLLAMA)
    _row(f"LLM prefill ({llm_model})",
         bas_llm.get("prefill_tps"), cur_llm.get("prefill_tps"), "tok/s")
    _row(f"LLM decode  ({llm_model})",
         bas_llm.get("decode_tps"),  cur_llm.get("decode_tps"),  "tok/s")

    print()
    if any_fail:
        print(fail(f"  ✗  Performance degradation detected (>{fail_pct}% drop on one or more benchmarks)"))
    elif any_warn:
        print(warn(f"  ⚠  Minor degradation detected (>{warn_pct}% drop on one or more benchmarks)"))
    else:
        print(ok("  ✓  All benchmarks within acceptable range"))

    return any_fail


# ── system info ───────────────────────────────────────────────────────────────

def collect_versions():
    """Return a dict of all relevant component versions."""
    import ctypes, subprocess

    import math
    props    = torch.cuda.get_device_properties(0)
    # round VRAM up to nearest power-of-2 GiB to match spec (e.g. 121 GiB → 128 GB)
    raw_vram_gib = props.total_memory / (1024 ** 3)
    vram_gb  = 2 ** math.ceil(math.log2(raw_vram_gib))
    cc       = f"{props.major}.{props.minor}"

    # cuBLASLt version — ctypes call, then fallback via ldconfig
    cublaslt_ver = "unknown"
    try:
        lib = ctypes.cdll.LoadLibrary("libcublasLt.so")
        v   = ctypes.c_int()
        lib.cublasLtGetVersion(ctypes.byref(v))
        n = v.value
        if n > 0:
            cublaslt_ver = f"{n//10000}.{(n%10000)//100}.{n%100}"
    except Exception:
        pass
    if cublaslt_ver == "unknown":
        try:
            import glob
            matches = sorted(glob.glob(
                "/usr/local/cuda*/targets/*/lib/libcublasLt.so.*.*.*"))
            if matches:
                cublaslt_ver = Path(matches[-1]).name.split("libcublasLt.so.")[-1]
        except Exception:
            pass

    # CUDA SDK version from version.json
    cuda_sdk = torch.version.cuda
    try:
        vj = json.loads(Path("/usr/local/cuda/version.json").read_text())
        cuda_sdk = vj.get("cuda", {}).get("version", cuda_sdk)
    except Exception:
        pass

    # ollama version
    ollama_ver = "unknown"
    try:
        r = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        ollama_ver = (r.stdout + r.stderr).strip().split()[-1]
    except Exception:
        pass

    # CPU models
    cpus = []
    try:
        cpus = list(dict.fromkeys(
            l.split(":")[1].strip()
            for l in open("/proc/cpuinfo")
            if "Model name" in l or "model name" in l
        ))
    except Exception:
        pass

    # RAM
    ram_gb = 0
    try:
        import psutil, math
        raw_gib = psutil.virtual_memory().total / (1024 ** 3)
        # round up to nearest power-of-2 GiB (matches physical DRAM spec)
        ram_gb = 2 ** math.ceil(math.log2(raw_gib))
    except Exception:
        pass

    return {
        "gpu":          props.name,
        "gpu_vram_gb":  round(vram_gb),
        "gpu_sm_count": props.multi_processor_count,
        "gpu_cc":       cc,
        "cuda_sdk":     cuda_sdk,
        "cublaslt":     cublaslt_ver,
        "pytorch":      torch.__version__,
        "driver":       torch.version.cuda,   # driver-reported CUDA version
        "cpus":         cpus,
        "ram_gb":       ram_gb,
        "ollama":       ollama_ver,
    }


def print_sysinfo(versions=None):
    v = versions or collect_versions()
    print(f"  GPU          : {v['gpu']}  "
          f"({v['gpu_vram_gb']} GB VRAM, {v['gpu_sm_count']} SMs, CC {v['gpu_cc']})")
    print(f"  CUDA SDK     : {v['cuda_sdk']}  |  cuBLASLt: {v['cublaslt']}")
    print(f"  PyTorch      : {v['pytorch']}")
    for cpu in v.get("cpus", []):
        print(f"  CPU          : {cpu}")
    if v.get("ram_gb"):
        print(f"  System RAM   : {v['ram_gb']} GB")
    if v.get("ollama") and v["ollama"] != "unknown":
        print(f"  Ollama       : {v['ollama']}")


# ── LLM token generation benchmark ───────────────────────────────────────────

# ── LLM token generation benchmark ───────────────────────────────────────────


def _run_ollama(model, prompt, max_tokens, runs):
    """Benchmark token generation via ollama REST API (localhost:11434)."""
    import urllib.request, urllib.error

    api = "http://localhost:11434/api/generate"
    payload = json.dumps({
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "options": {"num_predict": max_tokens},
    }).encode()

    print(f"\n  Backend : ollama  ({api})")
    print(f"  Model   : {model}")
    print(f"  Tokens  : {max_tokens}  |  Runs: {runs}\n")

    # warm-up: ensure model is loaded before timing
    print("  Warming up (loading model) …", end="", flush=True)
    try:
        req = urllib.request.Request(api, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            _ = json.loads(resp.read())
        print(ok("  ready"))
    except urllib.error.URLError as e:
        sys.exit(fail(f"\n  Cannot reach ollama at {api}: {e}\n  Is ollama running? (ollama serve)"))

    prefill_tps_list, decode_tps_list, gpu_stats_list = [], [], []

    for run in range(runs):
        req = urllib.request.Request(api, data=payload,
                                     headers={"Content-Type": "application/json"})
        monitor = GpuMonitor()
        monitor.start()
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            r = json.loads(resp.read())
        elapsed = time.perf_counter() - t0
        gpu = monitor.stop()

        n_prompt = r.get("prompt_eval_count", 0)
        n_gen    = r.get("eval_count", 0)
        # durations are in nanoseconds
        pre_ns   = r.get("prompt_eval_duration", 0)
        dec_ns   = r.get("eval_duration", 0)

        prefill_tps = n_prompt / (pre_ns / 1e9) if pre_ns else None
        decode_tps  = n_gen    / (dec_ns / 1e9) if dec_ns else None

        parts = [f"Run {run+1}/{runs}  ({elapsed:.1f}s total)"]
        if prefill_tps:
            parts.append(f"prefill {prefill_tps:6.0f} tok/s  ({n_prompt} tok)")
            prefill_tps_list.append(prefill_tps)
        if decode_tps:
            parts.append(f"decode {decode_tps:6.1f} tok/s  ({n_gen} tok)")
            decode_tps_list.append(decode_tps)
        if gpu:
            parts.append(GpuMonitor.fmt_inline(gpu))
            gpu_stats_list.append(gpu)
        print("  " + "  |  ".join(parts))

    print()
    if prefill_tps_list:
        print(f"  {'Median prefill':<28} {ok(f'{float(np.median(prefill_tps_list)):.0f} tok/s')}")
    if decode_tps_list:
        print(f"  {'Median decode':<28} {ok(f'{float(np.median(decode_tps_list)):.1f} tok/s')}")
    for line in GpuMonitor.fmt_summary(gpu_stats_list):
        print(line)
    print()


def _run_llama_cpp(model_path, prompt, max_tokens, runs, n_gpu_layers=-1):
    """Benchmark token generation via llama-cli."""
    import re

    llama_cli = "/home/tyrel/llama.cpp/build/bin/llama-cli"
    if not Path(llama_cli).exists():
        import shutil
        llama_cli = shutil.which("llama-cli") or llama_cli

    # Validate: llama.cpp needs a GGUF file path, not an ollama model tag
    if not Path(model_path).exists():
        looks_like_tag = "/" not in model_path and not model_path.endswith(".gguf")
        lines = ["\n  ERROR: model not found.",
                 "  Hint: llama.cpp requires a local GGUF file path."]
        if looks_like_tag:
            lines.append(f"  '{model_path}' looks like an Ollama tag — try: --backend ollama")
        else:
            lines.append(f"  File not found: {model_path}")

        # Search common GGUF locations
        search_roots = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / "Downloads",
            Path.home() / "models",
            Path("/models"),
        ]
        found_ggufs = []
        for root in search_roots:
            if root.exists():
                found_ggufs.extend(sorted(root.rglob("*.gguf")))

        if found_ggufs:
            lines.append("\n  Available GGUFs on this system:")
            for g in found_ggufs:
                lines.append(f"    {g}")
        else:
            lines.append("  No .gguf files found in ~/Downloads, ~/.cache/huggingface, or /models")

        sys.exit("\n".join(lines) + "\n")

    print(f"\n  Backend : llama.cpp  ({Path(llama_cli).parent})")
    print(f"  Model   : {Path(model_path).name}")
    print(f"  Tokens  : {max_tokens}  |  Runs: {runs}\n")

    prefill_tps_list, decode_tps_list, gpu_stats_list = [], [], []

    for run in range(runs):
        cmd = [
            llama_cli,
            "-m",  model_path,
            "-p",  prompt,
            "-n",  str(max_tokens),
            "--n-gpu-layers", str(n_gpu_layers),
            "--no-display-prompt",
            "--single-turn",
            "--simple-io",
            "-s",  "42",
        ]
        monitor = GpuMonitor()
        monitor.start()
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.perf_counter() - t0
        gpu = monitor.stop()

        output = result.stdout + result.stderr

        # New llama.cpp format: [ Prompt: 485.2 t/s | Generation: 44.7 t/s ]
        new_m = re.search(
            r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]",
            output)
        # Legacy format: "X tokens per second"
        prefill_m = re.search(
            r"prompt eval time.*?([\d.]+)\s+tokens per second", output)
        decode_m  = re.search(
            r"(?<!prompt )eval time.*?([\d.]+)\s+tokens per second", output)

        if new_m:
            prefill_tps = float(new_m.group(1))
            decode_tps  = float(new_m.group(2))
        elif prefill_m and decode_m:
            prefill_tps = float(prefill_m.group(1))
            decode_tps  = float(decode_m.group(1))
        else:
            prefill_tps = decode_tps = None

        parts = [f"Run {run+1}/{runs}  ({elapsed:.1f}s total)"]
        if prefill_tps:
            parts.append(f"prefill {prefill_tps:6.0f} tok/s")
            prefill_tps_list.append(prefill_tps)
        if decode_tps:
            parts.append(f"decode {decode_tps:6.1f} tok/s")
            decode_tps_list.append(decode_tps)
        if not prefill_tps and not decode_tps:
            parts.append(warn("no stats parsed"))
            # Show last line of stderr to hint at what went wrong
            err_hint = (result.stderr or result.stdout or "").strip().splitlines()
            if err_hint:
                print("  " + "  |  ".join(parts))
                print(f"    stderr: {err_hint[-1][:120]}")
                continue
        if gpu:
            parts.append(GpuMonitor.fmt_inline(gpu))
            gpu_stats_list.append(gpu)
        print("  " + "  |  ".join(parts))

    print()
    if prefill_tps_list:
        print(f"  {'Median prefill':<28} {ok(f'{float(np.median(prefill_tps_list)):.0f} tok/s')}")
    if decode_tps_list:
        print(f"  {'Median decode':<28} {ok(f'{float(np.median(decode_tps_list)):.1f} tok/s')}")
    for line in GpuMonitor.fmt_summary(gpu_stats_list):
        print(line)
    print()


def cmd_llm(args):
    print(head(f"\n╔══ DGX Spark LLM Token Generation Benchmark ════════════════════╗"))
    print_sysinfo()
    print(head(f"╚════════════════════════════════════════════════════════════════╝"))

    if args.backend == "ollama":
        _run_ollama(args.model or DEFAULT_MODEL_OLLAMA, args.prompt, args.max_tokens, args.runs)
    else:
        if not args.model:
            sys.exit(fail("  --model <path-to-gguf> is required for llama.cpp backend"))
        _run_llama_cpp(args.model, args.prompt, args.max_tokens, args.runs)


# ── entry points ──────────────────────────────────────────────────────────────

def cmd_baseline(args):
    print(head("\n╔══ DGX Spark Baseline Capture ══════════════════════════════════╗"))
    versions = collect_versions()
    print_sysinfo(versions)
    print(head("╚════════════════════════════════════════════════════════════════╝"))
    print("\nRunning benchmark suite …")
    results = run_suite()
    print("\nRunning LLM benchmark …")
    llm_result = run_llm_bench(model=args.llm_model, runs=args.llm_runs)
    save_baseline(results, llm_result, versions)
    print(ok("\n  Baseline captured successfully.\n"))


def cmd_measure(args):
    baseline_data = load_baseline()
    cap = baseline_data.get("captured_at", "unknown")
    print(head("\n╔══ DGX Spark Performance Measurement ═══════════════════════════╗"))
    versions = collect_versions()
    print_sysinfo(versions)
    print(f"  Baseline date: {cap}")
    print(head("╚════════════════════════════════════════════════════════════════╝"))
    print("\nRunning benchmark suite …")
    current = run_suite()
    print("\nRunning LLM benchmark …")
    bas_llm = baseline_data.get("llm", {})
    llm_model = bas_llm.get("model", args.llm_model)
    cur_llm = run_llm_bench(model=llm_model, runs=args.llm_runs)
    degraded = compare(current, cur_llm, baseline_data, args.warn, args.fail)
    print()
    sys.exit(1 if degraded else 0)


def cmd_show(args):
    data = load_baseline()
    v    = data.get("versions", {})
    print(head("\n╔══ Saved Baseline ═══════════════════════════════════════════════╗"))
    print(f"  Captured : {data.get('captured_at', 'N/A')}")
    print(f"  GPU      : {v.get('gpu', 'N/A')}  (CC {v.get('gpu_cc','?')}, {v.get('gpu_vram_gb','?')} GB)")
    print(f"  CUDA SDK : {v.get('cuda_sdk', 'N/A')}  |  cuBLASLt: {v.get('cublaslt', 'N/A')}")
    print(f"  PyTorch  : {v.get('pytorch', 'N/A')}")
    if v.get("ollama") and v["ollama"] != "unknown":
        print(f"  Ollama   : {v.get('ollama', 'N/A')}")
    print(head("╚════════════════════════════════════════════════════════════════╝"))
    print()
    print(f"  {'Benchmark':<42} {'Metric':>16}")
    print("  " + "─" * 62)
    for key, fn, label in SUITE:
        r = data["results"].get(key, {})
        if r.get("status") == "error":
            print(f"  {label:<42}  {warn('ERROR'):>16}")
        else:
            m = _primary_metric(key, r)
            print(f"  {label:<42}  {m:>16}")
    llm = data.get("llm", {})
    if llm and llm.get("status") == "ok":
        model = llm.get("model", "?")
        print("  " + "─" * 62)
        print(f"  {'LLM prefill  (' + model + ')':<42}  {llm['prefill_tps']:.0f} tok/s")
        print(f"  {'LLM decode   (' + model + ')':<42}  {llm['decode_tps']:.1f} tok/s")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="DGX Spark benchmark utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    bl = sub.add_parser("baseline", help="Capture and save baseline results")
    bl.add_argument("--llm-model", default=DEFAULT_MODEL_OLLAMA, metavar="MODEL",
                    help=f"Ollama model for LLM benchmark (default: {DEFAULT_MODEL_OLLAMA})")
    bl.add_argument("--llm-runs", type=int, default=3, metavar="N",
                    help="LLM runs for median (default: 3)")

    m = sub.add_parser("measure", help="Measure current performance vs baseline")
    m.add_argument("--warn", type=float, default=DEGRADATION_WARN_PCT,
                   metavar="PCT", help=f"Warning threshold %% (default {DEGRADATION_WARN_PCT})")
    m.add_argument("--fail", type=float, default=DEGRADATION_FAIL_PCT,
                   metavar="PCT", help=f"Fail threshold %% (default {DEGRADATION_FAIL_PCT})")
    m.add_argument("--llm-model", default=DEFAULT_MODEL_OLLAMA, metavar="MODEL",
                    help=f"Ollama model override (default: model from baseline)")
    m.add_argument("--llm-runs", type=int, default=3, metavar="N",
                    help="LLM runs for median (default: 3)")

    sub.add_parser("show", help="Print saved baseline")

    lm = sub.add_parser("llm", help="LLM token generation benchmark (ollama or llama.cpp)")
    lm.add_argument("--backend", choices=["ollama", "llama.cpp"], default="ollama",
                    help="Inference backend (default: ollama)")
    lm.add_argument("--model", default=None, metavar="NAME_OR_PATH",
                    help=f"Ollama model name (default: {DEFAULT_MODEL_OLLAMA}) or path to GGUF for llama.cpp")
    lm.add_argument("--prompt", default=DEFAULT_PROMPT, metavar="TEXT",
                    help="Prompt text (default: built-in DGX Spark question)")
    lm.add_argument("--max-tokens", type=int, default=200, metavar="N",
                    help="Max new tokens to generate per run (default: 200)")
    lm.add_argument("--runs", type=int, default=3, metavar="N",
                    help="Number of generation runs for median (default: 3)")

    args = p.parse_args()
    {"baseline": cmd_baseline, "measure": cmd_measure, "show": cmd_show,
     "llm": cmd_llm}[args.cmd](args)


if __name__ == "__main__":
    main()
