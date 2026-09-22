import gc
import math
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cuda_sync():
    torch.cuda.synchronize()


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_matmul(
    n,
    precision,
    warmup=10,
    repetitions=30
):
    clear_gpu()

    if precision == "FP32":
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    elif precision == "TF32":
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    elif precision == "FP16":
        dtype = torch.float16

    elif precision == "BF16":
        dtype = torch.bfloat16

    else:
        raise ValueError(f"Unknown precision: {precision}")

    a = torch.randn(
        n,
        n,
        device=device,
        dtype=dtype
    )

    b = torch.randn(
        n,
        n,
        device=device,
        dtype=dtype
    )

    for _ in range(warmup):
        c = torch.matmul(a, b)

    cuda_sync()

    times_ms = []

    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        c = torch.matmul(a, b)
        end.record()

        cuda_sync()

        times_ms.append(start.elapsed_time(end))

    median_ms = float(np.median(times_ms))
    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))

    flops = 2.0 * (n ** 3)

    achieved_tflops = flops / (median_ms / 1000.0) / 1e12

    theoretical = THEORETICAL_TFLOPS[precision]

    percent_peak = (
        achieved_tflops / theoretical
    ) * 100.0

    result = {
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "N": n,
        "precision": precision,
        "warmup": warmup,
        "repetitions": repetitions,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "achieved_tflops": achieved_tflops,
        "theoretical_tflops": theoretical,
        "percent_peak": percent_peak,
    }

    del a
    del b
    del c

    clear_gpu()

    return result


def benchmark_memory_copy(
    num_elements=256_000_000,
    warmup=10,
    repetitions=30
):
    clear_gpu()

    dtype = torch.float32

    x = torch.randn(
        num_elements,
        device=device,
        dtype=dtype
    )

    y = torch.empty_like(x)

    for _ in range(warmup):
        y.copy_(x)

    cuda_sync()

    times_ms = []

    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        y.copy_(x)

        end.record()

        cuda_sync()

        times_ms.append(
            start.elapsed_time(end)
        )

    median_ms = float(
        np.median(times_ms)
    )

    bytes_per_element = x.element_size()

    total_bytes = (
        num_elements
        * bytes_per_element
        * 2
    )

    effective_bandwidth = (
        total_bytes
        / (median_ms / 1000.0)
        / 1e9
    )

    percent_spec = (
        effective_bandwidth
        / GPU_MEMORY_BANDWIDTH_GB_S
        * 100.0
    )

    result = {
        "gpu_uuid": gpu_uuid,
        "num_elements": num_elements,
        "dtype": str(dtype),
        "median_ms": median_ms,
        "bytes_moved": total_bytes,
        "effective_bandwidth_gb_s": effective_bandwidth,
        "percent_spec_bandwidth": percent_spec,
    }

    del x
    del y

    clear_gpu()

    return result


def arithmetic_intensity_copy(dtype_bytes=4):
    flops = 0.0

    bytes_moved = (
        dtype_bytes
        + dtype_bytes
    )

    return flops / bytes_moved


def arithmetic_intensity_add(dtype_bytes=4):
    flops = 1.0

    bytes_moved = (
        dtype_bytes
        + dtype_bytes
        + dtype_bytes
    )

    return flops / bytes_moved


def arithmetic_intensity_matmul(n, dtype_bytes=2):
    flops = 2.0 * n ** 3

    bytes_moved = (
        3.0
        * n ** 2
        * dtype_bytes
    )

    return flops / bytes_moved


def naive_attention(
    q,
    k,
    v
):
    scale = 1.0 / math.sqrt(
        q.shape[-1]
    )

    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    )

    scores = scores * scale

    seq_len = q.shape[-2]

    mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            device=q.device,
            dtype=torch.bool
        ),
        diagonal=1
    )

    scores = scores.masked_fill(
        mask,
        float("-inf")
    )

    probs = torch.softmax(
        scores,
        dim=-1
    )

    output = torch.matmul(
        probs,
        v
    )

    return output


def fused_attention(
    q,
    k,
    v
):
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True
    )


def benchmark_attention(
    seq_len,
    implementation="naive",
    dtype=torch.float16,
    warmup=5,
    repetitions=20
):
    clear_gpu()

    q = torch.randn(
        ATTENTION_BATCH_SIZE,
        1,
        seq_len,
        ATTENTION_HEAD_DIM,
        device=device,
        dtype=dtype
    )

    k = torch.randn_like(q)
    v = torch.randn_like(q)

    attention_fn = (
        naive_attention
        if implementation == "naive"
        else fused_attention
    )

    for _ in range(warmup):
        out = attention_fn(
            q,
            k,
            v
        )

    cuda_sync()

    torch.cuda.reset_peak_memory_stats()

    times_ms = []

    for _ in range(repetitions):
        start = torch.cuda.Event(
            enable_timing=True
        )

        end = torch.cuda.Event(
            enable_timing=True
        )

        start.record()

        out = attention_fn(
            q,
            k,
            v
        )

        end.record()

        cuda_sync()

        times_ms.append(
            start.elapsed_time(end)
        )

    peak_memory_bytes = (
        torch.cuda.max_memory_allocated()
    )

    result = {
        "gpu_uuid": gpu_uuid,
        "implementation": implementation,
        "sequence_length": seq_len,
        "batch_size": ATTENTION_BATCH_SIZE,
        "head_dim": ATTENTION_HEAD_DIM,
        "dtype": str(dtype),
        "median_latency_ms": float(
            np.median(times_ms)
        ),
        "mean_latency_ms": float(
            np.mean(times_ms)
        ),
        "peak_memory_gb": (
            peak_memory_bytes / 1e9
        ),
        "success": True,
    }

    del q
    del k
    del v
    del out

    clear_gpu()

    return result


def attention_can_run(
    seq_len,
    implementation
):
    try:
        clear_gpu()

        q = torch.randn(
            ATTENTION_BATCH_SIZE,
            1,
            seq_len,
            ATTENTION_HEAD_DIM,
            device=device,
            dtype=torch.float16
        )

        k = torch.randn_like(q)
        v = torch.randn_like(q)

        if implementation == "naive":
            out = naive_attention(
                q,
                k,
                v
            )

        else:
            out = fused_attention(
                q,
                k,
                v
            )

        cuda_sync()

        del q
        del k
        del v
        del out

        clear_gpu()

        return True

    except torch.cuda.OutOfMemoryError:
        clear_gpu()

        return False


def refine_oom_boundary(
    implementation,
    coarse_lengths,
    step=256
):
    successes = []

    failures = []

    for seq_len in coarse_lengths:
        success = attention_can_run(
            seq_len,
            implementation
        )

        if success:
            successes.append(seq_len)

        else:
            failures.append(seq_len)

    if len(successes) == 0:
        return {
            "largest_success": None,
            "smallest_failure": min(
                failures
            ),
        }

    if len(failures) == 0:
        return {
            "largest_success": max(
                successes
            ),
            "smallest_failure": None,
        }

    low = max(successes)
    high = min(
        x
        for x in failures
        if x > low
    )

    tested_success = low
    tested_failure = high

    current = low + step

    while current < high:
        print(
            implementation,
            current
        )

        if attention_can_run(
            current,
            implementation
        ):
            tested_success = current

        else:
            tested_failure = current
            break

        current += step

    return {
        "largest_success": tested_success,
        "smallest_failure": tested_failure,
    }


def query_gpu_telemetry():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu="
            "timestamp,"
            "clocks.gr,"
            "clocks.mem,"
            "temperature.gpu,"
            "power.draw,"
            "power.limit,"
            "utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True
    )

    values = [
        item.strip()
        for item
        in result.stdout.strip().split(",")
    ]

    return {
        "timestamp": values[0],
        "gpu_clock_mhz": float(
            values[1]
        ),
        "memory_clock_mhz": float(
            values[2]
        ),
        "temperature_c": float(
            values[3]
        ),
        "power_draw_w": float(
            values[4]
        ),
        "power_limit_w": float(
            values[5]
        ),
        "utilization_gpu_percent": float(
            values[6]
        ),
    }


def sustained_matmul_worker():
    n = 8192

    a = torch.randn(
        n,
        n,
        device=device,
        dtype=torch.float16
    )

    b = torch.randn(
        n,
        n,
        device=device,
        dtype=torch.float16
    )

    return a, b
