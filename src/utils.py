"""Shared helpers: seeding, device, model size and latency benchmarking."""
import io
import os
import platform
import random
import time

import numpy as np
import torch

from src.config import SEED


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hardware_info():
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch": torch.__version__,
    }


def model_size_mb(model):
    """Size of the serialized state_dict in MB (1 MB = 10^6 bytes)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def latency_ms(model, input_shape=(1, 3, 128, 128), runs=100, warmup=10, device="cpu"):
    """Mean single-sample forward latency in ms, averaged over `runs` runs."""
    model = model.eval().to(device)
    x = torch.randn(*input_shape, device=device)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": float(np.mean(times)), "std_ms": float(np.std(times)), "runs": runs}
