"""GPU efficiency instrumentation — answers "is a cart actually more efficient, or a toy?"

The cartridge value-prop (papers: 38.6x memory / 26.4x throughput) is about replacing a
LARGE context with a few KV slots. For a *trigger* cart the replaced "context" is a tiny
conditional instruction, so the efficiency win may be negligible — this module measures the
real cost on the actual hardware instead of assuming, by comparing the SAME behavior produced
two ways (cart vs in-context) on these axes:

  - prefix size        : KV slots a cart adds        vs  tokens an instruction adds
  - prefix KV memory   : bytes the cache holds for that prefix
  - decode speed       : steady-state tok/s
  - decode energy      : mJ per generated token (NVML cumulative energy counter — exact, not sampled)
  - amortization       : one-time cart TRAIN energy / per-query prefill saving = break-even #queries

Punchline this is built to expose: per-token DECODE cost is ~identical whether the prefix is a
4-slot cart or a 30-token instruction (both are tiny next to the generated tokens). The cart's
win scales with the SIZE of the context it replaces — real for a knowledge corpus, negligible
for a short trigger string. So we profile at a short context AND a large one to show the crossover.

Energy via NVML `nvmlDeviceGetTotalEnergyConsumption` (cumulative mJ since driver load; delta a
workload to get exact joules). Falls back to no-energy (time only) if pynvml is unavailable.
"""
import time

try:
    import pynvml
    pynvml.nvmlInit()
    _H = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAVE_NVML = True

    def energy_mj():
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(_H)   # cumulative mJ since driver load

    def power_w():
        return pynvml.nvmlDeviceGetPowerUsage(_H) / 1000.0
except Exception:
    HAVE_NVML = False

    def energy_mj():
        return None

    def power_w():
        return None


class Meter:
    """Context manager capturing wall-time + GPU energy delta around a workload.

    Pass sync=torch.cuda.synchronize so timing/energy bracket the actual GPU work, not just
    the kernel launches. Reads .wall_s and .energy_j after the block (energy_j is None w/o NVML).
    """
    def __init__(self, sync=None):
        self._sync = sync

    def __enter__(self):
        if self._sync:
            self._sync()
        self.t0 = time.time()
        self.e0 = energy_mj()
        return self

    def __exit__(self, *exc):
        if self._sync:
            self._sync()
        self.wall_s = time.time() - self.t0
        self.energy_j = (energy_mj() - self.e0) / 1000.0 if self.e0 is not None else None


def kv_prefix_bytes(n_slots, n_layers, n_kv_heads, head_dim, dtype_bytes=2):
    """Bytes the KV cache holds for an n_slot prefix (K and V, every layer)."""
    return 2 * n_slots * n_layers * n_kv_heads * head_dim * dtype_bytes


def profile_gen(gen_fn, n_reps=5, warmup=1, sync=None):
    """Profile a generation closure that RETURNS the number of new tokens it produced.

    Returns means over n_reps (warmup excluded): wall_s, n_tok, tok_per_s, and — if NVML is
    present — j_per_gen, mj_per_tok, mean_power_w. For stable numbers the closure should
    generate a FIXED token count (disable EOS) so reps are comparable.
    """
    for _ in range(warmup):
        gen_fn()
    wall, ej, ntok = [], [], []
    for _ in range(n_reps):
        with Meter(sync=sync) as m:
            n = gen_fn()
        wall.append(m.wall_s)
        ntok.append(n)
        if m.energy_j is not None:
            ej.append(m.energy_j)
    W = sum(wall) / len(wall)
    N = sum(ntok) / len(ntok)
    out = dict(wall_s=W, n_tok=N, tok_per_s=(N / W if W else 0.0))
    if ej:
        E = sum(ej) / len(ej)
        out.update(j_per_gen=E, mj_per_tok=(1000 * E / N if N else 0.0),
                   mean_power_w=(E / W if W else 0.0))
    return out


def breakeven_queries(train_j, saving_j_per_query):
    """Queries until one-time TRAIN energy is repaid by the per-query prefill saving.
    inf => the cart never pays for itself as an efficiency device at that context size."""
    if saving_j_per_query <= 0:
        return float("inf")
    return train_j / saving_j_per_query


def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
