from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

# ---------- FP8xFP8 GEMM tile model (matches the screenshot table) ----------

BLOCK_SIZE = 32             # scaling granularity along K
A_LDS_PAD = 16              # extra columns padded for A in LDS
LDS_CAP_BYTES = 320 * 1024  # 320 KiB (gives 7.57 and 4.25 in the screenshot)
LDS_BYTES_PER_CYCLE = 128   # used for "lds time" rows
F8_MFMA_THROUGHPUT = 16 * 16 * 128 / 8


@dataclass(frozen=True)
class TileConfig:
    m: int
    n: int
    k: int

    @property
    def name(self) -> str:
        return f"{self.m}x{self.n}x{self.k}"


def _require_divisible(x: int, d: int, what: str) -> None:
    if x % d != 0:
        raise ValueError(f"{what} must be divisible by {d}, got {x}")


def fp8fp8_tile_metrics(tile_m: int, tile_n: int, tile_k: int) -> Dict[str, Optional[float]]:
    """Compute the table metrics from (tilem, tilen, tilek) for FP8xFP8 GEMM."""

    _require_divisible(tile_k, 32, "tile_k")
    _require_divisible(tile_n, 16, "tile_n")

    # LDS usage (bytes)
    a_lds = tile_m * (tile_k + A_LDS_PAD)         # matches 64*(128+16)=9216
    b_lds = tile_k * tile_n                       # matches 128*256=32768

    # Scale LDS (bytes): (tile_m*tile_k + tile_n*tile_k)/32
    a_scale = (tile_m * tile_k) // BLOCK_SIZE
    b_scale = (tile_n * tile_k) // BLOCK_SIZE
    scale = a_scale + b_scale                    

    total_lds = a_lds + b_lds + scale

    compute_cycles = tile_m * tile_n * tile_k / 4 / F8_MFMA_THROUGHPUT

    # TDM model 
    tdm_a = tile_m
    tdm_b = tile_n * tile_k  // 256
    tdm_issue_time = tdm_a + tdm_b

    # Wave-level LDS + time (bytes/cycle)
    lds_a_inst = (tile_m * tile_k) // 512
    lds_b_inst = (tile_n  / 4 * tile_k) // 512
    lds_scale_inst = (tile_m * tile_k / 32  + tile_n * tile_k / 4 / 32) // 128
    lds_1x4 = a_lds + b_lds // 4
    lds_1x4_time = lds_1x4 / LDS_BYTES_PER_CYCLE

    # Register model rows (matches screenshot values for N=256/512)
    acc_reg = tile_m * tile_n / 4 / 16 / 16 * 8
    a_regs = tile_m * tile_k / 16 / 128 * 16
    b_regs = tile_n  / 4 * tile_k / 16 / 128 * 16
    reg_per_msb = acc_reg / 4 + max(a_regs, b_regs) / 2

    return {
        "tilem": tile_m,
        "tilen": tile_n,
        "tilek": tile_k,
        "compute cycles": compute_cycles,
        "A lds": a_lds,
        "B lds": b_lds,
        "scale": scale,
        "Total lds": total_lds,
        "lds a inst": lds_a_inst,
        "lds b inst": lds_b_inst,
        "lds scale inst": lds_scale_inst,
        "max lds stage": LDS_CAP_BYTES / total_lds,
        "wave lds latency time": lds_1x4_time,
        "TDM A": tdm_a,
        "TDM B": tdm_b,
        "TDM issue time": tdm_issue_time,
        "acc reg": acc_reg,
        "a regs": a_regs,
        "b regs": b_regs,
        "reg per msb": reg_per_msb,
    }


def fp8fp8_table(configs: Iterable[TileConfig]):
    """Build a "rows x configs" table."""

    configs = list(configs)
    cols = {cfg.name: fp8fp8_tile_metrics(cfg.m, cfg.n, cfg.k) for cfg in configs}

    row_order = [
        "tilem", "tilen", "tilek",
        "compute cycles",
        "A lds", "B lds", "scale", "Total lds",
        "lds a inst", "lds b inst", "lds scale inst",
        "max lds stage", "wave lds latency time",
        "TDM A", "TDM B", "TDM issue time",
        "acc reg", "a regs", "b regs", "reg per msb",
    ]

    table = {row: {cfg.name: cols[cfg.name].get(row) for cfg in configs} for row in row_order}

    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(table).T
        df.columns = [cfg.name for cfg in configs]
        return df
    except Exception:
        return table


if __name__ == "__main__":
    # Example matching the screenshot:
    cfgs = [TileConfig(64, 256, 128), TileConfig(64, 512, 128), TileConfig(64, 256, 256)]
    print(fp8fp8_table(cfgs))
