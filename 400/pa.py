from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

# ---------- FP8xFP8 GEMM tile model (matches the screenshot table) ----------

MMA_THROUGHPUT = 16 * 16 * 128 * 2 / 8
HEAD_SIZE = 128
WARP_SIZE = 32
VOP_RATE = 1
DEP_VOP_RATE = 5
TRANS_RATE = 2
DEP_TRANS_RATE = 8



def gemm(qtile, kv_tile):
    ops = qtile * kv_tile * HEAD_SIZE * 2 * 2
    cycles = ops / MMA_THROUGHPUT
    return cycles

def softmax(qtile, kv_tile, is_packed: bool = False):
    s_elems = (qtile * kv_tile) / WARP_SIZE
    
    #dequant
    dequant = (s_elems / 2 if is_packed else s_elems) * VOP_RATE
    
    # intra_max: max3
    intra_max = (s_elems / 2) * VOP_RATE
    
    # inter_max: permlane
    inter_max = (4 * qtile / 16) * VOP_RATE
    
    # s = exp((s - max) * scale), fma + exp
    fma = s_elems / 2 if is_packed else s_elems 
    exp = s_elems
    softmax_fma_exp = fma * VOP_RATE + exp * TRANS_RATE
    
    # get sum
    softmax_sum = (s_elems / 2 if is_packed else s_elems)* VOP_RATE
    
    # quant s
    quant_scale = s_elems / 2 if is_packed else s_elems 
    cvt = s_elems / 2 if is_packed else s_elems 
    s_quant = quant_scale * VOP_RATE + cvt * VOP_RATE
    
    
    # recompute_output, r  and 
    out_elems = (qtile * HEAD_SIZE) / WARP_SIZE
    
    # recompute_output
    # rall *= detla_max
    # rall = r * scale_s + rall
    out_elems = (qtile * HEAD_SIZE) / WARP_SIZE
    recompute_output = (out_elems / 2 if is_packed else out_elems) * 2 * VOP_RATE 
    
    return {
        "dequant": dequant,
        "intra_max": intra_max,
        "inter_max": inter_max,
        "softmax_fma_exp": softmax_fma_exp,
        "softmax_sum": softmax_sum,
        "s_quant": s_quant,
        "recompute_output": recompute_output,
    }


def print_performance_table(qtile, kv_tile, co_exe=False, is_packed=False):
    """打印性能分析表格"""
    
    if co_exe:
        is_packed = False
        
    softmax_metrics = softmax(qtile, kv_tile, is_packed)
    gemm_cycles = gemm(qtile, kv_tile)
    
    softmax_total = sum(softmax_metrics.values())
    
    # compute total cycles
    tot_cycles = gemm_cycles + softmax_total
    
    # compute total cycles for coexecution
    tot_cycles_coexe = gemm_cycles + (softmax_total - gemm_cycles * 3 / 4) / 2
    
    
    # 打印表格
    print("\n" + "=" * 60)
    print(f"Performance Table: qtile={qtile}, kv_tile={kv_tile}, is_packed={is_packed}, co_exe={co_exe}")
    print("=" * 60)
    print(f"{'Component':<25} {'Cycles':<12}")
    print("-" * 60)
    
    # GEMM
    print(f"{'GEMM':<25} {gemm_cycles:<12.2f}")
    
    # Softmax各项
    print(f"{'Softmax Total':<25} {softmax_total:<12.2f}")
    
    for name, value in softmax_metrics.items():
        formatted_name = name.replace("_", " ").title()
        print(f"  {formatted_name:<23} {value:<12.2f}")
    
    # total cycles
    print(f"{'Total Cycles':<25} {tot_cycles:<12.2f}")
    if co_exe:
        print(f"{'Total Cycles for Coexecution':<25} {tot_cycles_coexe:<12.2f}")
    
    print("=" * 60)


if __name__ == "__main__":
    print_performance_table(16, 128, co_exe=False, is_packed=True)
    print_performance_table(32, 128, co_exe=False, is_packed=True)
    print_performance_table(48, 128, co_exe=False, is_packed=True)
    print_performance_table(64, 128, co_exe=False, is_packed=True)

    
    