# Autoresearch: Gemma decode tok/s

## Objective
Push decode throughput of `Examples/GemmaCLI` from ~8.5 tok/s to ~13 tok/s on
M-series Macs while preserving accuracy (no quantization). Optimizations should
ideally be general-purpose (apply to any shape / any chip), but shape-specific
matvec / attention tuning is allowed.

## Metrics
- **Primary**: `decode_tps` (tok/s, higher is better)
- **Secondary**: `prompt_tps` (prefill tok/s), `wall_s` (total CLI wall time)

## How to Run
`./autoresearch.sh` builds `gemma-cli` release and runs decoding once.
Output: `METRIC decode_tps=…`, `METRIC prompt_tps=…`, `METRIC wall_s=…`.

Cached model lives under `~/.cache/huggingface/...`. First run downloads
weights; subsequent runs are fast. We use a fixed prompt + 64 new tokens for
stable measurement.

## Files in Scope
- `Sources/Popcorn/Kernels/*.metal` — kernel implementations (Matvec, RMSNorm,
  RopeApply, Transpose12, AttentionScoresSoftmax, AttentionOutput, Add, Mul,
  GeluTanh, ScalarMul, KVCacheWrite, LogitSoftcap, Argmax, EmbeddingGather, …).
  Decode hot path is dominated by matvecs (q/k/v/o/gate/up/down/PLE/logits)
  followed by attention + small element-wise ops.
- `Sources/Popcorn/Kernels/*.swift` — host-side launch params (grid /
  threadgroup / function name selection). Free to tune.
- `Sources/Popcorn/Kernels/Metal4/MPPMatvec.metal`, `MPPMatmul.metal` — MPP
  tile sizes / vendor-primitive use.
- `Examples/GemmaCLI/Sources/GemmaCLI/Gemma4TextInference.swift` — encoding
  order / kernel fusion opportunities.
- `Examples/GemmaCLI/Sources/GemmaCLI/Metal4Encoding.swift` — barrier logic
  (`HazardTrackingEncoder` already coalesces disjoint dispatches; verify it's
  not over-barriering).
- `Examples/GemmaCLI/Sources/GemmaCLI/Gemma4Workspace.swift` — workspace
  buffers (may be widened/aliased for fused kernels).

## Off Limits
- Don't change quantization / numeric precision (must remain bf16 weights, no
  int4/int8).
- Don't break Popcorn's general-purpose contract — keep generic kernels, only
  add specialized variants or knobs.
- Don't change CLI behavior visible to user (prompt, sampling) other than what
  the benchmark script controls.
- Don't touch `Tests/` correctness; if a test fails, kernel logic regressed.

## Constraints
- Output text must remain identical (or numerically equivalent) for fixed
  prompt — sanity-check by eyeballing the printed completion in
  `bench_output.txt`.
- `mise build` and `mise test` must continue to pass.

## What's Been Tried
- **Argmax over 262k logits was a single-thread serial scan** — replaced with a
  parallel reduction kernel (`argmax_row`). Single biggest win: +22% (8.49 →
  10.38 tok/s).
- **Skip Transpose12 when T=1**: the layout transform `[B,T=1,Nh,Hd]` ↔
  `[B,Nh,T=1,Hd]` is identity. Skipped 4 dispatches per layer in decode. Plus
  SwigluMatvec fusion (gate+up+gelu+mul → 1 kernel). Together +1.3%.
- **Skip RowSlice2D when T=1** (row 0 of a 1-row tensor is the whole tensor)
  and **skip LogitSoftcap before argmax** (tanh is monotonic). +0.6%.
- **Tried but discarded**: matvec simd8/simd16 (register pressure / no gain),
  MPP matvec for transposed (8× wasted compute since TileM=8 vs M=1),
  RMSNorm with simdgroup reduction (no GPU time change), AddRMSNorm fusion (no
  effect), FusedKVCacheWrite (no effect), staging x in TG memory (regression).
- **Diagnostic findings** (informed future work):
  - GPU 90.5 ms vs wait 91.5 ms — commit-feedback overhead is negligible (<1 ms).
  - Barriers (with `.device` visibility) collectively cost ~5 ms / token —
    fusion ceiling for non-matvec speedups is ~6%.
  - Matvec bandwidth utilization: 37 GB/s achieved on M1 (peak ~68 GB/s, real-
    world streaming ~50 GB/s). Roughly bandwidth-bound. To beat the matvec
    floor we'd need to read fewer weight bytes (quantization is disallowed).
  - CPU encoding takes 5–7 ms / token — small but real, dominated by argument-
    table bind calls.

## Open Ideas
See `autoresearch.ideas.md`.

## Hot Path (decode, t=1)
Per layer:
1. RMSNorm(input) → normHidden
2. Matvec q/k/v (transposed bf16 weights) — biggest cost
3. RMSNorm q, k; RMSNorm v (no weight)
4. RopeApply q,k; Transpose12 q,k,v
5. KVCacheWrite k,v
6. AttentionScoresSoftmax (one TG per attention head, scans full keyLen)
7. AttentionOutput (Nq*Hd threads, scans keyLen serially)
8. Transpose12; Matvec o_proj (large); RMSNorm; Add
9. RMSNorm; Matvec gate+up (large, intermediate dim); Gelu; Mul;
   Matvec down (large); RMSNorm; Add
10. PLE slice + Matvec gate + GeluTanh + Mul + Matvec proj + RMSNorm + Add
11. ScalarMul

After all layers: RMSNorm, RowSlice, Matvec logits (vocab), softcap, argmax.
