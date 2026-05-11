# Autoresearch ideas backlog — Gemma decode tok/s

Things that look promising but I haven't pursued (or pursued partly).

## Promising but complex
- **Indirect Command Buffers (ICBs) for decode**. The forward-pass dispatch
  sequence is identical token-to-token; only the input ids and offset change.
  Pre-record an ICB once and re-bind ids + offset per token. Should drop CPU
  encode from ~5 ms to ~µs and overlap CPU/GPU latency.
- **Re-layout K/V cache as `[Nkv, Hd, S]`** (swap last two dims). Then
  `AttentionOutput` reads `V[hkv, d, k]` contiguously over `k` (current layout
  forces stride-Hd reads). Requires changes to KVCacheWrite, AttentionScores,
  and AttentionOutput.
- **Interleaved-row weight repacking for matvec**: pack 4 (or 8) consecutive
  rows of W into a single contiguous block so the inner-loop float4 of W
  covers 4 rows in one cache line. Should cut L1 misses for matvec_nk_simd4.
- **Fuse RMSNorm into the producing matvec's last writeback**: each TG of
  e.g. q_proj covers a contiguous chunk of Q's last-dim. If a TG covers an
  entire head (Hd elements), it can compute the head's RMS scale internally
  and emit normalized output. Avoids materializing qFlat/normHidden and
  saves 1 dispatch + 1 barrier per RMSNorm. Single-TG matvec is generally
  slower, so this only wins for small heads.
- **Streaming flash-attention decode kernel** (1-pass online softmax): merges
  AttentionScoresSoftmax + AttentionOutput into one kernel, avoids
  materializing the probs tensor. Modest win on M1 (these kernels are tiny in
  total) but valuable for long-context.

## Low-effort, low-impact (probably <1% each)
- Skip the per-layer `Slice2D` (which copies 256 floats from `pleFull` into
  `pleLayer`) by pointing `Mul`'s `b` input at `pleFull` with a per-layer
  `byteOffset`. Buffer-offset plumbing already exists on `Tensor.Binding`;
  just needs a `Mul` variant that accepts a `b` offset.
- Pre-compute the full RoPE cos/sin tables `[maxSeqLen, Hd/2]` once at init
  and have `RopeApply` read at `byteOffset = position * Hd/2 * 4`. Skips the
  two `RopeBuildCosSin` dispatches per decode token.
- Fuse `kProj` + `vProj` matvecs that share `normHidden` as input (similar
  to SwigluMatvec). Saves 1 dispatch / layer.
- `KVCacheWrite` does a trivial memcpy: have RopeApply (k path) write directly
  into the cache at offset. Requires a stride-aware RopeApply variant.

## Bigger ideas (unlikely to fit "general-purpose Popcorn")
- Custom logits-matvec-then-argmax fused kernel: stream the logits matrix-
  vector product through registers, emit only the top-1 index. Saves the
  ~1 MB logits buffer write/read (tiny on M1).
- Speculative decoding with a small draft model (out of scope per user).
