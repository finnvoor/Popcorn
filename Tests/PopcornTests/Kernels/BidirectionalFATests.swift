import Metal
@testable import Popcorn
import Testing

// Compare MPP and plain FlashAttention on bidirectional input. Both kernels
// should produce identical output to within float tolerances; if the MPP path
// diverges this catches the regression.

private func referenceBidirectionalAttention(
    q: [Float], k: [Float], v: [Float],
    batch: Int, heads: Int, sq: Int, hd: Int
) -> [Float] {
    let scale: Float = 1.0 / Float(hd).squareRoot()
    var out = [Float](repeating: 0, count: batch * heads * sq * hd)
    for b in 0..<batch {
        for h in 0..<heads {
            // scores[i, j] = q[i] dot k[j] * scale
            var scores = [Float](repeating: 0, count: sq * sq)
            for i in 0..<sq {
                for j in 0..<sq {
                    var s: Float = 0
                    for d in 0..<hd {
                        s += q[((b * heads + h) * sq + i) * hd + d]
                            * k[((b * heads + h) * sq + j) * hd + d]
                    }
                    scores[i * sq + j] = s * scale
                }
            }
            for i in 0..<sq {
                var rowMax: Float = -.infinity
                for j in 0..<sq {
                    rowMax = max(rowMax, scores[i * sq + j])
                }
                var sumExp: Float = 0
                for j in 0..<sq {
                    scores[i * sq + j] = expf(scores[i * sq + j] - rowMax)
                    sumExp += scores[i * sq + j]
                }
                for j in 0..<sq {
                    scores[i * sq + j] /= sumExp
                }
                for d in 0..<hd {
                    var acc: Float = 0
                    for j in 0..<sq {
                        acc += scores[i * sq + j] * v[((b * heads + h) * sq + j) * hd + d]
                    }
                    out[((b * heads + h) * sq + i) * hd + d] = acc
                }
            }
        }
    }
    return out
}

private func runFA(
    device: MTLDevice,
    q: [Float], k: [Float], v: [Float],
    batch: Int, heads: Int, sq: Int, hd: Int,
    mask: AttentionMask
) throws -> [Float] {
    let total = batch * heads * sq * hd
    let bytes = total * MemoryLayout<Float>.stride
    let qBuf = try #require(device.makeBuffer(bytes: q, length: bytes, options: .storageModeShared))
    let kBuf = try #require(device.makeBuffer(bytes: k, length: bytes, options: .storageModeShared))
    let vBuf = try #require(device.makeBuffer(bytes: v, length: bytes, options: .storageModeShared))
    let oBuf = try #require(device.makeBuffer(length: bytes, options: .storageModeShared))

    let qT = Tensor(buffer: qBuf, shape: [batch, heads, sq, hd], dataType: .f32)
    let kT = Tensor(buffer: kBuf, shape: [batch, heads, sq, hd], dataType: .f32)
    let vT = Tensor(buffer: vBuf, shape: [batch, heads, sq, hd], dataType: .f32)
    let oT = Tensor(buffer: oBuf, shape: [batch, heads, sq, hd], dataType: .f32)

    let queue = try #require(device.makeCommandQueue())
    let cb = try #require(queue.makeCommandBuffer())
    let cmp = try #require(cb.makeComputeCommandEncoder(dispatchType: .concurrent))
    let lib = try KernelLibrary(device: device)
    let kenc = KernelCommandEncoder(compute: cmp, library: lib, scratch: TestScratchAllocator(device: device))
    let scale: Float = 1.0 / Float(hd).squareRoot()
    try kenc.encode(Kernels.FlashAttention(
        q: qT, k: kT, v: vT, into: oT,
        scale: scale, mask: mask
    ))
    cmp.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let ptr = oBuf.contents().bindMemory(to: Float.self, capacity: total)
    return Array(UnsafeBufferPointer(start: ptr, count: total))
}

@Test func bidirectionalFlashAttentionMatchesReference() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let batch = 1
    let heads = 2
    let sq = 96 // exercises a partial Bc=64 tail tile and qRows=Br=8 boundary
    let hd = 64
    let total = batch * heads * sq * hd

    var rng: UInt64 = 0xdeadbeef_cafebabe
    func nextFloat() -> Float {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
        return Float(Int64(bitPattern: rng) >> 32) / Float(Int32.max) * 0.2
    }
    var q = [Float](repeating: 0, count: total)
    var k = [Float](repeating: 0, count: total)
    var v = [Float](repeating: 0, count: total)
    for i in 0..<total {
        q[i] = nextFloat()
    }
    for i in 0..<total {
        k[i] = nextFloat()
    }
    for i in 0..<total {
        v[i] = nextFloat()
    }

    let expected = referenceBidirectionalAttention(
        q: q, k: k, v: v,
        batch: batch, heads: heads, sq: sq, hd: hd
    )

    let actual = try runFA(
        device: device,
        q: q, k: k, v: v,
        batch: batch, heads: heads, sq: sq, hd: hd,
        mask: .bidirectional
    )

    // Sanity: causal path should be NaN-free (it's the Gemma workload).
    let causalActual = try runFA(
        device: device,
        q: q, k: k, v: v,
        batch: batch, heads: heads, sq: sq, hd: hd,
        mask: .causal
    )
    var causalNans = 0
    for v in causalActual where v.isNaN {
        causalNans += 1
    }
    #expect(causalNans == 0)

    var maxAbs: Float = 0
    var nanCount = 0
    for i in 0..<total {
        if actual[i].isNaN { nanCount += 1; continue }
        let d = abs(actual[i] - expected[i])
        if d > maxAbs { maxAbs = d }
    }
    #expect(nanCount == 0, "MPP FlashAttention produced \(nanCount) NaNs")
    #expect(maxAbs < 1e-3, "MPP FlashAttention max |delta| = \(maxAbs)")
}
