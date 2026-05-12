import Metal
@testable import Popcorn
import Testing

// MARK: - Helpers

private func makeBuffer(_ device: MTLDevice, _ values: [Float]) throws -> Tensor {
    try makeBuffer(device, values, shape: [values.count])
}

private func makeBuffer(_ device: MTLDevice, _ values: [Float], shape: [Int]) throws -> Tensor {
    let len = max(values.count, 1) * MemoryLayout<Float>.stride
    guard let buf = device.makeBuffer(length: len, options: .storageModeShared) else {
        throw TestSupportError.bufferAllocationFailed
    }
    if !values.isEmpty {
        memcpy(buf.contents(), values, values.count * MemoryLayout<Float>.stride)
    }
    return Tensor(buffer: buf, shape: shape, dataType: .f32)
}

private func makeIntBuffer(_ device: MTLDevice, _ values: [Int32]) throws -> Tensor {
    let len = max(values.count, 1) * MemoryLayout<Int32>.stride
    guard let buf = device.makeBuffer(length: len, options: .storageModeShared) else {
        throw TestSupportError.bufferAllocationFailed
    }
    if !values.isEmpty {
        memcpy(buf.contents(), values, values.count * MemoryLayout<Int32>.stride)
    }
    return Tensor(buffer: buf, shape: [values.count], dataType: .i32)
}

private func makeOutputBuffer(_ device: MTLDevice, count: Int) throws -> Tensor {
    try makeOutputBuffer(device, count: count, shape: [count])
}

private func makeOutputBuffer(_ device: MTLDevice, count: Int, shape: [Int]) throws -> Tensor {
    guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared) else {
        throw TestSupportError.bufferAllocationFailed
    }
    return Tensor(buffer: buffer, shape: shape, dataType: .f32)
}

private func makeIntOutputBuffer(_ device: MTLDevice, count: Int) throws -> Tensor {
    guard let buffer = device.makeBuffer(length: count * MemoryLayout<Int32>.stride, options: .storageModeShared) else {
        throw TestSupportError.bufferAllocationFailed
    }
    return Tensor(buffer: buffer, shape: [count], dataType: .i32)
}

private func readFloats(_ tensor: Tensor, count: Int) -> [Float] {
    Array(UnsafeBufferPointer(start: tensor.buffer.contents().bindMemory(to: Float.self, capacity: count), count: count))
}

private func readInts(_ tensor: Tensor, count: Int) -> [Int32] {
    Array(UnsafeBufferPointer(start: tensor.buffer.contents().bindMemory(to: Int32.self, capacity: count), count: count))
}

private func runKernel(_ device: MTLDevice, _ kernel: some Kernel) throws {
    let queue = try #require(device.makeCommandQueue())
    let cb = try #require(queue.makeCommandBuffer())
    let enc = try #require(cb.makeComputeCommandEncoder())
    let library = try KernelLibrary(device: device)
    let kernelEncoder = KernelCommandEncoder(compute: enc, library: library, scratch: TestScratchAllocator(device: device))
    try kernelEncoder.encode(kernel)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    try #require(cb.status == .completed)
}

private func approxEqual(_ a: Float, _ b: Float, eps: Float = 1e-4) -> Bool {
    abs(a - b) <= eps + eps * max(abs(a), abs(b))
}

// MARK: - ScalarMul

@Test func scalarMul() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let x = try makeBuffer(device, [1, 2, -3, 4])
    let out = try makeOutputBuffer(device, count: 4)
    try runKernel(device, Kernels.ScalarMul(x, by: 2.5, into: out))
    #expect(readFloats(out, count: 4) == [2.5, 5.0, -7.5, 10.0])
}

// MARK: - Mul / BroadcastAdd / GeluTanh

@Test func mul() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let a = try makeBuffer(device, [1, 2, 3, 4])
    let b = try makeBuffer(device, [5, -1, 0.5, 2])
    let out = try makeOutputBuffer(device, count: 4)
    try runKernel(device, Kernels.Mul(a, b, into: out))
    #expect(readFloats(out, count: 4) == [5, -2, 1.5, 8])
}

@Test func broadcastAdd() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let a = try makeBuffer(device, [0, 0, 0, 0, 0, 0])
    let b = try makeBuffer(device, [1, 2, 3])
    let out = try makeOutputBuffer(device, count: 6)
    try runKernel(device, Kernels.BroadcastAdd(a, b, into: out))
    #expect(readFloats(out, count: 6) == [1, 2, 3, 1, 2, 3])
}

@Test func geluTanh() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let x = try makeBuffer(device, [0, 1, -1, 2])
    let out = try makeOutputBuffer(device, count: 4)
    try runKernel(device, Kernels.GeluTanh(x, into: out))
    let result = readFloats(out, count: 4)
    // Reference values from torch.nn.functional.gelu(x, approximate="tanh").
    let expected: [Float] = [0.0, 0.8411920, -0.1588080, 1.9545977]
    for (r, e) in zip(result, expected) {
        #expect(approxEqual(r, e), "\(r) vs \(e)")
    }
}

// MARK: - EmbeddingGather

@Test func embeddingGather() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    // V=3, H=4
    let table: [Float] = [
        0, 0, 0, 0,
        1, 2, 3, 4,
        -1, -2, -3, -4,
    ]
    let ids: [Int32] = [2, 0, 1]
    let tableBuf = try makeBuffer(device, table)
    let idsBuf = try makeIntBuffer(device, ids)
    let out = try makeOutputBuffer(device, count: 3 * 4)
    try runKernel(device, Kernels.EmbeddingGather(
        ids: idsBuf, table: tableBuf, out: out,
        tokenCount: 3, hiddenSize: 4
    ))
    #expect(readFloats(out, count: 12) == [-1, -2, -3, -4, 0, 0, 0, 0, 1, 2, 3, 4])
}

// MARK: - RMSNorm

@Test func rmsNormGemmaStyle() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let H = 4
    let x: [Float] = [1, 2, 3, 4]
    let weight: [Float] = [0, 0, 0, 0] // initial Gemma weights → effective scale = 1
    let xBuf = try makeBuffer(device, x)
    let wBuf = try makeBuffer(device, weight)
    let out = try makeOutputBuffer(device, count: H)
    try runKernel(device, Kernels.RMSNorm(
        x: xBuf, weight: wBuf, out: out,
        rowCount: 1, hiddenSize: H, eps: 1e-6,
        addOneToWeight: true
    ))

    // ms = (1+4+9+16)/4 = 7.5, rsqrt(7.5) ≈ 0.36515
    let scale: Float = 1.0 / (7.5 as Float).squareRoot()
    let result = readFloats(out, count: H)
    for (i, r) in result.enumerated() {
        #expect(approxEqual(r, x[i] * scale), "[\(i)] \(r) vs \(x[i] * scale)")
    }
}

@Test func rmsNormNoWeight() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let x: [Float] = [3, 4]
    let xBuf = try makeBuffer(device, x, shape: [1, 2])
    let out = try makeOutputBuffer(device, count: 2, shape: [1, 2])
    try runKernel(device, Kernels.RMSNorm(xBuf, into: out, eps: 0))
    // ms = 12.5, rsqrt = 1/sqrt(12.5)
    let s: Float = 1.0 / (12.5 as Float).squareRoot()
    let r = readFloats(out, count: 2)
    #expect(approxEqual(r[0], 3 * s))
    #expect(approxEqual(r[1], 4 * s))
}

// MARK: - Matmul

@Test func matmulPlain() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    // A [2,3] * B [3,2]
    let a: [Float] = [1, 2, 3, 4, 5, 6]
    let b: [Float] = [7, 8, 9, 10, 11, 12]
    let aBuf = try makeBuffer(device, a, shape: [2, 3])
    let bBuf = try makeBuffer(device, b, shape: [3, 2])
    let out = try makeOutputBuffer(device, count: 4, shape: [2, 2])
    try runKernel(device, Kernels.Matmul(aBuf, bBuf, into: out))
    // Expected: [[58,64],[139,154]]
    #expect(readFloats(out, count: 4) == [58, 64, 139, 154])
}

@Test func matmulRoutesVectorTransposedWeights() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let x = try makeBuffer(device, [1, 2, 3], shape: [1, 3])
    let w = try makeBuffer(device, [7, 9, 11, 8, 10, 12], shape: [2, 3])
    let out = try makeOutputBuffer(device, count: 2, shape: [1, 2])
    try runKernel(device, Kernels.Matmul(x, w, into: out, transposeB: true))
    #expect(readFloats(out, count: 2) == [58, 64])
}

/// Exercises the vec4 inner loop of `matvec_nk_simd4_typed` with a K that is
/// a multiple of 4 (entire reduction goes through the vec path) and an N that
/// isn't a multiple of 4 (last simdgroup has fewer than 4 valid rows).
@Test func matvecVec4PathLarge() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let (actual, expected) = try runMatmul(device: device, m: 1, k: 256, n: 67, transposeB: true)
    expectAllApproxEqual(actual, expected)
}

/// K not divisible by 4 forces the scalar tail to run on top of the vec4 body.
@Test func matvecVec4PathTail() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let (actual, expected) = try runMatmul(device: device, m: 1, k: 131, n: 64, transposeB: true)
    expectAllApproxEqual(actual, expected)
}

@Test func matmulTransposedB() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    // A [2,3] * B^T where B is stored as [N, K] = [2, 3]
    let a: [Float] = [1, 2, 3, 4, 5, 6]
    let bT: [Float] = [7, 9, 11, 8, 10, 12] // rows are columns of original B
    let aBuf = try makeBuffer(device, a)
    let bBuf = try makeBuffer(device, bT)
    let out = try makeOutputBuffer(device, count: 4)
    try runKernel(device, Kernels.Matmul(
        a: aBuf, b: bBuf, c: out, m: 2, k: 3, n: 2, transposeB: true
    ))
    #expect(readFloats(out, count: 4) == [58, 64, 139, 154])
}

// MARK: - Matmul (large shapes / MPP routing)

//
// Larger shapes exercise the MPP `matmul2d` path on devices that support it
// (Metal 4 / macOS 26+) and the naive fallback elsewhere. They also catch
// edge-tile bounds-check bugs on shapes that aren't multiples of the 64x64
// MPP tile size, and verify the `transposeB` stride flip used by MPP.

private func runMatmul(
    device: MTLDevice, m: Int, k: Int, n: Int, transposeB: Bool
) throws -> (actual: [Float], expected: [Float]) {
    var rng = SystemRandomNumberGenerator()
    let a = (0..<m * k).map { _ in Float.random(in: -1...1, using: &rng) }
    let b = (0..<k * n).map { _ in Float.random(in: -1...1, using: &rng) }

    let aBuf = try makeBuffer(device, a)
    let bBuf = try makeBuffer(device, b)
    let out = try makeOutputBuffer(device, count: m * n)

    try runKernel(device, Kernels.Matmul(
        a: aBuf, b: bBuf, c: out, m: m, k: k, n: n, transposeB: transposeB
    ))

    var expected = [Float](repeating: 0, count: m * n)
    for i in 0..<m {
        for j in 0..<n {
            var acc: Float = 0
            for kk in 0..<k {
                let av = a[i * k + kk]
                let bv = transposeB ? b[j * k + kk] : b[kk * n + j]
                acc += av * bv
            }
            expected[i * n + j] = acc
        }
    }
    return (readFloats(out, count: m * n), expected)
}

private func expectAllApproxEqual(
    _ actual: [Float], _ expected: [Float], eps: Float = 1e-3
) {
    for index in expected.indices {
        #expect(approxEqual(actual[index], expected[index], eps: eps),
                "[\(index)] \(actual[index]) vs \(expected[index])")
    }
}

@Test func matmulAlignedTile() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let (actual, expected) = try runMatmul(device: device, m: 64, k: 64, n: 64, transposeB: false)
    expectAllApproxEqual(actual, expected)
}

@Test func matmulUnalignedShape() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let (actual, expected) = try runMatmul(device: device, m: 70, k: 33, n: 50, transposeB: false)
    expectAllApproxEqual(actual, expected)
}

@Test func matmulLargeTransposedB() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let (actual, expected) = try runMatmul(device: device, m: 64, k: 64, n: 64, transposeB: true)
    expectAllApproxEqual(actual, expected)
}

// MARK: - Softmax

@Test func softmax() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let x: [Float] = [1, 2, 3, 1, 1, 1]
    let xBuf = try makeBuffer(device, x, shape: [2, 3])
    let out = try makeOutputBuffer(device, count: 6, shape: [2, 3])
    try runKernel(device, Kernels.Softmax(xBuf, into: out))
    let r = readFloats(out, count: 6)
    // Row 1: e^1/Z, e^2/Z, e^3/Z
    let e1 = exp(Float(1)); let e2 = exp(Float(2)); let e3 = exp(Float(3))
    let z1 = e1 + e2 + e3
    #expect(approxEqual(r[0], e1 / z1))
    #expect(approxEqual(r[1], e2 / z1))
    #expect(approxEqual(r[2], e3 / z1))
    // Row 2: uniform 1/3
    #expect(approxEqual(r[3], 1.0 / 3))
    #expect(approxEqual(r[4], 1.0 / 3))
    #expect(approxEqual(r[5], 1.0 / 3))
}

// MARK: - Argmax / Gather / TopK

@Test func argmax() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let x: [Float] = [1, 4, 2, 9, 5, 5]
    let xBuf = try makeBuffer(device, x, shape: [2, 3])
    let out = try makeIntOutputBuffer(device, count: 2)
    try runKernel(device, Kernels.Argmax(xBuf, indices: out))
    #expect(readInts(out, count: 2) == [1, 0])
}

@Test func gather() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let table = try makeBuffer(device, [10, 20, 30, 40, 50])
    let idx = try makeIntBuffer(device, [4, 0, 2])
    let out = try makeOutputBuffer(device, count: 3)
    try runKernel(device, Kernels.Gather(table: table, indices: idx, into: out))
    #expect(readFloats(out, count: 3) == [50, 10, 30])
}

@Test func topK() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    // 2 rows, E=5, K=2.
    let x: [Float] = [
        0.1, 0.5, 0.2, 0.8, 0.3,
        9, 1, 5, 1, 7,
    ]
    let xBuf = try makeBuffer(device, x)
    let values = try makeOutputBuffer(device, count: 4)
    let indices = try makeIntOutputBuffer(device, count: 4)
    try runKernel(device, Kernels.TopK(
        x: xBuf, values: values, indices: indices,
        rowCount: 2, elementCount: 5, k: 2
    ))
    let v = readFloats(values, count: 4)
    let i = readInts(indices, count: 4)
    #expect(approxEqual(v[0], 0.8)); #expect(i[0] == 3)
    #expect(approxEqual(v[1], 0.5)); #expect(i[1] == 1)
    #expect(v[2] == 9); #expect(i[2] == 0)
    #expect(v[3] == 7); #expect(i[3] == 4)
}

// MARK: - WeightedSum

@Test func weightedSum() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    // rows=1, K=2, H=3
    // contrib[0] = [[1,2,3],[4,5,6]], weights = [0.25, 0.75]
    // out = 0.25*[1,2,3] + 0.75*[4,5,6] = [3.25, 4.25, 5.25]
    let contrib = try makeBuffer(device, [1, 2, 3, 4, 5, 6])
    let weights = try makeBuffer(device, [0.25, 0.75])
    let out = try makeOutputBuffer(device, count: 3)
    try runKernel(device, Kernels.WeightedSum(
        contrib: contrib, weights: weights, out: out,
        rowCount: 1, k: 2, hiddenSize: 3
    ))
    let r = readFloats(out, count: 3)
    #expect(approxEqual(r[0], 3.25))
    #expect(approxEqual(r[1], 4.25))
    #expect(approxEqual(r[2], 5.25))
}

// MARK: - IndexedMatmul

@Test func indexedMatmul() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    // E=2, K=2, M=2.
    // Expert 0 weight (K,M layout) = [[1,0],[0,1]] (identity)
    // Expert 1 weight              = [[0,1],[1,0]] (swap)
    let weights: [Float] = [
        1, 0, 0, 1,
        0, 1, 1, 0,
    ]
    let x: [Float] = [3, 4, 5, 6] // 2 rows
    let expertIdx: [Int32] = [0, 1]
    let xBuf = try makeBuffer(device, x)
    let wBuf = try makeBuffer(device, weights)
    let eBuf = try makeIntBuffer(device, expertIdx)
    let out = try makeOutputBuffer(device, count: 4)
    try runKernel(device, Kernels.IndexedMatmul(
        x: xBuf, weights: wBuf, expertIndex: eBuf, out: out,
        rowCount: 2, inDim: 2, outDim: 2
    ))
    // Row 0 (expert 0, identity): [3,4]
    // Row 1 (expert 1, swap):     [6,5]
    #expect(readFloats(out, count: 4) == [3, 4, 6, 5])
}

// MARK: - AttentionMaskBuild

@Test func attentionMaskCausal() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let out = try makeOutputBuffer(device, count: 9)
    try runKernel(device, Kernels.AttentionMaskBuild(
        mask: out, queryLen: 3, keyLen: 3, slidingWindow: nil
    ))
    let m = readFloats(out, count: 9)
    // Causal lower triangle: row i allows j <= i.
    #expect(m[0] == 0); #expect(m[1].isInfinite); #expect(m[2].isInfinite)
    #expect(m[3] == 0); #expect(m[4] == 0); #expect(m[5].isInfinite)
    #expect(m[6] == 0); #expect(m[7] == 0); #expect(m[8] == 0)
}

@Test func attentionMaskSlidingWindow() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let out = try makeOutputBuffer(device, count: 16)
    try runKernel(device, Kernels.AttentionMaskBuild(
        mask: out, queryLen: 4, keyLen: 4, slidingWindow: 2
    ))
    let m = readFloats(out, count: 16)
    // Window 2 means each row i sees only j in (i-1, i].
    // Row 0: only j=0.
    #expect(m[0] == 0); #expect(m[1].isInfinite)
    // Row 1: j=0 and j=1.
    #expect(m[4] == 0); #expect(m[5] == 0); #expect(m[6].isInfinite)
    // Row 2: j=1, j=2 only (j=0 excluded by window).
    #expect(m[8].isInfinite); #expect(m[9] == 0); #expect(m[10] == 0); #expect(m[11].isInfinite)
}

// MARK: - RoPE

@Test func ropeBuildAndApply() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let Hd = 4
    let T = 2
    let positions: [Int32] = [0, 1]
    let invFreq: [Float] = [1.0, 0.5] // Hd/2 = 2

    let posBuf = try makeIntBuffer(device, positions)
    let invBuf = try makeBuffer(device, invFreq)
    let cosBuf = try makeOutputBuffer(device, count: T * (Hd / 2))
    let sinBuf = try makeOutputBuffer(device, count: T * (Hd / 2))

    try runKernel(device, Kernels.RopeBuildCosSin(
        positions: posBuf, invFreq: invBuf,
        cosOut: cosBuf, sinOut: sinBuf,
        seqLen: T, halfHeadDim: Hd / 2,
        attentionScaling: 1.0
    ))

    let cos0 = readFloats(cosBuf, count: T * Hd / 2)
    let sin0 = readFloats(sinBuf, count: T * Hd / 2)
    // t=0 → angle=0 → cos=1, sin=0 for all i.
    #expect(approxEqual(cos0[0], 1)); #expect(approxEqual(cos0[1], 1))
    #expect(approxEqual(sin0[0], 0)); #expect(approxEqual(sin0[1], 0))
    // t=1 → angle = inv_freq[i]
    #expect(approxEqual(cos0[2], cos(Float(1.0))))
    #expect(approxEqual(sin0[2], sin(Float(1.0))))
    #expect(approxEqual(cos0[3], cos(Float(0.5))))
    #expect(approxEqual(sin0[3], sin(Float(0.5))))

    // Apply RoPE to a [B=1, T=2, Nh=1, Hd=4] tensor.
    // x = [[a0, a1, a2, a3], [b0, b1, b2, b3]]
    // For half-rotate: out[..., :Hd/2] = x1*cos - x2*sin,
    //                  out[..., Hd/2:] = x1*sin + x2*cos.
    let x: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
    let xBuf = try makeBuffer(device, x)
    let outBuf = try makeOutputBuffer(device, count: 8)
    try runKernel(device, Kernels.RopeApply(
        x: xBuf, cos: cosBuf, sin: sinBuf, out: outBuf,
        batch: 1, seqLen: T, headCount: 1, headDim: Hd
    ))
    let r = readFloats(outBuf, count: 8)
    // t=0: cos=1, sin=0 → unchanged.
    #expect(approxEqual(r[0], 1)); #expect(approxEqual(r[1], 2))
    #expect(approxEqual(r[2], 3)); #expect(approxEqual(r[3], 4))
    // t=1: angles (1, 0.5)
    let c0 = cos(Float(1.0)), s0 = sin(Float(1.0))
    let c1 = cos(Float(0.5)), s1 = sin(Float(0.5))
    #expect(approxEqual(r[4], 5 * c0 - 7 * s0))
    #expect(approxEqual(r[5], 6 * c1 - 8 * s1))
    #expect(approxEqual(r[6], 5 * s0 + 7 * c0))
    #expect(approxEqual(r[7], 6 * s1 + 8 * c1))
}

// MARK: - Attention scores / output

@Test func attentionScoresSoftmaxFused() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let q = try makeBuffer(device, [1, 0, 0, 1], shape: [1, 1, 2, 2])
    let k = try makeBuffer(device, [1, 0, 0, 1], shape: [1, 1, 2, 2])
    let probs = try makeOutputBuffer(device, count: 4, shape: [1, 1, 2, 2])
    try runKernel(device, Kernels.AttentionScoresSoftmax(
        q: q, k: k, into: probs, scale: 1
    ))
    let r = readFloats(probs, count: 4)
    #expect(r[0] == 1)
    #expect(r[1] == 0)
    let e = exp(Float(1))
    #expect(approxEqual(r[2], 1 / (1 + e)))
    #expect(approxEqual(r[3], e / (1 + e)))
}

@Test func attentionScoresAndOutputGQA() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    // B=1, Nq=2, Nkv=1 (so both query heads share the single KV head),
    // Sq=1, Sk=2, Hd=2.
    let Q: [Float] = [1, 0, // head 0
                      0, 1] // head 1
    let K: [Float] = [1, 0, // k=0
                      0, 1] // k=1
    let V: [Float] = [10, 20, // k=0
                      30, 40] // k=1
    let qBuf = try makeBuffer(device, Q)
    let kBuf = try makeBuffer(device, K)
    let vBuf = try makeBuffer(device, V)
    let scoresBuf = try makeOutputBuffer(device, count: 2 * 1 * 2)

    try runKernel(device, Kernels.AttentionScores(
        q: qBuf, k: kBuf, scores: scoresBuf,
        batch: 1, queryHeads: 2, kvHeads: 1,
        queryLen: 1, keyLen: 2, headDim: 2
    ))
    let s = readFloats(scoresBuf, count: 4)
    // Head 0 (Q=[1,0]) · K rows → [1, 0]
    // Head 1 (Q=[0,1]) · K rows → [0, 1]
    #expect(s == [1, 0, 0, 1])

    // Now use these as if they were softmax outputs (peaked) to verify
    // attention_output broadcasts the same KV head across query heads.
    let outBuf = try makeOutputBuffer(device, count: 2 * 1 * 2)
    try runKernel(device, Kernels.AttentionOutput(
        scores: scoresBuf, v: vBuf, out: outBuf,
        batch: 1, queryHeads: 2, kvHeads: 1,
        queryLen: 1, keyLen: 2, headDim: 2
    ))
    let o = readFloats(outBuf, count: 4)
    // Head 0: 1*V[0]+0*V[1] = [10,20]
    // Head 1: 0*V[0]+1*V[1] = [30,40]
    #expect(o == [10, 20, 30, 40])
}

// MARK: - KV cache write

@Test func kvCacheWrite() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    // B=1, Nkv=1, Snew=2, Smax=4, Hd=2, offset=1.
    let src: [Float] = [1, 2, 3, 4]
    let cache = try makeBuffer(device, [Float](repeating: 0, count: 8))
    let srcBuf = try makeBuffer(device, src)
    try runKernel(device, Kernels.KVCacheWrite(
        source: srcBuf, cache: cache,
        batch: 1, kvHeads: 1, newLen: 2, maxLen: 4, headDim: 2,
        offset: 1
    ))
    let r = readFloats(cache, count: 8)
    #expect(r == [0, 0, 1, 2, 3, 4, 0, 0])
}
