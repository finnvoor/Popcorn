import Metal
@testable import Popcorn
import Testing

@Test func layerNormMatchesReference() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let commandQueue = try #require(device.makeCommandQueue())
    let commandBuffer = try #require(commandQueue.makeCommandBuffer())
    let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

    let xs: [Float] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        -3.0, -1.5, 0.25, 0.0, 0.5, 1.0, 1.5, 2.0
    ]
    let w: [Float] = [0.5, 1.0, 1.5, 2.0, 0.25, 0.75, 1.25, 1.75]
    let b: [Float] = [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4]
    let H = 8
    let rows = xs.count / H
    let eps: Float = 1e-5

    let xLen = xs.count * MemoryLayout<Float>.stride
    let wLen = w.count * MemoryLayout<Float>.stride
    let bLen = b.count * MemoryLayout<Float>.stride
    let bx = try #require(device.makeBuffer(bytes: xs, length: xLen, options: .storageModeShared))
    let bw = try #require(device.makeBuffer(bytes: w, length: wLen, options: .storageModeShared))
    let bb = try #require(device.makeBuffer(bytes: b, length: bLen, options: .storageModeShared))
    let bo = try #require(device.makeBuffer(length: xLen, options: .storageModeShared))

    let x = Tensor(buffer: bx, shape: [rows, H], dataType: .f32)
    let wt = Tensor(buffer: bw, shape: [H], dataType: .f32)
    let bt = Tensor(buffer: bb, shape: [H], dataType: .f32)
    let out = Tensor(buffer: bo, shape: [rows, H], dataType: .f32)

    let lib = try KernelLibrary(device: device)
    let kenc = KernelCommandEncoder(compute: encoder, library: lib, scratch: TestScratchAllocator(device: device))
    try kenc.encode(Kernels.LayerNorm(x, weight: wt, bias: bt, into: out, eps: eps))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    var expected = [Float](repeating: 0, count: xs.count)
    for r in 0..<rows {
        var mean: Float = 0
        for i in 0..<H {
            mean += xs[r * H + i]
        }
        mean /= Float(H)
        var v: Float = 0
        for i in 0..<H {
            let d = xs[r * H + i] - mean
            v += d * d
        }
        v /= Float(H)
        let inv: Float = 1.0 / (v + eps).squareRoot()
        for i in 0..<H {
            expected[r * H + i] = (xs[r * H + i] - mean) * inv * w[i] + b[i]
        }
    }

    let ptr = bo.contents().bindMemory(to: Float.self, capacity: expected.count)
    for i in expected.indices {
        let diff = abs(ptr[i] - expected[i])
        #expect(diff < 1e-4)
    }
}

private func referenceGroupNorm(
    x: [Float], n: Int, c: Int, l: Int, groups: Int,
    weight: [Float], bias: [Float], eps: Float
) -> [Float] {
    let cPerGroup = c / groups
    var out = [Float](repeating: 0, count: x.count)
    for ni in 0..<n {
        for g in 0..<groups {
            let cStart = g * cPerGroup
            var sum: Float = 0
            var count: Float = 0
            for ci in cStart..<(cStart + cPerGroup) {
                for li in 0..<l {
                    sum += x[(ni * c + ci) * l + li]
                    count += 1
                }
            }
            let mean = sum / count
            var v: Float = 0
            for ci in cStart..<(cStart + cPerGroup) {
                for li in 0..<l {
                    let d = x[(ni * c + ci) * l + li] - mean
                    v += d * d
                }
            }
            let inv: Float = 1.0 / (v / count + eps).squareRoot()
            for ci in cStart..<(cStart + cPerGroup) {
                for li in 0..<l {
                    out[(ni * c + ci) * l + li] = (x[(ni * c + ci) * l + li] - mean) * inv * weight[ci] + bias[ci]
                }
            }
        }
    }
    return out
}

@Test func groupNormMatchesReference() throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let commandQueue = try #require(device.makeCommandQueue())
    let commandBuffer = try #require(commandQueue.makeCommandBuffer())
    let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

    let N = 1
    let C = 8
    let L = 5
    let groups = 4
    let eps: Float = 1e-5

    var xs = [Float]()
    for i in 0..<(N * C * L) {
        xs.append(Float(i % 11) - 5.0 + Float(i) * 0.01)
    }
    var w = [Float]()
    var b = [Float]()
    for c in 0..<C {
        w.append(0.5 + Float(c) * 0.1)
        b.append(-0.1 + Float(c) * 0.05)
    }

    let expected = referenceGroupNorm(
        x: xs, n: N, c: C, l: L, groups: groups,
        weight: w, bias: b, eps: eps
    )

    let s = MemoryLayout<Float>.stride
    let xLen = xs.count * s
    let wLen = w.count * s
    let bLen = b.count * s
    let bx = try #require(device.makeBuffer(bytes: xs, length: xLen, options: .storageModeShared))
    let bw = try #require(device.makeBuffer(bytes: w, length: wLen, options: .storageModeShared))
    let bb = try #require(device.makeBuffer(bytes: b, length: bLen, options: .storageModeShared))
    let bo = try #require(device.makeBuffer(length: xLen, options: .storageModeShared))

    let x = Tensor(buffer: bx, shape: [N, C, L], dataType: .f32)
    let wt = Tensor(buffer: bw, shape: [C], dataType: .f32)
    let bt = Tensor(buffer: bb, shape: [C], dataType: .f32)
    let out = Tensor(buffer: bo, shape: [N, C, L], dataType: .f32)

    let lib = try KernelLibrary(device: device)
    let kenc = KernelCommandEncoder(compute: encoder, library: lib, scratch: TestScratchAllocator(device: device))
    try kenc.encode(Kernels.GroupNorm(x, weight: wt, bias: bt, into: out, groups: groups, eps: eps))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let ptr = bo.contents().bindMemory(to: Float.self, capacity: expected.count)
    for i in expected.indices {
        let diff = abs(ptr[i] - expected[i])
        #expect(diff < 1e-4)
    }
}
