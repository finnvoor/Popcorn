import Metal
import PopcornShaderTypes

// MARK: - Kernels.Softmax

public extension Kernels {
    struct Softmax: Kernel {
        // MARK: Lifecycle

        public init(
            x: Tensor,
            out: Tensor,
            rowCount: Int,
            n: Int,
        ) throws {
            self.x = x
            self.out = out
            functionName = switch (x.dataType, out.dataType) {
            case (.f32, .f32): "softmax"
            case (.bf16, .bf16): "softmax_bf16"
            case (.bf16, .f32): "softmax_bf16_to_f32"
            case (.f32, .bf16): "softmax_f32_to_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported softmax data type combination: \(x.dataType), \(out.dataType).")
            }
            constants = [SoftmaxConstants(N: UInt32(n))]
            let tgSize = min(1024, max(32, nextPowerOfTwo(min(n, 256))))
            grid = MTLSize(width: tgSize * rowCount, height: 1, depth: 1)
            threadgroupSize = MTLSize(width: tgSize, height: 1, depth: 1)
        }

        public init(_ x: Tensor, into out: Tensor, axis: Int = -1) throws {
            let rank = x.shape.rank
            let normalizedAxis = axis < 0 ? rank + axis : axis
            guard normalizedAxis == rank - 1, rank > 0 else {
                throw PopcornError.tensorInvalidAxis(axis, rank: rank)
            }
            guard x.shape == out.shape else {
                throw PopcornError.tensorShapeMismatch("Softmax output shape must match input; got \(out.shape.dimensions), expected \(x.shape.dimensions).")
            }

            let n = x.shape[normalizedAxis]
            let rowCount = x.shape.elementCount / n
            try self.init(x: x, out: out, rowCount: rowCount, n: n)
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]
        public let grid: MTLSize
        public let threadgroupSize: MTLSize

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: x, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        // MARK: Private

        private let x: Tensor
        private let out: Tensor
    }
}

private func nextPowerOfTwo(_ n: Int) -> Int {
    var v = 1
    while v < n {
        v <<= 1
    }
    return v
}
