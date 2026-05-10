import Metal
import PopcornShaderTypes

public extension Kernels {
    struct Argmax: Kernel {
        // MARK: Lifecycle

        public init(
            x: Tensor,
            indices: Tensor,
            rowCount: Int,
            n: Int
        ) {
            self.x = x
            self.indices = indices
            constants = [ArgmaxConstants(rows: UInt32(rowCount), N: UInt32(n))]
            grid = MTLSize(width: rowCount, height: 1, depth: 1)
        }

        public init(_ x: Tensor, indices: Tensor, axis: Int = -1) throws {
            let rank = x.shape.rank
            let normalizedAxis = axis < 0 ? rank + axis : axis
            guard normalizedAxis == rank - 1, rank > 0 else {
                throw PopcornError.tensorInvalidAxis(axis, rank: rank)
            }

            let n = x.shape[normalizedAxis]
            let rowCount = x.shape.elementCount / n
            guard indices.shape.elementCount == rowCount else {
                throw PopcornError.tensorShapeMismatch("Argmax indices must contain \(rowCount) elements; got \(indices.shape.elementCount).")
            }
            self.init(x: x, indices: indices, rowCount: rowCount, n: n)
        }

        // MARK: Public

        public let functionName: String = "argmax"
        public let constants: [any BitwiseCopyable]
        public let grid: MTLSize
        public let threadgroupSize = MTLSize(width: 64, height: 1, depth: 1)

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: x, access: .read),
                .init(tensor: indices, access: .write)
            ]
        }

        // MARK: Private

        private let x: Tensor
        private let indices: Tensor
    }
}
