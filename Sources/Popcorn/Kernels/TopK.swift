import Metal
import PopcornShaderTypes

public extension Kernels {
    struct TopK: Kernel {
        // MARK: Lifecycle

        public init(
            x: Tensor,
            values: Tensor,
            indices: Tensor,
            rowCount: Int,
            elementCount: Int,
            k: Int
        ) throws {
            guard k > 0, k <= 32 else {
                throw PopcornError.tensorShapeMismatch("TopK kernel supports k in 1...32; got \(k).")
            }
            self.x = x
            self.values = values
            self.indices = indices
            constants = [TopKConstants(
                rows: UInt32(rowCount),
                E: UInt32(elementCount),
                K: UInt32(k)
            )]
            grid = MTLSize(width: rowCount, height: 1, depth: 1)
        }

        public init(_ x: Tensor, values: Tensor, indices: Tensor, axis: Int = -1) throws {
            let rank = x.shape.rank
            let normalizedAxis = axis < 0 ? rank + axis : axis
            guard normalizedAxis == rank - 1, rank > 0 else { throw PopcornError.tensorInvalidAxis(axis, rank: rank) }
            guard values.shape == indices.shape else {
                throw PopcornError.tensorShapeMismatch("TopK values and indices shapes must match; got \(values.shape.dimensions) and \(indices.shape.dimensions).")
            }
            let elementCount = x.shape[normalizedAxis]
            let rowCount = x.shape.elementCount / elementCount
            guard values.shape.rank > 0, values.shape.elementCount == rowCount * values.shape[values.shape.rank - 1] else {
                throw PopcornError.tensorShapeMismatch("TopK values shape is invalid: \(values.shape.dimensions).")
            }
            let k = values.shape[values.shape.rank - 1]
            guard values.shape.elementCount == rowCount * k else {
                throw PopcornError.tensorShapeMismatch("TopK output must contain rowCount * k elements; got \(values.shape.elementCount).")
            }
            try self.init(x: x, values: values, indices: indices, rowCount: rowCount, elementCount: elementCount, k: k)
        }

        // MARK: Public

        public let functionName: String = "topk"
        public let constants: [any BitwiseCopyable]
        public let grid: MTLSize
        public let threadgroupSize = MTLSize(width: 64, height: 1, depth: 1)

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: x, access: .read),
                .init(tensor: values, access: .write),
                .init(tensor: indices, access: .write)
            ]
        }

        // MARK: Private

        private let x: Tensor
        private let values: Tensor
        private let indices: Tensor
    }
}
