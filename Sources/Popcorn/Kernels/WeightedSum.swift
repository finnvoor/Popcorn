import Metal
import PopcornShaderTypes

public extension Kernels {
    struct WeightedSum: Kernel {
        // MARK: Lifecycle

        public init(
            contrib: Tensor,
            weights: Tensor,
            out: Tensor,
            rowCount: Int,
            k: Int,
            hiddenSize: Int
        ) {
            self.contrib = contrib
            self.weights = weights
            self.out = out
            constants = [WeightedSumConstants(
                rows: UInt32(rowCount),
                K: UInt32(k),
                H: UInt32(hiddenSize)
            )]
            dispatchGrid = MTLSize(width: rowCount, height: hiddenSize, depth: 1)
        }

        public init(contrib: Tensor, weights: Tensor, into out: Tensor) throws {
            guard contrib.shape.rank == 3 else { throw PopcornError.tensorInvalidRank(expected: 3, actual: contrib.shape.rank) }
            let rows = contrib.shape[0]
            let k = contrib.shape[1]
            let hiddenSize = contrib.shape[2]
            guard weights.shape.dimensions == [rows, k] else {
                throw PopcornError.tensorShapeMismatch("WeightedSum weights shape must be [\(rows), \(k)]; got \(weights.shape.dimensions).")
            }
            guard out.shape.dimensions == [rows, hiddenSize] else {
                throw PopcornError.tensorShapeMismatch("WeightedSum output shape must be [\(rows), \(hiddenSize)]; got \(out.shape.dimensions).")
            }
            self.init(contrib: contrib, weights: weights, out: out, rowCount: rows, k: k, hiddenSize: hiddenSize)
        }

        // MARK: Public

        public let functionName: String = "weighted_sum"
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: contrib, access: .read),
                .init(tensor: weights, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        public func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            (dispatchGrid, MTLSize(width: 8, height: 8, depth: 1))
        }

        // MARK: Private

        private let dispatchGrid: MTLSize
        private let contrib: Tensor
        private let weights: Tensor
        private let out: Tensor
    }
}
