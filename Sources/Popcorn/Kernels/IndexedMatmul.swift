import Metal
import PopcornShaderTypes

public extension Kernels {
    struct IndexedMatmul: Kernel {
        // MARK: Lifecycle

        public init(
            x: Tensor,
            weights: Tensor,
            expertIndex: Tensor,
            out: Tensor,
            rowCount: Int,
            inDim: Int,
            outDim: Int,
            transposeW: Bool = false
        ) {
            self.x = x
            self.weights = weights
            self.expertIndex = expertIndex
            self.out = out
            constants = [IndexedMatmulConstants(
                N: UInt32(rowCount),
                K: UInt32(inDim),
                M: UInt32(outDim),
                transposeW: transposeW ? 1 : 0
            )]
            self.rowCount = rowCount
            columnCount = outDim
        }

        public init(_ x: Tensor, weights: Tensor, expertIndex: Tensor, into out: Tensor, transposeW: Bool = false) throws {
            guard x.shape.rank == 2 else { throw PopcornError.tensorInvalidRank(expected: 2, actual: x.shape.rank) }
            guard weights.shape.rank == 3 else { throw PopcornError.tensorInvalidRank(expected: 3, actual: weights.shape.rank) }
            guard expertIndex.shape.elementCount == x.shape[0] else {
                throw PopcornError.tensorShapeMismatch("IndexedMatmul expertIndex count must equal row count \(x.shape[0]); got \(expertIndex.shape.elementCount).")
            }
            let rowCount = x.shape[0]
            let inDim = x.shape[1]
            let outDim = transposeW ? weights.shape[1] : weights.shape[2]
            let expectedWeightsTail = transposeW ? [outDim, inDim] : [inDim, outDim]
            guard Array(weights.shape.dimensions.dropFirst()) == expectedWeightsTail else {
                throw PopcornError.tensorShapeMismatch("IndexedMatmul weights trailing shape must be \(expectedWeightsTail); got \(Array(weights.shape.dimensions.dropFirst())).")
            }
            guard out.shape.dimensions == [rowCount, outDim] else {
                throw PopcornError.tensorShapeMismatch("IndexedMatmul output shape must be [\(rowCount), \(outDim)]; got \(out.shape.dimensions).")
            }
            self.init(x: x, weights: weights, expertIndex: expertIndex, out: out, rowCount: rowCount, inDim: inDim, outDim: outDim, transposeW: transposeW)
        }

        // MARK: Public

        public let functionName: String = "indexed_matmul"
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: x, access: .read),
                .init(tensor: weights, access: .read),
                .init(tensor: expertIndex, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        public func dispatchSize(for pipelineState: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            DispatchSize.rowsColumns(rowCount: rowCount, columnCount: columnCount, pipelineState: pipelineState)
        }

        // MARK: Private

        private let rowCount: Int
        private let columnCount: Int
        private let x: Tensor
        private let weights: Tensor
        private let expertIndex: Tensor
        private let out: Tensor
    }
}
