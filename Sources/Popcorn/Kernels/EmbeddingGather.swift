import Metal
import PopcornShaderTypes

public extension Kernels {
    struct EmbeddingGather: Kernel {
        // MARK: Lifecycle

        public init(
            ids: Tensor,
            table: Tensor,
            out: Tensor,
            tokenCount: Int,
            hiddenSize: Int,
        ) throws {
            self.ids = ids
            self.table = table
            self.out = out
            functionName = switch (table.dataType, out.dataType) {
            case (.f32, .f32): "embedding_gather"
            case (.f16, .f16): "embedding_gather_f16"
            case (.bf16, .bf16): "embedding_gather_bf16"
            case (.bf16, .f32): "embedding_gather_bf16_to_f32"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported embedding gather data type combination: \(table.dataType), \(out.dataType).")
            }
            constants = [EmbeddingGatherConstants(N: UInt32(tokenCount), H: UInt32(hiddenSize))]
            dispatchGrid = MTLSize(width: tokenCount, height: hiddenSize, depth: 1)
        }

        public init(ids: Tensor, table: Tensor, into out: Tensor) throws {
            guard table.shape.rank == 2 else { throw PopcornError.tensorInvalidRank(expected: 2, actual: table.shape.rank) }
            guard ids.shape.elementCount == out.shape.dimensions.first else {
                throw PopcornError.tensorShapeMismatch("EmbeddingGather output first dimension must match ids element count; got \(out.shape.dimensions), ids count \(ids.shape.elementCount).")
            }
            guard out.shape.rank == 2, out.shape[1] == table.shape[1] else {
                throw PopcornError.tensorShapeMismatch("EmbeddingGather output shape must be [ids.count, \(table.shape[1])]; got \(out.shape.dimensions).")
            }
            try self.init(ids: ids, table: table, out: out, tokenCount: ids.shape.elementCount, hiddenSize: table.shape[1])
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: ids, access: .read),
                .init(tensor: table, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        public func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            (dispatchGrid, MTLSize(width: 1, height: 64, depth: 1))
        }

        // MARK: Private

        private let dispatchGrid: MTLSize
        private let ids: Tensor
        private let table: Tensor
        private let out: Tensor
    }
}
