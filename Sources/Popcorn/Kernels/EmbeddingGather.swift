import Metal
import PopcornShaderTypes

public extension Kernels {
    struct EmbeddingGather: DispatchKernel {
        // MARK: Lifecycle

        public init(
            ids: Tensor,
            table: Tensor,
            out: Tensor,
            tokenCount: Int,
            hiddenSize: Int,
        ) throws {
            self.ids = ids
            storage = .dense(table)
            self.out = out
            functionName = switch (table.dataType, out.dataType) {
            case (.f32, .f32): "embedding_gather"
            case (.f16, .f16): "embedding_gather_f16"
            case (.bf16, .bf16): "embedding_gather_bf16"
            case (.bf16, .f32): "embedding_gather_bf16_to_f32"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported embedding gather data type combination: \(table.dataType), \(out.dataType).")
            }
            constants = [EmbeddingGatherConstants(N: UInt32(tokenCount), H: UInt32(hiddenSize))]
            dispatchGrid = MTLSize(width: tokenCount, height: (hiddenSize + 3) / 4, depth: 1)
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

        /// Gathers rows from a block-affine quantized embedding table, dequantizing on the fly.
        ///
        /// Equivalent to `out[i, :] = dequantize(table[ids[i], :])`. `table` has logical
        /// shape `[vocabSize, hiddenSize]`, `ids` is rank-1 `i32`, and `out` is
        /// `[ids.count, hiddenSize]`.
        public init(ids: Tensor, table: AffineQuantizedTensor, into out: Tensor) throws {
            guard ids.shape.rank == 1 else {
                throw PopcornError.tensorInvalidRank(expected: 1, actual: ids.shape.rank)
            }
            guard ids.dataType == .i32 else {
                throw PopcornError.unsupportedDataTypeCombination("EmbeddingGather with quantized table requires i32 ids; got \(ids.dataType).")
            }
            guard out.shape.rank == 2 else {
                throw PopcornError.tensorInvalidRank(expected: 2, actual: out.shape.rank)
            }

            let tokenCount = ids.shape[0]
            let hiddenSize = table.inFeatures
            guard out.shape.dimensions == [tokenCount, hiddenSize] else {
                throw PopcornError.tensorShapeMismatch(
                    "EmbeddingGather output shape mismatch: expected [\(tokenCount),\(hiddenSize)], got \(out.shape.dimensions)."
                )
            }
            guard hiddenSize % table.format.groupSize == 0 else {
                throw PopcornError.tensorShapeMismatch(
                    "EmbeddingGather quantized table hiddenSize=\(hiddenSize) is not divisible by groupSize=\(table.format.groupSize)."
                )
            }
            guard table.format.bits == 4, table.format.groupSize == 64, table.format.packing == .uint32LittleEndian else {
                throw PopcornError.unsupportedDataTypeCombination(
                    "EmbeddingGather currently only supports affine quantized tables with bits=4, groupSize=64, packing=.uint32LittleEndian."
                )
            }

            self.ids = ids
            storage = .affineQuantized(packedValues: table.packedValues, scales: table.scales, biases: table.biases ?? table.scales)
            self.out = out

            let sTag = Self.dtypeTag(table.scales.dataType)
            let oTag = Self.dtypeTag(out.dataType)
            functionName = "aq_embedding_gather_\(sTag)_\(oTag)_b\(table.format.bits)_g\(table.format.groupSize)"

            let perWord = table.format.valuesPerPackedElement
            constants = [AffineQEmbeddingGatherConstants(
                T: UInt32(tokenCount),
                H: UInt32(hiddenSize),
                kGroups: UInt32(hiddenSize / table.format.groupSize),
                wordsPerRow: UInt32(hiddenSize / perWord),
                hasBias: table.biases == nil ? 0 : 1
            )]
            dispatchGrid = MTLSize(width: tokenCount, height: hiddenSize, depth: 1)
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            switch storage {
            case let .dense(table):
                [
                    .init(tensor: ids, access: .read),
                    .init(tensor: table, access: .read),
                    .init(tensor: out, access: .write)
                ]
            case let .affineQuantized(packedValues, scales, biases):
                [
                    .init(tensor: ids, access: .read),
                    .init(tensor: packedValues, access: .read),
                    .init(tensor: scales, access: .read),
                    .init(tensor: biases, access: .read),
                    .init(tensor: out, access: .write)
                ]
            }
        }

        public func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            (dispatchGrid, MTLSize(width: 1, height: 64, depth: 1))
        }

        // MARK: Private

        private enum Storage {
            case dense(Tensor)
            case affineQuantized(packedValues: Tensor, scales: Tensor, biases: Tensor)
        }

        private let dispatchGrid: MTLSize
        private let ids: Tensor
        private let storage: Storage
        private let out: Tensor

        private static func dtypeTag(_ dt: Tensor.DataType) -> String {
            switch dt {
            case .f32: "f32"
            case .f16: "f16"
            case .bf16: "bf16"
            default: "unknown"
            }
        }
    }
}
