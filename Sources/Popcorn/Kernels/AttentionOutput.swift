import Metal
import PopcornShaderTypes

public extension Kernels {
    struct AttentionOutput: DispatchKernel {
        // MARK: Lifecycle

        public init(
            scores: Tensor,
            v: Tensor,
            out: Tensor,
            batch: Int,
            queryHeads: Int,
            kvHeads: Int,
            queryLen: Int,
            keyLen: Int,
            headDim: Int,
        ) throws {
            guard queryHeads.isMultiple(of: kvHeads) else {
                throw PopcornError.tensorShapeMismatch("AttentionOutput query heads must be divisible by KV heads; got \(queryHeads) and \(kvHeads).")
            }
            self.scores = scores
            self.v = v
            self.out = out
            functionName = switch (scores.dataType, v.dataType, out.dataType) {
            case (.f32, .f32, .f32): "attention_output"
            case (.bf16, .bf16, .bf16): "attention_output_bf16"
            case (.bf16, .bf16, .f32): "attention_output_bf16_to_f32"
            case (.f32, .bf16, .bf16): "attention_output_f32_bf16_to_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported attention output data type combination: \(scores.dataType), \(v.dataType), \(out.dataType).")
            }
            constants = [AttentionOutputConstants(
                B: UInt32(batch), Nq: UInt32(queryHeads), Nkv: UInt32(kvHeads),
                Sq: UInt32(queryLen), Sk: UInt32(keyLen), Hd: UInt32(headDim)
            )]
            dispatchGrid = MTLSize(width: batch, height: queryHeads, depth: queryLen * headDim)
        }

        public init(scores: Tensor, v: Tensor, into out: Tensor) throws {
            guard scores.shape.rank == 4 else { throw PopcornError.tensorInvalidRank(expected: 4, actual: scores.shape.rank) }
            guard v.shape.rank == 4 else { throw PopcornError.tensorInvalidRank(expected: 4, actual: v.shape.rank) }
            let batch = scores.shape[0], queryHeads = scores.shape[1], queryLen = scores.shape[2], keyLen = scores.shape[3]
            let kvHeads = v.shape[1], headDim = v.shape[3]
            guard v.shape[0] == batch, v.shape[2] == keyLen else {
                throw PopcornError.tensorShapeMismatch("AttentionOutput V shape must match scores batch/keyLen; scores \(scores.shape.dimensions), v \(v.shape.dimensions).")
            }
            guard queryHeads.isMultiple(of: kvHeads) else {
                throw PopcornError.tensorShapeMismatch("AttentionOutput query heads must be divisible by KV heads; got \(queryHeads) and \(kvHeads).")
            }
            let expectedOut = [batch, queryHeads, queryLen, headDim]
            guard out.shape.dimensions == expectedOut else {
                throw PopcornError.tensorShapeMismatch("AttentionOutput output shape must be \(expectedOut); got \(out.shape.dimensions).")
            }
            try self.init(scores: scores, v: v, out: out, batch: batch, queryHeads: queryHeads, kvHeads: kvHeads, queryLen: queryLen, keyLen: keyLen, headDim: headDim)
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: scores, access: .read),
                .init(tensor: v, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        public func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            (dispatchGrid, MTLSize(width: 1, height: 1, depth: 64))
        }

        // MARK: Private

        private let dispatchGrid: MTLSize
        private let scores: Tensor
        private let v: Tensor
        private let out: Tensor
    }
}
