import Metal
import PopcornShaderTypes

public extension Kernels {
    struct AttentionScores: Kernel {
        // MARK: Lifecycle

        public init(
            q: Tensor,
            k: Tensor,
            scores: Tensor,
            batch: Int,
            queryHeads: Int,
            kvHeads: Int,
            queryLen: Int,
            keyLen: Int,
            headDim: Int,
        ) throws {
            guard queryHeads.isMultiple(of: kvHeads) else {
                throw PopcornError.tensorShapeMismatch("AttentionScores query heads must be divisible by KV heads; got \(queryHeads) and \(kvHeads).")
            }
            self.q = q
            self.k = k
            self.scores = scores
            functionName = switch (q.dataType, scores.dataType) {
            case (.f32, .f32): "attention_scores"
            case (.bf16, .bf16): "attention_scores_bf16"
            case (.bf16, .f32): "attention_scores_bf16_to_f32"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported attention scores data type combination: \(q.dataType), \(scores.dataType).")
            }
            constants = [AttentionScoresConstants(
                B: UInt32(batch), Nq: UInt32(queryHeads), Nkv: UInt32(kvHeads),
                Sq: UInt32(queryLen), Sk: UInt32(keyLen), Hd: UInt32(headDim)
            )]
            dispatchGrid = MTLSize(width: batch, height: queryHeads, depth: queryLen * keyLen)
        }

        public init(q: Tensor, k: Tensor, into scores: Tensor) throws {
            guard q.shape.rank == 4 else { throw PopcornError.tensorInvalidRank(expected: 4, actual: q.shape.rank) }
            guard k.shape.rank == 4 else { throw PopcornError.tensorInvalidRank(expected: 4, actual: k.shape.rank) }
            let batch = q.shape[0], queryHeads = q.shape[1], queryLen = q.shape[2], headDim = q.shape[3]
            let kvHeads = k.shape[1], keyLen = k.shape[2]
            guard k.shape[0] == batch, k.shape[3] == headDim else {
                throw PopcornError.tensorShapeMismatch("AttentionScores K shape must match Q batch/headDim; q \(q.shape.dimensions), k \(k.shape.dimensions).")
            }
            guard queryHeads.isMultiple(of: kvHeads) else {
                throw PopcornError.tensorShapeMismatch("AttentionScores query heads must be divisible by KV heads; got \(queryHeads) and \(kvHeads).")
            }
            let expectedScores = [batch, queryHeads, queryLen, keyLen]
            guard scores.shape.dimensions == expectedScores else {
                throw PopcornError.tensorShapeMismatch("AttentionScores output shape must be \(expectedScores); got \(scores.shape.dimensions).")
            }
            try self.init(q: q, k: k, scores: scores, batch: batch, queryHeads: queryHeads, kvHeads: kvHeads, queryLen: queryLen, keyLen: keyLen, headDim: headDim)
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: q, access: .read),
                .init(tensor: k, access: .read),
                .init(tensor: scores, access: .write)
            ]
        }

        public func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            (dispatchGrid, MTLSize(width: 1, height: 1, depth: 64))
        }

        // MARK: Private

        private let dispatchGrid: MTLSize
        private let q: Tensor
        private let k: Tensor
        private let scores: Tensor
    }
}
