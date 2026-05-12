import Metal
import PopcornShaderTypes

// MARK: - Kernels.AttentionScoresSoftmax

public extension Kernels {
    struct AttentionScoresSoftmax: DispatchKernel {
        // MARK: Lifecycle

        public init(
            q: Tensor,
            k: Tensor,
            probs: Tensor,
            batch: Int,
            queryHeads: Int,
            kvHeads: Int,
            queryLen: Int,
            keyLen: Int,
            headDim: Int,
            scale: Float,
            slidingWindow: Int? = nil
        ) throws {
            guard queryHeads.isMultiple(of: kvHeads) else {
                throw PopcornError.tensorShapeMismatch("AttentionScoresSoftmax query heads must be divisible by KV heads; got \(queryHeads) and \(kvHeads).")
            }
            self.q = q
            self.k = k
            self.probs = probs
            functionName = switch (q.dataType, k.dataType, probs.dataType) {
            case (.f32, .f32, .f32): "attention_scores_softmax"
            case (.bf16, .bf16, .f32): "attention_scores_softmax_bf16_to_f32"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported attention scores softmax data type combination: \(q.dataType), \(k.dataType), \(probs.dataType).")
            }
            constants = [AttentionScoresSoftmaxConstants(
                B: UInt32(batch), Nq: UInt32(queryHeads), Nkv: UInt32(kvHeads),
                Sq: UInt32(queryLen), Sk: UInt32(keyLen), Hd: UInt32(headDim),
                slidingWindow: Int32(slidingWindow ?? -1), scale: scale
            )]
            rowCount = batch * queryHeads * queryLen
            self.keyLen = keyLen
        }

        public init(
            q: Tensor,
            k: Tensor,
            into probs: Tensor,
            scale: Float,
            slidingWindow: Int? = nil
        ) throws {
            guard q.shape.rank == 4 else { throw PopcornError.tensorInvalidRank(expected: 4, actual: q.shape.rank) }
            guard k.shape.rank == 4 else { throw PopcornError.tensorInvalidRank(expected: 4, actual: k.shape.rank) }
            let batch = q.shape[0], queryHeads = q.shape[1], queryLen = q.shape[2], headDim = q.shape[3]
            let kvHeads = k.shape[1], keyLen = k.shape[2]
            guard k.shape[0] == batch, k.shape[3] == headDim else {
                throw PopcornError.tensorShapeMismatch("AttentionScoresSoftmax K shape must match Q batch/headDim; q \(q.shape.dimensions), k \(k.shape.dimensions).")
            }
            let expectedProbs = [batch, queryHeads, queryLen, keyLen]
            guard probs.shape.dimensions == expectedProbs else {
                throw PopcornError.tensorShapeMismatch("AttentionScoresSoftmax output shape must be \(expectedProbs); got \(probs.shape.dimensions).")
            }
            try self.init(
                q: q, k: k, probs: probs,
                batch: batch, queryHeads: queryHeads, kvHeads: kvHeads,
                queryLen: queryLen, keyLen: keyLen, headDim: headDim,
                scale: scale, slidingWindow: slidingWindow
            )
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: q, access: .read),
                .init(tensor: k, access: .read),
                .init(tensor: probs, access: .write)
            ]
        }

        public func dispatchSize(for pipelineState: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            DispatchSize.reduction(rowCount: rowCount, n: keyLen, pipelineState: pipelineState)
        }

        // MARK: Private

        private let q: Tensor
        private let k: Tensor
        private let probs: Tensor
        private let rowCount: Int
        private let keyLen: Int
    }
}
