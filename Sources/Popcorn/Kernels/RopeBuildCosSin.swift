import Metal
import PopcornShaderTypes

public extension Kernels {
    struct RopeBuildCosSin: Kernel {
        // MARK: Lifecycle

        public init(
            positions: Tensor,
            invFreq: Tensor,
            cosOut: Tensor,
            sinOut: Tensor,
            seqLen: Int,
            halfHeadDim: Int,
            attentionScaling: Float
        ) {
            self.positions = positions
            self.invFreq = invFreq
            self.cosOut = cosOut
            self.sinOut = sinOut
            constants = [RopeBuildCosSinConstants(
                T: UInt32(seqLen),
                Hd2: UInt32(halfHeadDim),
                scaling: attentionScaling
            )]
            grid = MTLSize(width: seqLen, height: halfHeadDim, depth: 1)
        }

        public init(
            positions: Tensor,
            invFreq: Tensor,
            cosOut: Tensor,
            sinOut: Tensor,
            attentionScaling: Float
        ) throws {
            guard cosOut.shape == sinOut.shape else {
                throw PopcornError.tensorShapeMismatch("RoPE cos and sin output shapes must match; got \(cosOut.shape.dimensions) and \(sinOut.shape.dimensions).")
            }
            guard cosOut.shape.rank == 2 else { throw PopcornError.tensorInvalidRank(expected: 2, actual: cosOut.shape.rank) }
            let seqLen = cosOut.shape[0]
            let halfHeadDim = cosOut.shape[1]
            guard positions.shape.elementCount == seqLen else {
                throw PopcornError.tensorShapeMismatch("RoPE positions count must equal seqLen \(seqLen); got \(positions.shape.elementCount).")
            }
            guard invFreq.shape.elementCount == halfHeadDim else {
                throw PopcornError.tensorShapeMismatch("RoPE invFreq count must equal halfHeadDim \(halfHeadDim); got \(invFreq.shape.elementCount).")
            }
            self.init(positions: positions, invFreq: invFreq, cosOut: cosOut, sinOut: sinOut, seqLen: seqLen, halfHeadDim: halfHeadDim, attentionScaling: attentionScaling)
        }

        // MARK: Public

        public let functionName: String = "rope_build_cos_sin"
        public let constants: [any BitwiseCopyable]
        public let grid: MTLSize
        public let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: positions, access: .read),
                .init(tensor: invFreq, access: .read),
                .init(tensor: cosOut, access: .write),
                .init(tensor: sinOut, access: .write)
            ]
        }

        // MARK: Private

        private let positions: Tensor
        private let invFreq: Tensor
        private let cosOut: Tensor
        private let sinOut: Tensor
    }
}
