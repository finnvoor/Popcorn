import Metal
import PopcornShaderTypes

public extension Kernels {
    struct AttentionMaskBuild: Kernel {
        // MARK: Lifecycle

        public init(
            mask: Tensor,
            queryLen: Int,
            keyLen: Int,
            slidingWindow: Int?
        ) {
            self.mask = mask
            constants = [AttentionMaskBuildConstants(
                Sq: UInt32(queryLen),
                Sk: UInt32(keyLen),
                slidingWindow: Int32(slidingWindow ?? -1)
            )]
            grid = MTLSize(width: queryLen, height: keyLen, depth: 1)
        }

        public init(mask: Tensor, slidingWindow: Int? = nil) throws {
            guard mask.shape.rank == 2 else {
                throw PopcornError.tensorInvalidRank(expected: 2, actual: mask.shape.rank)
            }
            self.init(
                mask: mask,
                queryLen: mask.shape[0],
                keyLen: mask.shape[1],
                slidingWindow: slidingWindow
            )
        }

        // MARK: Public

        public let functionName: String = "attention_mask_build"
        public let constants: [any BitwiseCopyable]
        public let grid: MTLSize
        public let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: mask, access: .write)
            ]
        }

        // MARK: Private

        private let mask: Tensor
    }
}
