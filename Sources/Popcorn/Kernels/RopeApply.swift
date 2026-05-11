import Metal
import PopcornShaderTypes

public extension Kernels {
    struct RopeApply: Kernel {
        // MARK: Lifecycle

        public init(
            x: Tensor,
            cos: Tensor,
            sin: Tensor,
            out: Tensor,
            batch: Int,
            seqLen: Int,
            headCount: Int,
            headDim: Int,
        ) throws {
            guard headDim.isMultiple(of: 2) else {
                throw PopcornError.tensorShapeMismatch("RoPE head dimension must be even; got \(headDim).")
            }
            self.x = x
            self.cos = cos
            self.sin = sin
            self.out = out
            functionName = switch (x.dataType, cos.dataType) {
            case (.f32, .f32): "rope_apply"
            case (.bf16, .bf16): "rope_apply_bf16"
            case (.bf16, .f32): "rope_apply_f32_tables_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported RoPE data type combination: \(x.dataType), \(cos.dataType).")
            }
            constants = [RopeApplyConstants(
                B: UInt32(batch),
                T: UInt32(seqLen),
                Nh: UInt32(headCount),
                Hd2: UInt32(headDim / 2)
            )]
            dispatchGrid = MTLSize(
                width: batch * seqLen * headCount * (headDim / 2),
                height: 1,
                depth: 1
            )
        }

        public init(_ x: Tensor, cos: Tensor, sin: Tensor, into out: Tensor) throws {
            guard x.shape.rank == 4 else { throw PopcornError.tensorInvalidRank(expected: 4, actual: x.shape.rank) }
            guard x.shape == out.shape else {
                throw PopcornError.tensorShapeMismatch("RoPE output shape must match input; got \(out.shape.dimensions), expected \(x.shape.dimensions).")
            }
            let batch = x.shape[0], seqLen = x.shape[1], headCount = x.shape[2], headDim = x.shape[3]
            guard headDim.isMultiple(of: 2) else {
                throw PopcornError.tensorShapeMismatch("RoPE head dimension must be even; got \(headDim).")
            }
            let tableShape = [seqLen, headDim / 2]
            guard cos.shape.dimensions == tableShape, sin.shape.dimensions == tableShape else {
                throw PopcornError.tensorShapeMismatch("RoPE tables must have shape \(tableShape); got cos \(cos.shape.dimensions), sin \(sin.shape.dimensions).")
            }
            try self.init(x: x, cos: cos, sin: sin, out: out, batch: batch, seqLen: seqLen, headCount: headCount, headDim: headDim)
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: x, access: .read),
                .init(tensor: cos, access: .read),
                .init(tensor: sin, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        public func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            (dispatchGrid, MTLSize(width: 256, height: 1, depth: 1))
        }

        // MARK: Private

        private let dispatchGrid: MTLSize
        private let x: Tensor
        private let cos: Tensor
        private let sin: Tensor
        private let out: Tensor
    }
}
