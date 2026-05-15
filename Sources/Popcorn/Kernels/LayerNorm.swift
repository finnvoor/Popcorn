import Metal
import PopcornShaderTypes

public extension Kernels {
    /// LayerNorm over the last axis with optional learnable weight and bias.
    /// Matches `nn.LayerNorm(H, eps=...)` from PyTorch when both are provided.
    struct LayerNorm: DispatchKernel {
        // MARK: Lifecycle

        public init(
            x: Tensor,
            weight: Tensor?,
            bias: Tensor?,
            out: Tensor,
            rowCount: Int,
            hiddenSize: Int,
            eps: Float
        ) throws {
            self.x = x
            self.weight = weight ?? x
            self.bias = bias ?? x
            self.out = out
            functionName = switch x.dataType {
            case .f32: "layernorm"
            case .f16: "layernorm_f16"
            case .bf16: "layernorm_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported LayerNorm data type: \(x.dataType).")
            }
            constants = [LayerNormConstants(
                H: UInt32(hiddenSize),
                hasWeight: weight == nil ? 0 : 1,
                hasBias: bias == nil ? 0 : 1,
                eps: eps
            )]
            self.rowCount = rowCount
            self.hiddenSize = hiddenSize
        }

        public init(
            _ x: Tensor,
            weight: Tensor? = nil,
            bias: Tensor? = nil,
            into out: Tensor,
            eps: Float
        ) throws {
            let rank = x.shape.rank
            guard rank > 0 else { throw PopcornError.tensorInvalidAxis(-1, rank: rank) }
            guard x.shape == out.shape else {
                throw PopcornError.tensorShapeMismatch("LayerNorm output shape must match input; got \(out.shape.dimensions), expected \(x.shape.dimensions).")
            }
            let hiddenSize = x.shape[rank - 1]
            let rowCount = x.shape.elementCount / hiddenSize
            if let weight, weight.shape.dimensions != [hiddenSize] {
                throw PopcornError.tensorShapeMismatch("LayerNorm weight must be [\(hiddenSize)]; got \(weight.shape.dimensions).")
            }
            if let bias, bias.shape.dimensions != [hiddenSize] {
                throw PopcornError.tensorShapeMismatch("LayerNorm bias must be [\(hiddenSize)]; got \(bias.shape.dimensions).")
            }
            try self.init(
                x: x, weight: weight, bias: bias, out: out,
                rowCount: rowCount, hiddenSize: hiddenSize, eps: eps
            )
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: x, access: .read),
                .init(tensor: weight, access: .read),
                .init(tensor: bias, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        public func dispatchSize(for pipelineState: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            DispatchSize.reduction(rowCount: rowCount, n: hiddenSize, pipelineState: pipelineState)
        }

        // MARK: Private

        private let x: Tensor
        private let weight: Tensor
        private let bias: Tensor
        private let out: Tensor
        private let rowCount: Int
        private let hiddenSize: Int
    }
}
