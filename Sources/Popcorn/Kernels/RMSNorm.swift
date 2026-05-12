import Metal
import PopcornShaderTypes

// MARK: - Kernels.RMSNorm

public extension Kernels {
    struct RMSNorm: DispatchKernel {
        // MARK: Lifecycle

        public init(
            x: Tensor,
            weight: Tensor?,
            out: Tensor,
            rowCount: Int,
            hiddenSize: Int,
            eps: Float,
            addOneToWeight: Bool,
        ) throws {
            self.x = x

            self.weight = weight ?? x
            self.out = out
            functionName = switch x.dataType {
            case .f32: "rmsnorm"
            case .f16: "rmsnorm_f16"
            case .bf16: "rmsnorm_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported RMSNorm data type: \(x.dataType).")
            }
            constants = [RMSNormConstants(
                H: UInt32(hiddenSize),
                hasWeight: weight == nil ? 0 : 1,
                addOneToWeight: addOneToWeight ? 1 : 0,
                eps: eps
            )]
            self.rowCount = rowCount
            self.hiddenSize = hiddenSize
        }

        public init(
            _ x: Tensor,
            weight: Tensor? = nil,
            into out: Tensor,
            axis: Int = -1,
            eps: Float,
            addOneToWeight: Bool = false
        ) throws {
            let rank = x.shape.rank
            let normalizedAxis = axis < 0 ? rank + axis : axis
            guard normalizedAxis == rank - 1, rank > 0 else {
                throw PopcornError.tensorInvalidAxis(axis, rank: rank)
            }
            guard x.shape == out.shape else {
                throw PopcornError.tensorShapeMismatch("RMSNorm output shape must match input; got \(out.shape.dimensions), expected \(x.shape.dimensions).")
            }

            let hiddenSize = x.shape[normalizedAxis]
            let rowCount = x.shape.elementCount / hiddenSize

            if let weight {
                guard weight.shape.dimensions == [hiddenSize] || weight.shape == x.shape else {
                    throw PopcornError.tensorShapeMismatch("RMSNorm weight shape must be [\(hiddenSize)] or \(x.shape.dimensions); got \(weight.shape.dimensions).")
                }
            }

            try self.init(
                x: x, weight: weight, out: out,
                rowCount: rowCount, hiddenSize: hiddenSize, eps: eps,
                addOneToWeight: addOneToWeight
            )
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: x, access: .read),
                .init(tensor: weight, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        public func dispatchSize(for pipelineState: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            DispatchSize.reduction(rowCount: rowCount, n: hiddenSize, pipelineState: pipelineState)
        }

        // MARK: Private

        private let x: Tensor
        private let weight: Tensor
        private let out: Tensor
        private let rowCount: Int
        private let hiddenSize: Int
    }
}
