import Metal
import PopcornShaderTypes

public extension Kernels {
    /// GroupNorm over `[N, C, L]` with per-channel affine.
    /// Each group covers `C/groups` channels across all spatial positions —
    /// matches `nn.GroupNorm(num_groups, C)`.
    struct GroupNorm: DispatchKernel {
        // MARK: Lifecycle

        public init(
            _ x: Tensor,
            weight: Tensor? = nil,
            bias: Tensor? = nil,
            into out: Tensor,
            groups: Int,
            eps: Float
        ) throws {
            guard x.shape.rank == 3 else { throw PopcornError.tensorInvalidRank(expected: 3, actual: x.shape.rank) }
            guard x.shape == out.shape else {
                throw PopcornError.tensorShapeMismatch("GroupNorm output shape must match input; got \(out.shape.dimensions), expected \(x.shape.dimensions).")
            }
            let n = x.shape[0]
            let c = x.shape[1]
            let l = x.shape[2]
            guard c.isMultiple(of: groups) else {
                throw PopcornError.tensorShapeMismatch("GroupNorm: C (\(c)) must be divisible by groups (\(groups)).")
            }
            if let weight, weight.shape.dimensions != [c] {
                throw PopcornError.tensorShapeMismatch("GroupNorm: weight must be [\(c)]; got \(weight.shape.dimensions).")
            }
            if let bias, bias.shape.dimensions != [c] {
                throw PopcornError.tensorShapeMismatch("GroupNorm: bias must be [\(c)]; got \(bias.shape.dimensions).")
            }
            self.x = x
            self.weight = weight ?? x
            self.bias = bias ?? x
            self.out = out
            self.n = n
            self.groups = groups
            functionName = switch x.dataType {
            case .f32: "groupnorm"
            case .f16: "groupnorm_f16"
            case .bf16: "groupnorm_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported GroupNorm data type: \(x.dataType).")
            }
            constants = [GroupNormConstants(
                N: UInt32(n), C: UInt32(c), L: UInt32(l),
                groups: UInt32(groups),
                hasWeight: weight == nil ? 0 : 1,
                hasBias: bias == nil ? 0 : 1,
                eps: eps
            )]
            rowCount = n * groups
            elementsPerRow = (c / groups) * l
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
            // `dispatchThreads` interprets `grid` as total thread count and divides by
            // `threadgroupSize` to get the threadgroup grid. We want one TG per (g, n).
            let perTG = DispatchSize.reduction(rowCount: 1, n: elementsPerRow, pipelineState: pipelineState).threadgroupSize.width
            return (
                MTLSize(width: groups * perTG, height: n, depth: 1),
                MTLSize(width: perTG, height: 1, depth: 1)
            )
        }

        // MARK: Private

        private let x: Tensor
        private let weight: Tensor
        private let bias: Tensor
        private let out: Tensor
        private let n: Int
        private let groups: Int
        private let rowCount: Int
        private let elementsPerRow: Int
    }
}
