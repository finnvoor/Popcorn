import Metal
import PopcornShaderTypes

public extension Kernels {
    struct ScalarMul: Kernel {
        // MARK: Lifecycle

        public init(x: Tensor, out: Tensor, scalar: Float, count: Int) throws {
            self.x = x
            self.out = out
            functionName = switch x.dataType {
            case .f32: "scalar_mul"
            case .f16: "scalar_mul_f16"
            case .bf16: "scalar_mul_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported scalar mul data type: \(x.dataType).")
            }
            constants = [ScalarMulConstants(count: UInt32(count), scalar: scalar)]
            dispatchGrid = MTLSize(width: count, height: 1, depth: 1)
        }

        public init(_ x: Tensor, by scalar: Float, into out: Tensor) throws {
            guard x.shape == out.shape else {
                throw PopcornError.tensorShapeMismatch("ScalarMul output shape must match input; got \(out.shape.dimensions), expected \(x.shape.dimensions).")
            }
            try self.init(x: x, out: out, scalar: scalar, count: x.shape.elementCount)
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: x, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        public func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            (dispatchGrid, MTLSize(width: 256, height: 1, depth: 1))
        }

        // MARK: Private

        private let dispatchGrid: MTLSize
        private let x: Tensor
        private let out: Tensor
    }
}
