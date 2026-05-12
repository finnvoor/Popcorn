import Metal
import PopcornShaderTypes

public extension Kernels {
    struct GeluTanh: Kernel {
        // MARK: Lifecycle

        public init(x: Tensor, out: Tensor, count: Int) throws {
            self.x = x
            self.out = out
            functionName = switch x.dataType {
            case .f32: "gelu_tanh"
            case .f16: "gelu_tanh_f16"
            case .bf16: "gelu_tanh_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported gelu tanh data type: \(x.dataType).")
            }
            constants = [GeluTanhConstants(count: UInt32(count))]
            dispatchGrid = MTLSize(width: (count + 3) / 4, height: 1, depth: 1)
        }

        public init(_ x: Tensor, into out: Tensor) throws {
            guard x.shape == out.shape else {
                throw PopcornError.tensorShapeMismatch("GeluTanh output shape must match input; got \(out.shape.dimensions), expected \(x.shape.dimensions).")
            }
            try self.init(x: x, out: out, count: x.shape.elementCount)
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
