import Metal
import PopcornShaderTypes

public extension Kernels {
    struct Add: Kernel {
        // MARK: Lifecycle

        public init(
            a: Tensor,
            b: Tensor,
            out: Tensor,
            count: Int,
        ) throws {
            self.a = a
            self.b = b
            self.out = out
            self.count = count
            functionName = switch a.dataType {
            case .f32: "add"
            case .f16: "add_f16"
            case .bf16: "add_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported add data type: \(a.dataType).")
            }
            constants = [AddConstants(count: UInt32(count))]
        }

        public init(_ a: Tensor, _ b: Tensor, into out: Tensor) throws {
            guard a.shape == b.shape else {
                throw PopcornError.tensorShapeMismatch("Add requires matching input shapes; got \(a.shape.dimensions) and \(b.shape.dimensions).")
            }
            guard a.shape == out.shape else {
                throw PopcornError.tensorShapeMismatch("Add output shape must match inputs; got \(out.shape.dimensions), expected \(a.shape.dimensions).")
            }
            try self.init(a: a, b: b, out: out, count: a.shape.elementCount)
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: a, access: .read),
                .init(tensor: b, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        public func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            (
                MTLSize(width: (count + 3) / 4, height: 1, depth: 1),
                MTLSize(width: 256, height: 1, depth: 1)
            )
        }

        // MARK: Private

        private let a: Tensor
        private let b: Tensor
        private let out: Tensor
        private let count: Int
    }
}
