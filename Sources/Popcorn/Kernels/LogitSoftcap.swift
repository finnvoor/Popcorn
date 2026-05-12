import Metal
import PopcornShaderTypes

public extension Kernels {
    struct LogitSoftcap: DispatchKernel {
        // MARK: Lifecycle

        public init(x: Tensor, out: Tensor, count: Int, cap: Float) throws {
            self.x = x
            self.out = out
            functionName = switch x.dataType {
            case .f32: "logit_softcap"
            case .bf16: "logit_softcap_bf16"
            case .f16: throw PopcornError.unsupportedDataTypeCombination("Unsupported logit softcap data type: \(x.dataType).")
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported logit softcap data type: \(x.dataType).")
            }
            constants = [LogitSoftcapConstants(count: UInt32(count), cap: cap)]
            dispatchGrid = MTLSize(width: count, height: 1, depth: 1)
        }

        public init(_ x: Tensor, cap: Float, into out: Tensor) throws {
            guard x.shape == out.shape else {
                throw PopcornError.tensorShapeMismatch("LogitSoftcap output shape must match input; got \(out.shape.dimensions), expected \(x.shape.dimensions).")
            }
            try self.init(x: x, out: out, count: x.shape.elementCount, cap: cap)
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
