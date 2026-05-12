import Metal
import PopcornShaderTypes

public extension Kernels {
    struct BroadcastAdd: Kernel {
        // MARK: Lifecycle

        public init(
            a: Tensor,
            b: Tensor,
            out: Tensor,
            count: Int,
            bCount: Int,
        ) throws {
            self.a = a
            self.b = b
            self.out = out
            let canVec4 = count.isMultiple(of: 4) && bCount.isMultiple(of: 4)
            functionName = switch (a.dataType, b.dataType, out.dataType, canVec4) {
            case (.f32, .f32, .f32, true): "broadcast_add_vec4"
            case (.bf16, .bf16, .bf16, true): "broadcast_add_vec4_bf16"
            case (.bf16, .f32, .f32, true): "broadcast_add_vec4_bf16_f32_to_f32"
            case (.f32, .f32, .f32, false): "broadcast_add"
            case (.bf16, .bf16, .bf16, false): "broadcast_add_bf16"
            case (.bf16, .f32, .f32, false): "broadcast_add_bf16_f32_to_f32"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported broadcast add data type combination: \(a.dataType), \(b.dataType), \(out.dataType).")
            }
            constants = [BroadcastAddConstants(count: UInt32(count), bCount: UInt32(bCount))]
            let threads = canVec4 ? count / 4 : count
            dispatchGrid = MTLSize(width: threads, height: 1, depth: 1)
        }

        public init(_ a: Tensor, _ b: Tensor, into out: Tensor) throws {
            guard a.shape == out.shape else {
                throw PopcornError.tensorShapeMismatch("BroadcastAdd output shape must match lhs; got \(out.shape.dimensions), expected \(a.shape.dimensions).")
            }
            guard a.shape.elementCount.isMultiple(of: b.shape.elementCount) else {
                throw PopcornError.tensorShapeMismatch("BroadcastAdd rhs element count \(b.shape.elementCount) must divide lhs element count \(a.shape.elementCount).")
            }
            try self.init(a: a, b: b, out: out, count: a.shape.elementCount, bCount: b.shape.elementCount)
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
            (dispatchGrid, MTLSize(width: 256, height: 1, depth: 1))
        }

        // MARK: Private

        private let dispatchGrid: MTLSize
        private let a: Tensor
        private let b: Tensor
        private let out: Tensor
    }
}
