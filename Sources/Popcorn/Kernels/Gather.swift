import Metal
import PopcornShaderTypes

public extension Kernels {
    struct Gather: DispatchKernel {
        // MARK: Lifecycle

        public init(
            table: Tensor,
            indices: Tensor,
            out: Tensor,
            count: Int
        ) {
            self.table = table
            self.indices = indices
            self.out = out
            constants = [GatherConstants(count: UInt32(count))]
            dispatchGrid = MTLSize(width: count, height: 1, depth: 1)
        }

        public init(table: Tensor, indices: Tensor, into out: Tensor) throws {
            guard indices.shape == out.shape else {
                throw PopcornError.tensorShapeMismatch("Gather output shape must match indices; got \(out.shape.dimensions), expected \(indices.shape.dimensions).")
            }
            self.init(table: table, indices: indices, out: out, count: indices.shape.elementCount)
        }

        // MARK: Public

        public let functionName: String = "gather"
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: table, access: .read),
                .init(tensor: indices, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        public func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            (dispatchGrid, MTLSize(width: 256, height: 1, depth: 1))
        }

        // MARK: Private

        private let dispatchGrid: MTLSize
        private let table: Tensor
        private let indices: Tensor
        private let out: Tensor
    }
}
