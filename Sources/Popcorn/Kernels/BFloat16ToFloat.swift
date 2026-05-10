import Metal
import PopcornShaderTypes

public extension Kernels {
    struct BFloat16ToFloat: Kernel {
        // MARK: Lifecycle

        public init(input: Tensor, output: Tensor, count: Int) {
            self.input = input
            self.output = output
            constants = [BFloat16ToFloatConstants(count: UInt32(count))]
            grid = MTLSize(width: count, height: 1, depth: 1)
        }

        public init(_ input: Tensor, into output: Tensor) throws {
            guard input.shape == output.shape else {
                throw PopcornError.tensorShapeMismatch("BFloat16ToFloat output shape must match input; got \(output.shape.dimensions), expected \(input.shape.dimensions).")
            }
            self.init(input: input, output: output, count: input.shape.elementCount)
        }

        // MARK: Public

        public let functionName: String = "bfloat16_to_float"
        public let constants: [any BitwiseCopyable]
        public let grid: MTLSize
        public let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: input, access: .read),
                .init(tensor: output, access: .write)
            ]
        }

        // MARK: Private

        private let input: Tensor
        private let output: Tensor
    }
}
