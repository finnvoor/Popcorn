import Metal
import PopcornShaderTypes

public extension Kernels {
    struct Transpose12: Kernel {
        // MARK: Lifecycle

        public init(
            src: Tensor,
            out: Tensor,
            d0: Int, d1: Int, d2: Int, d3: Int,
        ) throws {
            self.src = src
            self.out = out
            functionName = switch src.dataType {
            case .f32: "transpose12"
            case .f16: "transpose12_f16"
            case .bf16: "transpose12_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported transpose12 data type: \(src.dataType).")
            }
            constants = [Transpose12Constants(
                D0: UInt32(d0), D1: UInt32(d1), D2: UInt32(d2), D3: UInt32(d3)
            )]
            dispatchGrid = MTLSize(width: d0 * d1 * d2 * d3, height: 1, depth: 1)
        }

        public init(_ src: Tensor, into out: Tensor) throws {
            guard src.shape.rank == 4 else { throw PopcornError.tensorInvalidRank(expected: 4, actual: src.shape.rank) }
            let d0 = src.shape[0], d1 = src.shape[1], d2 = src.shape[2], d3 = src.shape[3]
            let expectedOut = [d0, d2, d1, d3]
            guard out.shape.dimensions == expectedOut else {
                throw PopcornError.tensorShapeMismatch("Transpose12 output shape must be \(expectedOut); got \(out.shape.dimensions).")
            }
            try self.init(src: src, out: out, d0: d0, d1: d1, d2: d2, d3: d3)
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: src, access: .read),
                .init(tensor: out, access: .write)
            ]
        }

        public func dispatchSize(for _: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            (dispatchGrid, MTLSize(width: 256, height: 1, depth: 1))
        }

        // MARK: Private

        private let dispatchGrid: MTLSize
        private let src: Tensor
        private let out: Tensor
    }
}
