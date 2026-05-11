import Metal
import PopcornShaderTypes

public extension Kernels {
    struct Slice2D: Kernel {
        // MARK: Lifecycle

        public init(
            src: Tensor,
            out: Tensor,
            rowCount: Int,
            outColumnCount: Int,
            srcRowStride: Int,
            srcColumnOffset: Int,
        ) throws {
            self.src = src
            self.out = out
            functionName = switch src.dataType {
            case .f32: "slice2d"
            case .f16: "slice2d_f16"
            case .bf16: "slice2d_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported slice2d data type: \(src.dataType).")
            }
            constants = [Slice2DConstants(
                rowCount: UInt32(rowCount),
                outColumnCount: UInt32(outColumnCount),
                srcRowStride: UInt32(srcRowStride),
                srcColumnOffset: UInt32(srcColumnOffset)
            )]
            dispatchGrid = MTLSize(width: rowCount, height: outColumnCount, depth: 1)
        }

        public init(_ src: Tensor, into out: Tensor, columnOffset: Int) throws {
            guard src.shape.rank == 2 else { throw PopcornError.tensorInvalidRank(expected: 2, actual: src.shape.rank) }
            guard out.shape.rank == 2 else { throw PopcornError.tensorInvalidRank(expected: 2, actual: out.shape.rank) }
            guard out.shape[0] == src.shape[0] else {
                throw PopcornError.tensorShapeMismatch("Slice2D output row count must match source; got \(out.shape[0]), expected \(src.shape[0]).")
            }
            guard columnOffset >= 0, columnOffset + out.shape[1] <= src.shape[1] else {
                throw PopcornError.tensorShapeMismatch("Slice2D column range \(columnOffset)..<\(columnOffset + out.shape[1]) exceeds source column count \(src.shape[1]).")
            }
            try self.init(
                src: src, out: out,
                rowCount: out.shape[0], outColumnCount: out.shape[1],
                srcRowStride: src.shape[1], srcColumnOffset: columnOffset
            )
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
            (dispatchGrid, MTLSize(width: 8, height: 8, depth: 1))
        }

        // MARK: Private

        private let dispatchGrid: MTLSize
        private let src: Tensor
        private let out: Tensor
    }
}
