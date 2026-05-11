import Metal
import PopcornShaderTypes

public extension Kernels {
    struct RowSlice2D: Kernel {
        // MARK: Lifecycle

        public init(
            src: Tensor,
            out: Tensor,
            rowCount: Int,
            columnCount: Int,
            srcRowStride: Int,
            rowOffset: Int
        ) throws {
            self.src = src
            self.out = out
            functionName = switch src.dataType {
            case .f32: "row_slice2d"
            case .f16: "row_slice2d_f16"
            case .bf16: "row_slice2d_bf16"
            default: throw PopcornError.unsupportedDataTypeCombination("Unsupported row_slice2d data type: \(src.dataType).")
            }
            constants = [RowSlice2DConstants(
                rowCount: UInt32(rowCount),
                columnCount: UInt32(columnCount),
                srcRowStride: UInt32(srcRowStride),
                rowOffset: UInt32(rowOffset)
            )]
            self.rowCount = rowCount
            self.columnCount = columnCount
        }

        public init(_ src: Tensor, into out: Tensor, rowOffset: Int) throws {
            guard src.shape.rank == 2 else { throw PopcornError.tensorInvalidRank(expected: 2, actual: src.shape.rank) }
            guard out.shape.rank == 2 else { throw PopcornError.tensorInvalidRank(expected: 2, actual: out.shape.rank) }
            guard src.shape[1] == out.shape[1] else {
                throw PopcornError.tensorShapeMismatch("RowSlice2D column count must match source; got \(out.shape[1]), expected \(src.shape[1]).")
            }
            guard rowOffset >= 0, rowOffset + out.shape[0] <= src.shape[0] else {
                throw PopcornError.tensorShapeMismatch("RowSlice2D row range \(rowOffset)..<\(rowOffset + out.shape[0]) exceeds source row count \(src.shape[0]).")
            }
            try self.init(
                src: src, out: out,
                rowCount: out.shape[0], columnCount: out.shape[1],
                srcRowStride: src.shape[1], rowOffset: rowOffset
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

        public func dispatchSize(for pipelineState: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            DispatchSize.rowsColumns(rowCount: rowCount, columnCount: columnCount, pipelineState: pipelineState)
        }

        // MARK: Private

        private let rowCount: Int
        private let columnCount: Int
        private let src: Tensor
        private let out: Tensor
    }
}
