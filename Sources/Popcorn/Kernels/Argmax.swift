import Metal
import PopcornShaderTypes

public extension Kernels {
    struct Argmax: DispatchKernel {
        // MARK: Lifecycle

        public init(
            x: Tensor,
            indices: Tensor,
            rowCount: Int,
            n: Int
        ) {
            self.x = x
            self.indices = indices
            constants = [ArgmaxConstants(rows: UInt32(rowCount), N: UInt32(n))]

            self.rowCount = rowCount
            self.n = n

            functionName = n >= 32 ? "argmax_row" : "argmax"
        }

        public init(_ x: Tensor, indices: Tensor, axis: Int = -1) throws {
            let rank = x.shape.rank
            let normalizedAxis = axis < 0 ? rank + axis : axis
            guard normalizedAxis == rank - 1, rank > 0 else {
                throw PopcornError.tensorInvalidAxis(axis, rank: rank)
            }

            let n = x.shape[normalizedAxis]
            let rowCount = x.shape.elementCount / n
            guard indices.shape.elementCount == rowCount else {
                throw PopcornError.tensorShapeMismatch("Argmax indices must contain \(rowCount) elements; got \(indices.shape.elementCount).")
            }
            self.init(x: x, indices: indices, rowCount: rowCount, n: n)
        }

        // MARK: Public

        public let functionName: String
        public let constants: [any BitwiseCopyable]

        public var tensors: [Tensor.Binding] {
            [
                .init(tensor: x, access: .read),
                .init(tensor: indices, access: .write)
            ]
        }

        public func dispatchSize(for pipelineState: MTLComputePipelineState) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            Self.dispatchSize(
                functionName: functionName,
                rowCount: rowCount,
                n: n,
                threadExecutionWidth: pipelineState.threadExecutionWidth,
                maxTotalThreadsPerThreadgroup: pipelineState.maxTotalThreadsPerThreadgroup
            )
        }

        // MARK: Private

        private let x: Tensor
        private let indices: Tensor
        private let rowCount: Int
        private let n: Int

        private static func dispatchSize(
            functionName: String,
            rowCount: Int,
            n: Int,
            threadExecutionWidth: Int,
            maxTotalThreadsPerThreadgroup: Int
        ) -> (grid: MTLSize, threadgroupSize: MTLSize) {
            let simdWidth = max(1, threadExecutionWidth)
            let maxThreadgroupWidth = max(simdWidth, roundDown(min(1024, maxTotalThreadsPerThreadgroup), toMultipleOf: simdWidth))

            guard functionName == "argmax_row" else {
                let width = min(maxThreadgroupWidth, max(simdWidth, roundUp(64, toMultipleOf: simdWidth)))
                return (
                    MTLSize(width: rowCount, height: 1, depth: 1),
                    MTLSize(width: width, height: 1, depth: 1)
                )
            }

            let preferredWidth = preferredParallelThreadgroupWidth(
                rowCount: rowCount,
                n: n,
                simdWidth: simdWidth,
                maxThreadgroupWidth: maxThreadgroupWidth
            )

            let threadgroupWidth = max(simdWidth, min(preferredWidth, roundUp(nextPowerOfTwo(n), toMultipleOf: simdWidth)))
            return (
                MTLSize(width: rowCount * threadgroupWidth, height: 1, depth: 1),
                MTLSize(width: threadgroupWidth, height: 1, depth: 1)
            )
        }

        private static func preferredParallelThreadgroupWidth(
            rowCount: Int,
            n: Int,
            simdWidth: Int,
            maxThreadgroupWidth: Int
        ) -> Int {
            let simdgroupCount = if rowCount <= 4 {
                if n <= 256 { 8 }
                else if n <= 1_536 { 16 }
                else { 32 }
            } else if rowCount <= 16 {
                if n <= 256 { 4 }
                else if n <= 3_072 { 16 }
                else { 32 }
            } else if rowCount <= 64 {
                if n <= 768 { 4 }
                else if n <= 3_072 { 8 }
                else if n <= 8_192 { 16 }
                else { 8 }
            } else if rowCount <= 256 {
                if n <= 256 { 1 }
                else if n <= 3_072 { 2 }
                else if n <= 8_192 { 4 }
                else if n <= 32_768 { 32 }
                else { 2 }
            } else if rowCount <= 512 {
                if n <= 768 { 1 }
                else if n <= 1_536 { 2 }
                else if n <= 3_072 { 16 }
                else if n <= 8_192 { 8 }
                else { 32 }
            } else {
                if n <= 768 { 2 }
                else if n <= 1_536 { 8 }
                else if n <= 3_072 { 32 }
                else { 16 }
            }

            return min(maxThreadgroupWidth, simdgroupCount * simdWidth)
        }

        private static func roundDown(_ value: Int, toMultipleOf multiple: Int) -> Int {
            value / multiple * multiple
        }

        private static func roundUp(_ value: Int, toMultipleOf multiple: Int) -> Int {
            ((value + multiple - 1) / multiple) * multiple
        }

        private static func nextPowerOfTwo(_ n: Int) -> Int {
            var v = 1
            while v < n {
                v <<= 1
            }
            return v
        }
    }
}
